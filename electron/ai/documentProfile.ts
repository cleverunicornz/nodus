import { createHash, randomUUID } from 'node:crypto';
import type {
  DocumentIdeaLink,
  DocumentProfileAudit,
  DocumentProfileConfidenceSource,
  DocumentProfileFallbackMode,
  DocumentProfileFieldKind,
  DocumentProfileSupport,
  DocumentSection,
  ModelRef,
  PromptLanguage,
  Work,
} from '@shared/types';
import { getDb } from '../db/database';
import { getSettings } from '../db/settingsRepo';
import {
  advanceRunningDocumentIndexJob,
  clearDocumentCheckpoints,
  DocumentProfilePublicationError,
  publishDocumentProfile,
  readDocumentCheckpoint,
  saveDocumentCheckpoint,
  updateDocumentIndexJob,
} from '../db/documentProfilesRepo';
import { cosineSimilarity, currentEmbeddingConfig, decodeEmbedding } from '../db/ideasRepo';
import type { PassageInsert } from '../db/passagesRepo';
import {
  planRetrievalChunks,
  resolveWorkText,
  resolvedTextStateFromDoc,
  type ExtractedDoc,
  type RetrievalChunk,
} from '../extraction/textExtractor';
import {
  findLiteralSourceSpans,
  intactSourceExcerpt,
  selectLiteralPassageIndex,
  type SourceTextSpan,
} from '../extraction/sourceTextRanges';
import { setResolvedTextState } from '../db/worksRepo';
import { analysisFingerprint, analysisModelFingerprint, upsertLibraryAnalysisProvenance } from '../db/libraryAnalysisProvenance';
import { getItem, LOCAL_USER_ID } from '../zotero/zoteroClient';
import { AiError, completeJson, embedMany, estimateLocalTokens, localModelContextWindow, resolveModelRef } from './aiClient';
import { mapOrderedPool } from './orderedPool';
import { modelRefSupportsCapability } from '@shared/localAiModels';
import type { PerfContext } from '../perf';
import { documentProfilePromptPack } from '@shared/academicPromptPacks';
import {
  DOCUMENT_PROFILE_PIPELINE_VERSION,
  DOCUMENT_PROFILE_SCHEMA_VERSION,
} from '@shared/documentProfilePolicy';
import { logPipelineSuccess } from '../logging/pipelineLogCore';

export { DOCUMENT_PROFILE_PIPELINE_VERSION, DOCUMENT_PROFILE_SCHEMA_VERSION };

export class DocumentSourceUnavailableError extends Error {
  readonly code = 'no_legible_text';

  constructor(
    message: string,
    readonly sourceType: ExtractedDoc['sourceType'],
    readonly blockReason: ExtractedDoc['blockReason'] = null,
  ) {
    super(message);
    this.name = 'DocumentSourceUnavailableError';
  }
}
const ANALYSIS_WORDS = 2_500;
const MIN_SECTION_WORDS = 80;
const DIRECT_SUPPORT_CONFIDENCE_FLOOR = 0.8;
/**
 * Minimum semantic score the auditor must report for a synthesis to publish with no
 * caveat. It is a different decision from the direct-support floor above even though
 * both values are 0.8: the floor says what a literally supported field is worth, this
 * says how much the auditor must like the prose. Sharing one constant made a rejected
 * synthesis and a perfect one report the same number, and made "0.8" look like a score.
 */
const SEMANTIC_ACCEPTANCE_SCORE = 0.8;
const CENTRAL_FIELD_KINDS = new Set<DocumentProfileFieldKind>([
  'problem', 'question', 'hypothesis', 'thesis', 'method', 'finding', 'conclusion', 'contribution',
]);

export interface DerivedDocumentSection extends DocumentSection {
  body: string;
}

interface RawClaim { text: string; support_quote: string; page: string | null; confidence: number }
/** `degraded` marks an analysis that fell back to literal source text because no model
 *  synthesis survived its own section audit. Absent means synthesised; it is not persisted
 *  as such, but the scan counts it so a published profile can say how much of it is
 *  quotation. */
interface SectionAnalysis { title: string; summary: string; role: string; concepts: string[]; claims: RawClaim[]; degraded?: boolean }
interface RawProfileField {
  handle?: string;
  kind: DocumentProfileFieldKind;
  text: string;
  confidence: number;
  centrality: number;
  support_quote: string;
  page: string | null;
  /** Set when the published confidence is the deterministic floor rather than a
   * value the provider measured (see `retainLiterallySupportedFields`). */
  confidenceSource?: DocumentProfileConfidenceSource;
}
interface ProfileSynthesis { source_language: string; overview: string; fields: RawProfileField[] }
type AuditTargetKind = 'overview' | 'field' | 'section' | 'claim' | 'profile';
interface AuditIssue {
  code: string;
  blocking: boolean;
  target: { kind: AuditTargetKind; handle: string | null };
  explanation: string;
}
type FieldOperationKind = 'remove_field' | 'edit_field' | 'add_field';
interface FieldOperation {
  op: FieldOperationKind;
  target: string | null;
  kind: DocumentProfileFieldKind | null;
  text: string | null;
  supportQuote: string | null;
  evidenceHandle: string | null;
  confidence: number | null;
  centrality: number | null;
}
interface AuditResponse {
  candidateRevision: string;
  passed: boolean;
  /** null when the provider reported no usable score: "no reading", not "scored zero". */
  score: number | null;
  issues: AuditIssue[];
  operations: FieldOperation[];
  overview: string | null;
}
interface RepairDelta {
  baseRevision: string;
  operations: FieldOperation[];
  overview: string | null;
}
interface EvidenceHandle {
  handle: string;
  sectionId: string;
  quote: string;
  page: string | null;
}
interface SectionAuditResponse { passed: boolean; issues: string[]; analysis: SectionAnalysis | null }
interface PreparedPassagePublication {
  contentHash: string;
  rows: PassageInsert[];
  embeddingProvider: string;
  embeddingModel: string;
}
interface PreparedPassages {
  rows: RetrievalChunk[];
  publication: PreparedPassagePublication | null;
}

function isSectionAuditResponse(value: unknown): value is SectionAuditResponse {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

export interface DocumentProfileScanProgress {
  phase: 'waiting_source' | 'structuring' | 'analyzing_sections' | 'synthesizing' | 'auditing' | 'repairing' | 'embedding' | 'aligning' | 'publishing';
  progress: number;
  message: string;
  currentUnit?: number;
  totalUnits?: number;
}

export interface RunDocumentProfileOptions {
  jobId: string;
  generatorModel: ModelRef | null;
  auditorModel: ModelRef | null;
  signal?: AbortSignal;
  onProgress?: (progress: DocumentProfileScanProgress) => void;
  /** Audit-only timing context; never contains document text. */
  perf?: PerfContext;
  language?: PromptLanguage;
  /** Prompt budget in tokens for the model that will read it, or null for a cloud model
   *  whose window is managed server-side. Local servers load a small fixed window and
   *  reject a prompt that does not fit, so every prompt this pipeline builds is sized
   *  against it instead of discovering the limit through a failed request. */
  promptTokenBudget?: number | null;
  /** The same budget for the auditor model, which is usually a different one. */
  auditorTokenBudget?: number | null;
}

const clean = (value: unknown, max = 20_000): string => typeof value === 'string'
  ? value.replace(/\s+/g, ' ').trim().slice(0, max) : '';
const strings = (value: unknown, max = 24): string[] => Array.isArray(value)
  ? value.map((item) => clean(item, 500)).filter(Boolean).slice(0, max) : [];
const number01 = (value: unknown): number => Math.max(0, Math.min(1, Number(value) || 0));

/**
 * Read a provider verdict. The schema asks for a JSON boolean, but providers —
 * especially small and local models — answer with the affirmatives of their own
 * language and with 1/0, and a JSON `1` is not a JSON `true`. Reading those as a
 * rejection used to discard an entire audited synthesis, so anything explicit and
 * positive counts as an approval; an unrecognised or absent value stays false,
 * because a missing verdict must never be promoted to passed.
 */
function verdictPassed(value: unknown): boolean {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return value !== 0;
  if (typeof value !== 'string') return false;
  const normalized = value.trim().toLowerCase().replace(/[.!。]$/, '');
  return [
    'true', '1', 'yes', 'y', 'sí', 'si', 'verdadero', 'verdadera', 'correcto', 'aprobado',
    'oui', 'ja', 'sim', 'vero', 'doğru', 'evet',
    'да', 'так', '예', '네', 'はい', '是', '对', '對', 'đúng',
  ].includes(normalized);
}

/**
 * Read a provider score as a 0-1 fraction. Providers report `0.85`, `"0.85"`,
 * `"85%"`, `"0,85"` and `85` for the same judgement, and `number01` turned the
 * last three into a zero, which failed the acceptance gate on a formatting quirk.
 * Returns null when nothing usable was reported, so a missing score is never read
 * as "scored zero" — callers distinguish "no reading" from "a low reading".
 */
function scoreFraction(value: unknown): number | null {
  const raw = typeof value === 'number' ? value
    : typeof value === 'string' ? Number(value.trim().replace(/\s*%\s*$/, '').replace(',', '.'))
    : Number.NaN;
  if (!Number.isFinite(raw)) return null;
  // A value above 1 cannot be a 0-1 fraction, so it is a percentage out of a hundred.
  const fraction = raw > 1 && raw <= 100 ? raw / 100 : raw;
  return Math.max(0, Math.min(1, fraction));
}
const page = (value: unknown): string | null => {
  const match = clean(value, 30).match(/(?:p\.?|page|página)\s*(\d+)/i);
  return match ? `p. ${match[1]}` : null;
};
const sha256 = (value: string): string => createHash('sha256').update(value).digest('hex');
const sha1 = (value: string): string => createHash('sha1').update(value).digest('hex');
export function documentProfileCheckpointFingerprint(input: {
  pipelineVersion: string;
  kind: string;
  language: PromptLanguage;
  generatorModel: ModelRef | null;
  auditorModel: ModelRef | null;
  payload: unknown;
}): string {
  return sha256(JSON.stringify(input));
}

function profileCheckpointHash(kind: string, payload: unknown, options: RunDocumentProfileOptions): string {
  return documentProfileCheckpointFingerprint({
    pipelineVersion: DOCUMENT_PROFILE_PIPELINE_VERSION,
    kind,
    language: options.language ?? getSettings().promptLanguage ?? 'es',
    generatorModel: options.generatorModel,
    auditorModel: options.auditorModel,
    payload,
  });
}


function isSectionAnalysis(value: unknown): value is SectionAnalysis {
  // Providers commonly omit optional empty arrays or wrap the requested object.
  // Accept only an object here and let the conservative normalizer plus literal
  // quote matching and the independent audit reject unsupported content.
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

export function normalizeSectionAnalysis(value: unknown, fallbackTitle: string): SectionAnalysis {
  const root = value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
  const nested = [root.section_analysis, root.analysis].find(
    (candidate) => candidate && typeof candidate === 'object' && !Array.isArray(candidate),
  ) as Record<string, unknown> | undefined;
  const item = nested ?? root;
  return {
    title: clean(item.title, 300) || fallbackTitle,
    summary: clean(item.summary, 4_000),
    role: clean(item.role, 300),
    concepts: strings(item.concepts),
    claims: (Array.isArray(item.claims) ? item.claims : []).flatMap((entry) => {
      if (!entry || typeof entry !== 'object') return [];
      const claim = entry as unknown as Record<string, unknown>;
      const text = clean(claim.text, 1_500);
      const quote = clean(claim.support_quote, 1_200);
      return text && quote ? [{ text, support_quote: quote, page: page(claim.page), confidence: number01(claim.confidence) }] : [];
    }).slice(0, 16),
  };
}

function normalizeDirectSupportConfidence(analysis: SectionAnalysis): SectionAnalysis {
  return {
    ...analysis,
    claims: analysis.claims.map((claim) => ({
      ...claim,
      // Confidence is evidence metadata, not an unchecked model opinion. Once a
      // claim has survived literal matching and the independent section audit,
      // zero is internally contradictory and makes the document auditor reject
      // otherwise valid evidence at random.
      confidence: Math.max(DIRECT_SUPPORT_CONFIDENCE_FLOOR, claim.confidence),
    })),
  };
}

function normalizeSectionAuditResponse(value: unknown, fallbackTitle: string): SectionAuditResponse {
  const root = value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
  const nested = root.section_audit && typeof root.section_audit === 'object' && !Array.isArray(root.section_audit)
    ? root.section_audit as Record<string, unknown>
    : null;
  const item = nested ?? root;
  const rawPassed = item.passed;
  const passed = verdictPassed(rawPassed);
  const rawIssues = Array.isArray(item.issues) ? item.issues : item.issues == null ? [] : [item.issues];
  const candidate = item.analysis ?? item.corrected_analysis;
  const analysis = candidate && typeof candidate === 'object' && !Array.isArray(candidate)
    ? normalizeSectionAnalysis(candidate as SectionAnalysis, fallbackTitle)
    : null;
  return { passed, issues: rawIssues.map((issue) => clean(issue, 1_000)).filter(Boolean).slice(0, 30), analysis };
}

function isProfileSynthesis(value: unknown): value is ProfileSynthesis {
  // The synthesis/repair pass is followed by stricter deterministic checks. Let
  // those checks reject an incomplete profile with a useful quality error instead
  // of failing early because a provider omitted an empty array or wrapped the
  // requested object in `profile`.
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

const FIELD_KINDS = new Set<DocumentProfileFieldKind>([
  'object','problem','question','hypothesis','thesis','argument','method','sources','concept','temporal_scope',
  'geographic_scope','disciplinary_scope','structure','finding','conclusion','contribution','limitation',
  'genre','audience','positioning','original_abstract',
]);

function normalizeProfile(value: unknown): ProfileSynthesis {
  const root = value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
  const nested = root.profile && typeof root.profile === 'object' && !Array.isArray(root.profile)
    ? root.profile as Record<string, unknown>
    : null;
  const item = nested ?? root;
  return {
    source_language: clean(item.source_language, 20) || 'und',
    overview: clean(item.overview, 5_000),
    fields: (Array.isArray(item.fields) ? item.fields : []).flatMap((entry) => {
      if (!entry || typeof entry !== 'object') return [];
      const field = entry as unknown as Record<string, unknown>;
      const kind = clean(field.kind, 40) as DocumentProfileFieldKind;
      const text = clean(field.text, 3_000);
      const support = clean(field.support_quote, 1_200);
      if (!FIELD_KINDS.has(kind) || !text || !support) return [];
      return [{ kind, text, support_quote: support, page: page(field.page), confidence: number01(field.confidence), centrality: number01(field.centrality), confidenceSource: 'model' as const }];
    }).slice(0, 80),
  };
}

function isAuditResponse(value: unknown): value is AuditResponse {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function strictOptionalString(value: unknown, max: number): string | null {
  if (value == null || value === '') return null;
  if (typeof value !== 'string') return null;
  const normalized = value.replace(/\s+/g, ' ').trim();
  return normalized && normalized.length <= max ? normalized : null;
}

function normalizeAuditIssue(value: unknown): AuditIssue | null {
  if (typeof value === 'string') {
    const explanation = strictOptionalString(value, 1_000);
    return explanation ? {
      code: 'legacy_unlocalized_issue',
      blocking: true,
      target: { kind: 'profile', handle: null },
      explanation,
    } : null;
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const item = value as Record<string, unknown>;
  const rawTarget = item.target && typeof item.target === 'object' && !Array.isArray(item.target)
    ? item.target as Record<string, unknown>
    : {};
  const kind = strictOptionalString(rawTarget.kind, 20) as AuditTargetKind | null;
  if (!kind || !['overview', 'field', 'section', 'claim', 'profile'].includes(kind)) return null;
  const explanation = strictOptionalString(item.explanation, 1_000);
  if (!explanation) return null;
  return {
    code: strictOptionalString(item.code, 80) ?? 'audit_issue',
    blocking: item.blocking !== false,
    target: {
      kind,
      handle: kind === 'overview' || kind === 'profile' ? null : strictOptionalString(rawTarget.handle, 200),
    },
    explanation,
  };
}

function normalizeFieldOperation(value: unknown): FieldOperation | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const item = value as Record<string, unknown>;
  const op = strictOptionalString(item.op, 30) as FieldOperationKind | null;
  if (!op || !['remove_field', 'edit_field', 'add_field'].includes(op)) return null;
  const kind = strictOptionalString(item.kind, 40) as DocumentProfileFieldKind | null;
  const target = strictOptionalString(item.target, 200);
  const text = strictOptionalString(item.text, 3_000);
  const supportQuote = strictOptionalString(item.support_quote, 1_200);
  const evidenceHandle = strictOptionalString(item.evidence_handle, 240);
  if (item.kind != null && (!kind || !FIELD_KINDS.has(kind))) return null;
  if (item.target != null && !target) return null;
  if (item.text != null && !text) return null;
  if (item.support_quote != null && !supportQuote) return null;
  if (item.evidence_handle != null && !evidenceHandle) return null;
  return {
    op, target, kind, text, supportQuote, evidenceHandle,
    confidence: item.confidence == null ? null : number01(item.confidence),
    centrality: item.centrality == null ? null : number01(item.centrality),
  };
}

/** Normalize provider JSON without ever promoting an ambiguous or stale audit. */
export function normalizeDocumentProfileAuditResponse(
  value: unknown,
  expectedRevision = '',
  fieldHandles: string[] = [],
): AuditResponse {
  const root = value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
  const nested = root.audit && typeof root.audit === 'object' && !Array.isArray(root.audit)
    ? root.audit as Record<string, unknown>
    : null;
  const item = nested ?? root;
  const candidateRevision = strictOptionalString(item.candidate_revision, 128) ?? '';
  const rawIssues = Array.isArray(item.issues) ? item.issues : item.issues == null ? [] : [item.issues];
  const issues = rawIssues.map(normalizeAuditIssue).filter((issue): issue is AuditIssue => !!issue).slice(0, 50);
  const rawOperations = Array.isArray(item.operations) ? item.operations : [];
  const parsedOperations = rawOperations.map(normalizeFieldOperation);
  const invalidOperations = rawOperations.length > 32 || parsedOperations.some((operation) => !operation);
  const operations = invalidOperations
    ? []
    : parsedOperations.filter((operation): operation is FieldOperation => !!operation).slice(0, 32);
  if (invalidOperations) {
    issues.unshift({
      code: 'invalid_repair_operations',
      blocking: true,
      target: { kind: 'profile', handle: null },
      explanation: 'One or more proposed repair operations were invalid or exceeded the bounded operation limit.',
    });
  }
  // Read old provider/test fixtures conservatively during the protocol cutover.
  const legacyFixes = (Array.isArray(item.field_fixes) ? item.field_fixes : []).flatMap((entry) => {
    if (!entry || typeof entry !== 'object') return [];
    const fix = entry as Record<string, unknown>;
    const index = Math.trunc(Number(fix.index));
    const target = fieldHandles[index];
    if (!Number.isFinite(index) || index < 0 || !target) return [];
    return [{
      op: 'edit_field' as const,
      target,
      kind: null,
      text: strictOptionalString(fix.text, 3_000),
      supportQuote: strictOptionalString(fix.support_quote, 1_200),
      evidenceHandle: strictOptionalString(fix.evidence_handle, 240),
      confidence: null,
      centrality: null,
    }];
  });
  if (expectedRevision && candidateRevision !== expectedRevision) {
    issues.unshift({
      code: 'stale_candidate_revision',
      blocking: true,
      target: { kind: 'profile', handle: null },
      explanation: 'The audit verdict did not echo the exact candidate revision.',
    });
  }
  if (item.overview != null && !strictOptionalString(item.overview, 5_000)) {
    issues.unshift({
      code: 'invalid_overview_fix',
      blocking: true,
      target: { kind: 'overview', handle: null },
      explanation: 'The proposed overview replacement is empty, invalid, or oversized.',
    });
  }
  return {
    candidateRevision,
    passed: verdictPassed(item.passed) && (!expectedRevision || candidateRevision === expectedRevision),
    score: scoreFraction(item.score),
    issues,
    operations: [...operations, ...legacyFixes].slice(0, 32),
    overview: strictOptionalString(item.overview, 5_000),
  };
}

function normalizeRepairDelta(value: unknown): RepairDelta | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const item = value as Record<string, unknown>;
  const baseRevision = strictOptionalString(item.base_revision, 128);

  if (!baseRevision) return null;
  const rawOperations = Array.isArray(item.operations) ? item.operations : [];
  if (rawOperations.length > 16) return null;
  const operations = rawOperations.map(normalizeFieldOperation);
  if (operations.some((operation) => !operation)) return null;
  const overview = strictOptionalString(item.overview, 5_000);
  if (item.overview != null && !overview) return null;
  return {
    baseRevision,
    operations: operations as FieldOperation[],
    overview,
  };
}
function isRepairDeltaResponse(value: unknown): value is RepairDelta {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function withFieldHandles(profile: ProfileSynthesis, seed: string): ProfileSynthesis {
  return {
    ...profile,
    fields: profile.fields.map((field, index) => ({
      ...field,
      handle: field.handle
        ?? `field-${sha256(`${seed}|${index}|${field.kind}|${field.text}|${field.support_quote}`).slice(0, 20)}`,
    })),
  };
}

function profileCandidateRevision(profile: ProfileSynthesis, sections: DerivedDocumentSection[]): string {
  return sha256(JSON.stringify({
    pipeline: DOCUMENT_PROFILE_PIPELINE_VERSION,
    overview: profile.overview,
    fields: profile.fields.map((field) => ({
      handle: field.handle, kind: field.kind, text: field.text,
      support_quote: field.support_quote, page: field.page,
      confidence: field.confidence, centrality: field.centrality,
    })),
    sections: sections.map((section) => ({
      id: section.sectionId, title: section.title, role: section.role, summary: section.summary,
      concepts: section.concepts, claims: section.claims,
    })),
  }));
}

function sectionEvidenceInventory(
  sections: DerivedDocumentSection[],
  analyses: Map<string, SectionAnalysis>,
): EvidenceHandle[] {
  return sections.flatMap((section) => (analyses.get(section.sectionId)?.claims ?? []).map((claim, index) => ({
    handle: `${section.sectionId}:claim:${index}`,
    sectionId: section.sectionId,
    quote: claim.support_quote,
    page: claim.page,
  })));
}

function applyRepairDelta(input: {
  profile: ProfileSynthesis;
  delta: RepairDelta;
  expectedRevision: string;
  evidence: EvidenceHandle[];
  sections: DerivedDocumentSection[];
}): { profile: ProfileSynthesis; changed: boolean; errors: string[] } {
  if (input.delta.baseRevision !== input.expectedRevision) {
    return { profile: input.profile, changed: false, errors: ['stale_revision'] };
  }
  const fields = input.profile.fields.map((field) => ({ ...field }));
  const byHandle = new Map(fields.map((field) => [field.handle ?? '', field]));
  const evidence = new Map(input.evidence.map((entry) => [entry.handle, entry]));
  const sections = new Map(input.sections.map((section) => [section.sectionId, section]));
  const touched = new Set<string>();
  const errors: string[] = [];
  let changed = false;

  for (let index = 0; index < input.delta.operations.length; index += 1) {
    const operation = input.delta.operations[index];
    if (operation.op !== 'add_field') {
      if (!operation.target || touched.has(operation.target)) { errors.push('conflicting_target'); continue; }
      touched.add(operation.target);
      if (!byHandle.has(operation.target)) { errors.push('unknown_target'); continue; }
    }
    if (operation.op === 'remove_field') {
      changed = true;
      continue;
    }
    const current = operation.op === 'edit_field' ? byHandle.get(operation.target ?? '') ?? null : null;
    const kind = operation.kind ?? current?.kind ?? null;
    const text = operation.text ?? current?.text ?? null;
    const supportQuote = operation.supportQuote ?? current?.support_quote ?? null;
    if (!kind || !FIELD_KINDS.has(kind) || !text || !supportQuote) {
      errors.push('invalid_field_shape');
      continue;
    }
    const quoteChanged = !current || supportQuote !== current.support_quote;
    let pageValue = current?.page ?? null;
    if (quoteChanged || operation.evidenceHandle) {
      const source = operation.evidenceHandle ? evidence.get(operation.evidenceHandle) : null;
      const section = source ? sections.get(source.sectionId) : null;
      if (!source || !section
        || !source.quote.includes(supportQuote)
        || quoteOffset(section.body, supportQuote) < 0) {
        errors.push('invalid_evidence');
        continue;
      }
      pageValue = source.page ?? section.pageStart;
    }
    if (current) {
      const before = { ...current };
      Object.assign(current, {
        kind, text, support_quote: supportQuote, page: pageValue,
        confidence: operation.confidence ?? current.confidence,
        centrality: operation.centrality ?? current.centrality,
      });
      changed = changed
        || kind !== before.kind
        || text !== before.text
        || supportQuote !== before.support_quote
        || current.confidence !== before.confidence
        || current.centrality !== before.centrality;
    } else {
      if (fields.length >= 80) { errors.push('field_limit'); continue; }
      const handle = `field-${sha256(`${input.expectedRevision}|add|${index}|${kind}|${text}|${supportQuote}`).slice(0, 20)}`;
      const added: RawProfileField = {
        handle, kind, text, support_quote: supportQuote, page: pageValue,
        confidence: operation.confidence ?? DIRECT_SUPPORT_CONFIDENCE_FLOOR,
        centrality: operation.centrality ?? 0.5,
        confidenceSource: operation.confidence == null ? 'floor' : 'model',
      };
      fields.push(added);
      byHandle.set(handle, added);
      changed = true;
    }
  }
  if (errors.length) return { profile: input.profile, changed: false, errors };
  const removed = new Set(input.delta.operations
    .filter((operation) => operation.op === 'remove_field' && operation.target)
    .map((operation) => operation.target as string));
  const overview = input.delta.overview ?? input.profile.overview;
  changed = changed || overview !== input.profile.overview || removed.size > 0;
  return {
    profile: { ...input.profile, overview, fields: fields.filter((field) => !removed.has(field.handle ?? '')) },
    changed,
    errors: [],
  };
}

export function applyDocumentProfileRepairDeltaForTest(input: {
  profile: ProfileSynthesis;
  delta: RepairDelta;
  expectedRevision: string;
  evidence: EvidenceHandle[];
  sections: DerivedDocumentSection[];
}): { profile: ProfileSynthesis; changed: boolean; errors: string[] } {
  return applyRepairDelta(input);
}


function auditFailureMessage(audit: DocumentProfileAudit): string {
  const details = [
    ...audit.issues,
    `veredicto=${audit.passed ? 'aprobado' : 'rechazado'}`,
    `puntuación=${audit.score == null ? 'sin puntuación' : audit.score.toFixed(2)}`,
    `apoyos=${audit.supportCoverage.toFixed(2)}`,
    `estructura=${audit.structureCoverage.toFixed(2)}`,
  ];
  return details.join(' · ');
}

/** A synthesis publishes without a caveat only when the auditor approved it and its
 *  score cleared the acceptance bar. A missing score is not a zero: the verdict is
 *  then decided by `passed` alone, and the profile is published with its own mode. */
function semanticApproved(verdict: AuditResponse | null): boolean {
  if (!verdict?.passed || verdict.issues.some((issue) => issue.blocking)) return false;
  return verdict.score == null || verdict.score >= SEMANTIC_ACCEPTANCE_SCORE;
}

interface SourceLocation {
  label: string | null;
  sourceRef: string | null;
  pageNumber: number | null;
}

function parseSourceLocationAt(text: string, offset: number, sourceMap: Record<string, string> = {}): SourceLocation {
  let found: RegExpExecArray | null = null;
  const pattern = /\[\[(?:src:(s\d+)(?:\s+p\.\s*(\d+))?|p\.\s*(\d+))\]\]/gi;
  for (const match of text.matchAll(pattern)) {
    if ((match.index ?? 0) > offset) break;
    found = match as RegExpExecArray;
  }
  if (!found) return { label: null, sourceRef: null, pageNumber: null };
  const marker = found[1] ?? null;
  const pageNumber = Number(found[2] ?? found[3]) || null;
  return {
    label: pageNumber == null ? null : `p. ${pageNumber}`,
    sourceRef: marker == null ? null : sourceMap[marker] ?? marker,
    pageNumber,
  };
}

function headingMatches(text: string, sourceMap: Record<string, string>): Array<{ index: number; end: number; level: number; title: string; location: SourceLocation }> {
  const result: Array<{ index: number; end: number; level: number; title: string; location: SourceLocation }> = [];
  const pattern = /^(#{1,6})[ \t]+([^\n]+)$/gm;
  for (const match of text.matchAll(pattern)) {
    const index = match.index ?? 0;
    result.push({ index, end: index + match[0].length, level: match[1].length, title: clean(match[2], 300), location: parseSourceLocationAt(text, index, sourceMap) });
  }
  return result;
}

function chunksWithOffsets(text: string, wordsPerChunk = 3_500): Array<{ start: number; end: number; body: string }> {
  const words = [...text.matchAll(/\S+/g)];
  if (!words.length) return [];
  const chunks: Array<{ start: number; end: number; body: string }> = [];
  for (let startWord = 0; startWord < words.length; startWord += wordsPerChunk) {
    const endWord = Math.min(words.length, startWord + wordsPerChunk);
    const start = words[startWord].index ?? 0;
    const last = words[endWord - 1];
    const end = (last.index ?? 0) + last[0].length;
    chunks.push({ start, end, body: text.slice(start, end) });
  }
  return chunks;
}

/** Pure, stable structural pass reused by tests and the scanner. */
export function deriveDocumentStructure(text: string, fallbackTitle: string, sourceMap: Record<string, string> = {}): DerivedDocumentSection[] {
  const headings = headingMatches(text, sourceMap);
  if (headings.length === 0) {
    return chunksWithOffsets(text).map((chunk, ordinal) => {
      const start = parseSourceLocationAt(text, chunk.start, sourceMap);
      const end = parseSourceLocationAt(text, chunk.end, sourceMap);
      return ({
      sectionId: `section-${sha256(`${fallbackTitle}|${ordinal}|${sha256(chunk.body)}`).slice(0, 24)}`,
      parentSectionId: null, level: 1, ordinal, title: ordinal === 0 ? fallbackTitle : '',
      role: null, summary: '', concepts: [], claims: [], pageStart: start.label,
      pageEnd: end.label, sourceRef: start.sourceRef ?? end.sourceRef,
      pageStartNumber: start.pageNumber, pageEndNumber: end.pageNumber,
      charStart: chunk.start, charEnd: chunk.end,
      contentHash: sha256(chunk.body), body: chunk.body,
      });
    });
  }
  const sections: DerivedDocumentSection[] = [];
  const parents: Array<{ level: number; id: string }> = [];
  const preamble = headings[0].index > 0 ? text.slice(0, headings[0].index) : '';
  const keepsPreambleSection = headings[0].index > 0 && preamble.split(/\s+/).length >= MIN_SECTION_WORDS;
  if (keepsPreambleSection) {
    const body = preamble;
    sections.push({
      sectionId: `section-${sha256(`${fallbackTitle}|front|${sha256(body)}`).slice(0, 24)}`,
      parentSectionId: null, level: 1, ordinal: 0, title: fallbackTitle, role: null, summary: '',
      concepts: [], claims: [], pageStart: parseSourceLocationAt(text, 0, sourceMap).label,
      pageEnd: parseSourceLocationAt(text, headings[0].index, sourceMap).label,
      sourceRef: parseSourceLocationAt(text, headings[0].index, sourceMap).sourceRef,
      pageStartNumber: parseSourceLocationAt(text, 0, sourceMap).pageNumber,
      pageEndNumber: parseSourceLocationAt(text, headings[0].index, sourceMap).pageNumber,
      charStart: 0, charEnd: headings[0].index, contentHash: sha256(body), body,
    });
  }
  // A short preamble (a title block, an author list) is not worth a section of its
  // own, but it still belongs to the document. Leaving it out of every section left
  // a hole that structure coverage counted as missing text, so an otherwise perfect
  // profile was rejected — and the literal fallback with it — for a reason no model
  // can influence. The first section absorbs it instead: every character of the
  // document stays inside exactly one section, whichever branch runs.
  const absorbPreamble = headings[0].index > 0 && !keepsPreambleSection && preamble.trim().length > 0;
  for (let index = 0; index < headings.length; index += 1) {
    const heading = headings[index];
    const end = headings[index + 1]?.index ?? text.length;
    const absorb = index === 0 && absorbPreamble;
    // A section covers its heading too: skip only the heading's own characters when
    // slicing the body, never when recording the range.
    const body = text.slice(absorb ? 0 : heading.end, end).trim();
    if (!body) continue;
    while (parents.length && parents.at(-1)!.level >= heading.level) parents.pop();
    const startLocation = absorb ? parseSourceLocationAt(text, 0, sourceMap) : heading.location;
    const endLocation = parseSourceLocationAt(text, end, sourceMap);
    const sectionId = `section-${sha256(`${heading.level}|${heading.title}|${startLocation.label ?? ''}|${index}|${sha256(body)}`).slice(0, 24)}`;
    sections.push({
      sectionId, parentSectionId: parents.at(-1)?.id ?? null, level: heading.level,
      ordinal: sections.length, title: heading.title, role: null, summary: '', concepts: [], claims: [],
      pageStart: startLocation.label, pageEnd: endLocation.label,
      sourceRef: startLocation.sourceRef ?? endLocation.sourceRef,
      pageStartNumber: startLocation.pageNumber, pageEndNumber: endLocation.pageNumber,
      charStart: absorb ? 0 : heading.index, charEnd: end,
      contentHash: sha256(body), body,
    });
    parents.push({ level: heading.level, id: sectionId });
  }
  return sections;
}

/**
 * How many tokens one background request may spend on its prompt before the model's own
 * window is at risk. Cloud models return null and keep their current sizing. The 60 %
 * leaves room for the answer, which for these calls is a JSON document of the same order
 * as the request.
 */
async function localPromptTokenBudget(model: ModelRef | null): Promise<number | null> {
  try {
    // The pipeline passes an override or null for "the configured model", and it is that
    // model whose window has to be respected.
    const window = await localModelContextWindow(model ?? resolveModelRef(null));
    if (!window) return null;
    return Math.max(1_000, Math.floor(window * 0.6));
  } catch {
    // A provider that cannot be interrogated is treated as a cloud model: the previous
    // behaviour, rather than a guess that could shrink every request.
    return null;
  }
}

/** Prompt cost of one request, system pack included. Measuring only the payload would
 *  leave the instruction pack — the larger half for the smallest windows — uncounted. */
function promptTokens(system: string, body: unknown): number {
  return estimateLocalTokens(system) + estimateLocalTokens(typeof body === 'string' ? body : JSON.stringify(body));
}

function promptFits(system: string, body: unknown, budget: number | null | undefined): boolean {
  return budget == null || promptTokens(system, body) <= budget;
}

function splitAnalysisParts(body: string, tokenBudget?: number | null): string[] {
  let wordsPerChunk = ANALYSIS_WORDS;
  // A local model with a small window would otherwise get a 2,500-word fragment it
  // cannot hold. Scale the fragment to the window using the text's own density.
  if (tokenBudget != null) {
    const words = body.split(/\s+/).filter(Boolean).length;
    const tokens = estimateLocalTokens(body);
    if (words > 0 && tokens > tokenBudget) {
      wordsPerChunk = Math.max(120, Math.min(ANALYSIS_WORDS, Math.floor((words * tokenBudget) / tokens)));
    }
  }
  return chunksWithOffsets(body, wordsPerChunk).map((chunk) => chunk.body);
}

function providerShapeFailure(error: unknown): boolean {
  return (error instanceof AiError && !error.retriable && !error.config
      && (error.code === 'invalid_json' || /json|esquema/i.test(error.message)))
    || error instanceof SyntaxError;
}

function structuredOutputFailure(error: unknown): boolean {
  return providerShapeFailure(error)
    || (error instanceof AiError && error.code === 'output_truncated');
}

/** A prompt the model could not hold. It is a sizing problem rather than a bad answer, so
 *  every stage here can answer it by shrinking (a smaller fragment, a compact audit) or by
 *  degrading locally — never by failing a whole work because its model loaded a small window. */
function contextOverflow(error: unknown): boolean {
  if (!(error instanceof AiError)) return false;
  return error.code === 'context_overflow'
    || /not enough context|context length|context window|n_ctx|tokens to keep|maximum context|suficiente contexto/i.test(error.message);
}

function recoverablePromptFailure(error: unknown): boolean {
  return structuredOutputFailure(error) || contextOverflow(error);
}

function describeFailure(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function mergeSectionAnalyses(values: SectionAnalysis[], fallbackTitle: string): SectionAnalysis {
  const claims = new Map<string, RawClaim>();
  for (const claim of values.flatMap((value) => value.claims)) {
    const existing = claims.get(claim.support_quote);
    if (!existing || existing.confidence < claim.confidence) claims.set(claim.support_quote, claim);
  }
  return {
    title: values.find((value) => value.title.trim())?.title ?? fallbackTitle,
    summary: clean(values.map((value) => value.summary).filter(Boolean).join(' '), 4_000),
    role: clean(values.map((value) => value.role).find(Boolean), 300),
    concepts: [...new Set(values.flatMap((value) => value.concepts).filter(Boolean))].slice(0, 24),
    claims: [...claims.values()].slice(0, 16),
    // Only a merge with no synthesised part at all is a quotation list.
    degraded: values.length > 0 && values.every((value) => value.degraded),
  };
}

function literalSectionFallback(evidence: string, title: string): SectionAnalysis {
  const quote = clean(intactSourceExcerpt(evidence, 900), 900);
  const claims: RawClaim[] = quote ? [{
    text: quote,
    support_quote: quote,
    page: null,
    confidence: DIRECT_SUPPORT_CONFIDENCE_FLOOR,
  }] : [];
  return { title, summary: quote, role: '', concepts: [], claims, degraded: true };
}

async function auditSectionAnalysis(
  evidence: string,
  candidate: SectionAnalysis,
  fallbackTitle: string,
  options: RunDocumentProfileOptions,
  splitDepth = 0,
  initialIssues: string[] = [],
): Promise<SectionAnalysis> {
  let current = { ...candidate, claims: candidate.claims.filter((claim) => quoteOffset(evidence, claim.support_quote) >= 0) };
  const literalClaims = new Map(current.claims.map((claim) => [claim.support_quote, claim]));
  let issues: string[] = [...initialIssues];
  for (let attempt = 0; attempt < 5; attempt += 1) {
    let response: SectionAuditResponse;
    try {
      response = normalizeSectionAuditResponse(await completeJson<SectionAuditResponse>({
        system: documentProfilePromptPack(options.language ?? getSettings().promptLanguage ?? 'es').sectionAudit,
        user: JSON.stringify({ fragment: evidence, analysis: current, prior_issues: issues }),
        temperature: 0, maxTokens: 5_000, signal: options.signal,
        requestClass: 'background', jobId: `${options.jobId}:section-audit`,
        perf: options.perf,
      }, isSectionAuditResponse, options.auditorModel), fallbackTitle);
    } catch (error) {
      if (recoverablePromptFailure(error) && current.claims.length > 1 && splitDepth < 4) {
        const middle = Math.ceil(current.claims.length / 2);
        const audited = await mapOrderedPool(
          [current.claims.slice(0, middle), current.claims.slice(middle)],
          2,
          (claims, _index, poolSignal) => auditSectionAnalysis(
            evidence,
            { ...current, claims },
            fallbackTitle,
            { ...options, signal: poolSignal },
            splitDepth + 1,
            initialIssues,
          ),
          options.signal,
        );
        return mergeSectionAnalyses(audited, fallbackTitle);
      }
      if (!recoverablePromptFailure(error)) throw error;
      break;
    }
    issues = response.issues;
    if (response.passed) return normalizeDirectSupportConfidence(current);
    if (!response.analysis) break;
    current = {
      ...response.analysis,
      claims: response.analysis.claims.filter((claim) => quoteOffset(evidence, claim.support_quote) >= 0),
    };
    for (const claim of current.claims) literalClaims.set(claim.support_quote, claim);
  }
  // A provider verdict must never force us to publish a disputed paraphrase, but
  // it also should not make a readable document permanently unindexable. Fall
  // back to an extractive representation whose prose is itself literal evidence;
  // the independent document-level audit still decides whether the resulting
  // macro profile is complete enough to publish.
  const sourceClaims = literalClaims.size
    ? [...literalClaims.values()]
    : literalSectionFallback(evidence, fallbackTitle).claims;
  const claims = sourceClaims.slice(0, 12).map((claim) => ({
    ...claim,
    text: claim.support_quote,
    confidence: DIRECT_SUPPORT_CONFIDENCE_FLOOR,
  }));
  return {
    title: fallbackTitle,
    summary: claims.slice(0, 3).map((claim) => claim.support_quote).join(' '),
    role: '',
    concepts: [],
    claims,
    degraded: true,
  };
}

async function analyzeSectionPart(
  evidence: string,
  key: string,
  title: string,
  pageStart: string | null,
  options: RunDocumentProfileOptions,
  depth = 0,
): Promise<SectionAnalysis> {
  const hash = profileCheckpointHash(key, evidence, options);
  const cached = readDocumentCheckpoint<SectionAnalysis>(options.jobId, key, hash);
  if (cached) return cached;
  let candidate: SectionAnalysis;
  try {
    candidate = normalizeSectionAnalysis(await completeJson<SectionAnalysis>({
      system: documentProfilePromptPack(options.language ?? getSettings().promptLanguage ?? 'es').section,
      user: JSON.stringify({ section_title: title, page_start: pageStart, fragment: evidence }),
      temperature: 0, maxTokens: 4_000, signal: options.signal,
      requestClass: 'background', jobId: `${options.jobId}:${key}`,
      perf: options.perf,
    }, isSectionAnalysis, options.generatorModel), title);
  } catch (error) {
    const words = evidence.split(/\s+/).filter(Boolean).length;
    if (recoverablePromptFailure(error) && words >= 400 && depth < 4) {
      const childWords = Math.max(120, Math.ceil(words / 2));
      const children = chunksWithOffsets(evidence, childWords).map((chunk) => chunk.body);
      if (children.length >= 2) {
        const values = await mapOrderedPool(
          children,
          Math.min(2, children.length),
          (child, index, poolSignal) => analyzeSectionPart(
            child,
            `${key}:split:${depth}:${index}`,
            title,
            pageStart,
            { ...options, signal: poolSignal },
            depth + 1,
          ),
          options.signal,
        );
        const merged = mergeSectionAnalyses(values, title);
        saveDocumentCheckpoint(options.jobId, key, hash, merged);
        return merged;
      }
    }
    if (!recoverablePromptFailure(error)) throw error;
    candidate = literalSectionFallback(evidence, title);
  }
  const literal = { ...candidate, claims: candidate.claims.filter((claim) => quoteOffset(evidence, claim.support_quote) >= 0) };
  const value = await auditSectionAnalysis(evidence, literal, title, options);
  saveDocumentCheckpoint(options.jobId, key, hash, value);
  return value;
}

async function analyzeSection(section: DerivedDocumentSection, options: RunDocumentProfileOptions): Promise<SectionAnalysis> {
  const sectionPack = documentProfilePromptPack(options.language ?? getSettings().promptLanguage ?? 'es').section;
  // The section audit sends the fragment AND the analysis of it, so the fragment may only
  // spend part of what is left after the pack; otherwise the audit prompt is the one that
  // no longer fits.
  const evidenceBudget = options.promptTokenBudget == null
    ? null
    : Math.max(120, Math.floor((options.promptTokenBudget - estimateLocalTokens(sectionPack)) * 0.8));
  const parts = splitAnalysisParts(section.body, evidenceBudget);
  const settings = getSettings();
  const poolSize = settings.aiConcurrencyMode === 'automatic' ? 8 : Math.max(1, Math.min(8, settings.concurrency));
  const analyses = await mapOrderedPool(parts, poolSize, async (part, index, poolSignal) => {
    const key = `section:${section.sectionId}:part:${index}`;
    return analyzeSectionPart(part, key, section.title, section.pageStart, { ...options, signal: poolSignal });
  }, options.signal);
  if (analyses.length === 1) return analyses[0];
  const reduceHash = profileCheckpointHash(`section:${section.sectionId}:reduced`, analyses, options);
  const cached = readDocumentCheckpoint<SectionAnalysis>(options.jobId, `section:${section.sectionId}:reduced`, reduceHash);
  if (cached) return cached;
  let candidate: SectionAnalysis;
  try {
    candidate = normalizeSectionAnalysis(await completeJson<SectionAnalysis>({
      system: documentProfilePromptPack(options.language ?? getSettings().promptLanguage ?? 'es').reduce, user: JSON.stringify({ title: section.title, analyses }),
      temperature: 0, maxTokens: 5_000, signal: options.signal,
      requestClass: 'background', jobId: `${options.jobId}:section:${section.sectionId}:reduce`,
      perf: options.perf,
    }, isSectionAnalysis, options.generatorModel), section.title);
  } catch (error) {
    // A merge the model cannot hold is not a reason to fail: the parts were analysed and
    // audited individually, so merging them here is the smaller, deterministic answer.
    if (!recoverablePromptFailure(error)) throw error;
    candidate = mergeSectionAnalyses(analyses, section.title);
  }
  const literal = { ...candidate, claims: candidate.claims.filter((claim) => quoteOffset(section.body, claim.support_quote) >= 0) };
  // Auditing the merged analysis against the whole section body is the largest prompt this
  // pipeline builds, and there is nothing to shrink: the evidence is the section, not a
  // fragment. When the model's window cannot hold it, the merged summary is left to the
  // document-level audit — which receives every section summary with its claims — instead of
  // failing the work or degrading a section whose parts each passed their own audit.
  if (!promptFits(sectionPack, { fragment: section.body, analysis: literal }, options.promptTokenBudget)) {
    saveDocumentCheckpoint(options.jobId, `section:${section.sectionId}:reduced`, reduceHash, literal);
    return literal;
  }
  const reduced = await auditSectionAnalysis(section.body, literal, section.title, options);
  saveDocumentCheckpoint(options.jobId, `section:${section.sectionId}:reduced`, reduceHash, reduced);
  return reduced;
}

function quoteOffset(text: string, quote: string): number {
  return findLiteralSourceSpans(text, quote)[0]?.charStart ?? -1;
}

function passageForQuote(
  nodusId: string,
  quote: string,
  span: SourceTextSpan,
  sourceRef: string | null,
  pageNumber: number | null,
  passages: RetrievalChunk[],
): string | null {
  const index = selectLiteralPassageIndex(passages, quote, span, sourceRef, pageNumber);
  return index == null ? null : `${nodusId}#${index}`;
}

function supportForQuote(input: {
  nodusId: string; text: string; quote: string; targetKind: 'field' | 'section'; targetId: string;
  sections: DerivedDocumentSection[]; confidence: number;
  passages: RetrievalChunk[];
  sourceMap: Record<string, string>;
}): DocumentProfileSupport | null {
  const targetSection = input.targetKind === 'section'
    ? input.sections.find((section) => section.sectionId === input.targetId) ?? null
    : null;
  const scope = targetSection?.charStart != null && targetSection.charEnd != null
    ? { charStart: targetSection.charStart, charEnd: targetSection.charEnd }
    : null;
  const matches = findLiteralSourceSpans(input.text, input.quote, scope);
  for (const match of matches) {
    const start = parseSourceLocationAt(input.text, match.charStart, input.sourceMap);
    const end = parseSourceLocationAt(input.text, Math.max(match.charStart, match.charEnd - 1), input.sourceMap);
    // One direct support cannot cross an attachment boundary. Null is a real
    // single-source identity here, not a wildcard for any marked source.
    if (start.sourceRef !== end.sourceRef) continue;
    const section = targetSection ?? input.sections.find((candidate) =>
      candidate.charStart != null && candidate.charEnd != null
      && match.charStart >= candidate.charStart && match.charEnd <= candidate.charEnd
    ) ?? null;
    return {
      supportId: randomUUID(), targetKind: input.targetKind, targetId: input.targetId,
      sectionId: section?.sectionId ?? null,
      passageId: passageForQuote(
        input.nodusId, input.quote, match, start.sourceRef, start.pageNumber, input.passages,
      ),
      // Provider-supplied page labels are never provenance. Both endpoints come
      // from the selected literal occurrence in the resolved source.
      pageStart: start.label, pageEnd: end.label,
      sourceRef: start.sourceRef, pageStartNumber: start.pageNumber, pageEndNumber: end.pageNumber,
      charStart: match.charStart, charEnd: match.charEnd,
      quote: input.quote, supportKind: 'direct', confidence: input.confidence, validationStatus: 'valid',
    };
  }
  return null;
}

async function preparePassages(
  work: Work,
  text: string,
  options: RunDocumentProfileOptions,
  sourceMap: Record<string, string> = {},
): Promise<PreparedPassages> {
  options.signal?.throwIfAborted();
  const contentHash = sha1(text);
  const chunks = planRetrievalChunks(text, { sourceMap });
  const current = getDb().prepare(
    'SELECT COUNT(*) count, MIN(content_hash) hash FROM passages WHERE nodus_id=?'
  ).get(work.nodus_id) as { count: number; hash: string | null };
  if (current.count > 0 && current.hash === contentHash) {
    const persisted = getDb().prepare(
      `SELECT chunk_index,text,page_label,source_ref,page_number
         FROM passages WHERE nodus_id=? ORDER BY chunk_index`
    ).all(work.nodus_id) as Array<{
      chunk_index: number; text: string; page_label: string | null;
      source_ref: string | null; page_number: number | null;
    }>;
    const compatible = persisted.length === chunks.length && persisted.every((row, index) => {
      const planned = chunks[index];
      return row.chunk_index === index
        && row.text === planned.text
        && row.page_label === planned.pageLabel
        && row.source_ref === planned.sourceRef
        && row.page_number === planned.pageNumber;
    });
    if (compatible) return { rows: chunks, publication: null };
  }
  const embeddingConfig = currentEmbeddingConfig();
  const embeddings = await embedMany(chunks.map((chunk) => chunk.text), options.signal, {
    perf: options.perf,
    jobId: `${options.jobId}:structure-embeddings`,
  });
  options.signal?.throwIfAborted();
  return {
    rows: chunks,
    publication: {
      contentHash,
      embeddingProvider: embeddingConfig.provider,
      embeddingModel: embeddingConfig.model,
      rows: chunks.map((chunk, index) => ({
        ...chunk, embedding: embeddings[index]?.length ? embeddings[index] : null,
      })),
    },
  };
}

function synthesisPayload(
  work: Work,
  sections: DerivedDocumentSection[],
  analyses: Map<string, SectionAnalysis>,
  abstract: string | null
): Record<string, unknown> {
  return {
    metadata: {
      title: work.title, authors: (() => { try { return JSON.parse(work.authors_json || '[]'); } catch { return []; } })(),
      year: work.year, item_type: work.item_type, original_abstract: abstract,
    },
    sections: sections.map((section) => ({
      id: section.sectionId, title: section.title, level: section.level, role: section.role,
      summary: section.summary, concepts: section.concepts,
      claims: analyses.get(section.sectionId)?.claims ?? section.claims.map((text) => ({ text })),
      page_start: section.pageStart, page_end: section.pageEnd,
    })),
  };
}

function mergeProfileSyntheses(profiles: ProfileSynthesis[]): ProfileSynthesis {
  const fields = new Map<string, RawProfileField>();
  for (const field of profiles.flatMap((profile) => profile.fields)) {
    const key = `${field.kind}\0${field.support_quote}`;
    const existing = fields.get(key);
    if (!existing || existing.confidence < field.confidence) fields.set(key, field);
  }
  return {
    source_language: profiles.find((profile) => profile.source_language !== 'und')?.source_language ?? 'und',
    overview: clean(profiles.map((profile) => profile.overview).filter(Boolean).join(' '), 5_000),
    fields: [...fields.values()].slice(0, 80),
  };
}

async function synthesizeProfileAdaptive(
  input: Record<string, unknown>,
  options: RunDocumentProfileOptions,
  splitPath = 'root',
  splitDepth = 0,
): Promise<ProfileSynthesis> {
  const inputHash = profileCheckpointHash(`profile:synthesis:${splitPath}`, input, options);
  const checkpointType = splitPath === 'root' ? 'profile:synthesis' : `profile:synthesis:${splitPath}`;
  const checkpoint = readDocumentCheckpoint<ProfileSynthesis>(options.jobId, checkpointType, inputHash);
  if (checkpoint) return checkpoint;
  const inputSections = Array.isArray(input.sections) ? input.sections : [];
  const synthesisPack = documentProfilePromptPack(options.language ?? getSettings().promptLanguage ?? 'es').profile;
  const splitInput = async (): Promise<ProfileSynthesis> => {
    const middle = Math.ceil(inputSections.length / 2);
    const halves = [inputSections.slice(0, middle), inputSections.slice(middle)];
    const profiles = await mapOrderedPool(
      halves,
      2,
      (sections, index, poolSignal) => synthesizeProfileAdaptive(
        { ...input, sections },
        { ...options, signal: poolSignal },
        `${splitPath}.${index}`,
        splitDepth + 1,
      ),
      options.signal,
    );
    const merged = mergeProfileSyntheses(profiles);
    saveDocumentCheckpoint(options.jobId, checkpointType, inputHash, merged);
    return merged;
  };
  // Proactive: a payload that cannot fit the loaded window is split before the request is
  // sent, rather than after the provider refuses it or truncates its answer.
  if (!promptFits(synthesisPack, input, options.promptTokenBudget) && inputSections.length >= 2 && splitDepth < 6) return splitInput();
  try {
    const profile = normalizeProfile(await completeJson<ProfileSynthesis>({
      system: documentProfilePromptPack(options.language ?? getSettings().promptLanguage ?? 'es').profile,
      user: JSON.stringify(input),
      temperature: 0,
      maxTokens: 8_000,
      signal: options.signal,
      requestClass: 'background',
      jobId: `${options.jobId}:profile:synthesis:${splitPath}`,
      perf: options.perf,
    }, isProfileSynthesis, options.generatorModel));
    saveDocumentCheckpoint(options.jobId, checkpointType, inputHash, profile);
    return profile;
  } catch (error) {
    if (!recoverablePromptFailure(error) || inputSections.length < 2 || splitDepth >= 6) throw error;
    return splitInput();
  }
}


function deterministicAudit(text: string, sections: DerivedDocumentSection[], profile: ProfileSynthesis): {
  supportCoverage: number; structureCoverage: number; supportedFields: RawProfileField[];
} {
  const supportedFields = profile.fields.filter((field) => quoteOffset(text, field.support_quote) >= 0);
  const covered = sections.reduce((total, section) => total + Math.max(0, (section.charEnd ?? 0) - (section.charStart ?? 0)), 0);
  return {
    supportCoverage: profile.fields.length ? supportedFields.length / profile.fields.length : 0,
    structureCoverage: text.length ? Math.min(1, covered / text.length) : 0,
    supportedFields,
  };
}

/** Keeps only fields whose support is literal in the source and records whether the
 *  published confidence was measured. Exported for unit testing. */
export function retainLiterallySupportedFields(text: string, profile: ProfileSynthesis): ProfileSynthesis {
  return {
    ...profile,
    fields: profile.fields
      .filter((field) => quoteOffset(text, field.support_quote) >= 0)
      .map((field) => ({
        ...field,
        // The floor is a publication contract (a literal support cannot be worth
        // nothing), not a measurement. When it replaces what the provider actually
        // reported, say so: the UI shows a bare percentage otherwise and every
        // field of a weak profile ends up reading "80 %" as if it had been scored.
        // This runs again after every audit and repair pass, so an existing floor
        // mark is sticky: the original reading is already gone from `confidence`.
        confidenceSource: field.confidenceSource === 'floor' || field.confidence < DIRECT_SUPPORT_CONFIDENCE_FLOOR ? 'floor' : 'model',
        confidence: Math.max(DIRECT_SUPPORT_CONFIDENCE_FLOOR, field.confidence),
      })),
  };
}

function evenlySample<T>(values: T[], limit: number): T[] {
  if (values.length <= limit) return values;
  return Array.from({ length: limit }, (_, index) => values[Math.floor((index * values.length) / limit)]);
}

/**
 * Last-resort profile for a readable work whose semantic synthesis remains
 * disputed after repair. It deliberately makes every field equal to literal
 * source text. This is less expressive than a thesis/method synthesis, but it
 * is safe for routing and prevents a large background campaign from acquiring
 * permanent holes because an auditor dislikes a paraphrase or a date boundary.
 */
function buildExtractiveProfileFallback(
  work: Work,
  sections: DerivedDocumentSection[],
  analyses: Map<string, SectionAnalysis>,
  sourceLanguage: string,
): ProfileSynthesis {
  const representatives: Array<{ quote: string; page: string | null }> = [];
  for (const section of sections) {
    const previous = analyses.get(section.sectionId);
    let claims = (previous?.claims ?? [])
      .filter((claim) => quoteOffset(section.body, claim.support_quote) >= 0)
      .map((claim) => ({
        ...claim,
        text: claim.support_quote,
        confidence: DIRECT_SUPPORT_CONFIDENCE_FLOOR,
      }));
    if (!claims.length) {
      const literal = clean(intactSourceExcerpt(section.body, 900), 900);
      if (literal) claims = [{
        text: literal,
        support_quote: literal,
        page: section.pageStart,
        confidence: DIRECT_SUPPORT_CONFIDENCE_FLOOR,
      }];
    }
    const safe = claims.slice(0, 3);
    const summary = safe.slice(0, 2).map((claim) => claim.support_quote).join(' ');
    analyses.set(section.sectionId, {
      title: section.title,
      summary,
      role: '',
      concepts: [],
      claims: safe,
      degraded: true,
    });
    section.role = null;
    section.summary = summary;
    section.concepts = [];
    section.claims = safe.map((claim) => claim.text);
    if (safe[0]) representatives.push({ quote: safe[0].support_quote, page: safe[0].page ?? section.pageStart });
  }
  const sampled = evenlySample(representatives, 12);
  if (!sampled.length) throw new Error('El documento no contiene ningún fragmento literal utilizable.');
  return {
    source_language: sourceLanguage || 'und',
    overview: clean(`${work.title}. ${sampled.slice(0, 4).map((item) => item.quote).join(' ')}`, 5_000),
    fields: sampled.map((item, index) => ({
      kind: 'argument',
      text: item.quote,
      // No provider measured this field: it *is* a quote, so the floor is the value.
      confidence: DIRECT_SUPPORT_CONFIDENCE_FLOOR,
      confidenceSource: 'floor' as const,
      centrality: index === 0 ? 0.7 : 0.6,
      support_quote: item.quote,
      page: item.page,
    })),
  };
}

function alignIdeas(nodusId: string, vectors: Array<{ sourceId: string; kind: string; embedding: number[] | null }>): DocumentIdeaLink[] {
  const ideas = getDb().prepare(
    `SELECT i.global_id,i.embedding FROM ideas i JOIN idea_occurrences io ON io.global_id=i.global_id
      WHERE io.nodus_id=? AND i.embedding IS NOT NULL`
  ).all(nodusId) as { global_id: string; embedding: Buffer }[];
  const links: DocumentIdeaLink[] = [];
  for (const idea of ideas) {
    const ideaVector = decodeEmbedding(idea.embedding);
    let best: { target: typeof vectors[number]; score: number } | null = null;
    for (const vector of vectors) {
      // DocumentIdeaLink can target only a materialized field or section. The
      // overview remains useful for document retrieval but has no persistent
      // target row, so aligning an idea to it would create a dangling link.
      if (vector.kind === 'overview' || !vector.embedding?.length) continue;
      const score = cosineSimilarity(ideaVector, vector.embedding);
      if (!best || score > best.score) best = { target: vector, score };
    }
    if (best && best.score >= 0.34) links.push({
      globalId: idea.global_id,
      targetKind: best.target.kind === 'section' ? 'section' : 'field',
      targetId: best.target.sourceId,
      role: best.score >= 0.62 ? 'principal' : best.score >= 0.48 ? 'supporting' : 'development',
      score: best.score,
    });
  }
  return links;
}

function emit(
  options: RunDocumentProfileOptions,
  phase: DocumentProfileScanProgress['phase'],
  progress: number,
  message: string,
  unit?: { current: number; total: number },
): void {
  options.onProgress?.({
    phase,
    progress,
    message,
    currentUnit: unit?.current,
    totalUnits: unit?.total,
  });
  const state = phase === 'analyzing_sections' ? 'analyzing'
    : phase === 'synthesizing' ? 'synthesizing'
    : phase === 'auditing' || phase === 'repairing' ? 'auditing'
    : phase === 'embedding' ? 'embedding'
    : phase === 'aligning' ? 'aligning'
    : phase === 'structuring' ? 'structuring'
    : phase === 'waiting_source' ? 'waiting_source' : null;
  if (!advanceRunningDocumentIndexJob(options.jobId, phase, progress, state, {
    message,
    currentUnit: unit?.current ?? null,
    totalUnits: unit?.total ?? null,
  })) {
    throw new Error('DOCUMENT_INDEX_CANCELLED');
  }
}

function materializeSectionAnalysis(
  section: DerivedDocumentSection,
  analysis: SectionAnalysis,
): DerivedDocumentSection {
  return {
    // A heading-less chunk intentionally keeps its generated title only after
    // analysis; the UI localizes a truly untitled section.
    ...section,
    title: section.title || analysis.title,
    role: analysis.role || null,
    summary: analysis.summary,
    concepts: analysis.concepts,
    claims: analysis.claims.map((claim) => claim.text),
  };
}

/** Full-text, hierarchical, audited document scan. */
export async function runDocumentProfileScan(work: Work, options: RunDocumentProfileOptions): Promise<string> {
  const scanStartedAt = Date.now();
  options = { ...options, perf: options.perf ?? { nodusId: work.nodus_id, title: work.title } };
  // The profile pipeline's prompts scale with the document, and a local server rejects a
  // prompt that does not fit the window it loaded. Asking each model up front lets every
  // request below be sized to it, instead of paying for a failed call first — or, for a
  // provider that truncates silently, for a plausible-looking answer built on half a prompt.
  options = {
    ...options,
    promptTokenBudget: options.promptTokenBudget ?? await localPromptTokenBudget(options.generatorModel),
    auditorTokenBudget: options.auditorTokenBudget ?? await localPromptTokenBudget(options.auditorModel),
  };
  options.signal?.throwIfAborted();
  if (!modelRefSupportsCapability(options.generatorModel, 'documentProfile')
    || !modelRefSupportsCapability(options.auditorModel, 'documentProfile')) {
    throw new AiError('El modelo local seleccionado no está certificado para perfiles documentales; no se inició la inferencia.', false, true);
  }
  const settings = getSettings();
  const userId = settings.zoteroUserId || LOCAL_USER_ID;
  emit(options, 'waiting_source', 0.01, 'Resolviendo el texto completo…');
  // Nodus-owned Library references already point at a materialized, clean local
  // document. Asking Zotero about that synthetic key first is both unnecessary
  // and harmful when Zotero's local server accepts the connection but never
  // answers: a manual Documentary Index then appears frozen at waiting_source.
  const item = work.zotero_key.startsWith('nodus-library:')
    ? null
    : await getItem(userId, work.zotero_key).catch(() => null);
  const document = await resolveWorkText(
    userId, work.zotero_key, settings.zoteroStoragePath, item?.abstract ?? null, work.doi,
    {
      unpaywallEmail: settings.unpaywallEmail,
      preferZoteroFulltext: settings.preferZoteroFulltext,
      ocr: { enabled: settings.ocrEnabled, languages: settings.ocrLanguages, maxPages: settings.ocrMaxPages },
      signal: options.signal,
    },
    work.item_type
  );
  setResolvedTextState(work.nodus_id, resolvedTextStateFromDoc(document));
  options.signal?.throwIfAborted();
  if (!document.text.trim() || document.sourceType === 'none' || document.sourceType === 'abstract_only') {
    throw new DocumentSourceUnavailableError(
      document.notes ?? 'No hay texto completo legible',
      document.sourceType,
      document.blockReason ?? null,
    );
  }
  const sourceFingerprint = sha256(document.text);
  const sourceContentHash = sha1(document.text);
  const sourceMap = Object.fromEntries((document.segments ?? []).map((segment) => [segment.marker, segment.sourceRef]));
  // deep_hash may come from a pre-inventory deep scan whose corpus did not carry
  // durable source markers. It is therefore not comparable with this resolved-text
  // hash. Real races are guarded below by a second read plus the atomic revision check.
  updateDocumentIndexJob(options.jobId, { sourceFingerprint });
  emit(options, 'structuring', 0.04, 'Reconstruyendo la estructura…');
  const sections = deriveDocumentStructure(document.text, work.title, sourceMap);
  if (!sections.length) throw new Error('El documento no contiene texto estructurable.');

  emit(options, 'embedding', 0.05, 'Indexando los pasajes del texto completo…');
  const preparedPassages = await preparePassages(
    work,
    document.text,
    options,
    sourceMap,
  );
  const sectionPool = settings.aiConcurrencyMode === 'automatic' ? 8 : Math.max(1, Math.min(8, settings.concurrency));
  const orderedAnalyses = await mapOrderedPool(sections, sectionPool, async (section, index, poolSignal) => {
    emit(
      options,
      'analyzing_sections',
      0.08 + (index / sections.length) * 0.54,
      `Analizando sección ${index + 1} de ${sections.length}…`,
      { current: index + 1, total: sections.length },
    );
    return analyzeSection(section, { ...options, signal: poolSignal });
  }, options.signal);
  const sectionAnalyses = new Map<string, SectionAnalysis>();
  for (let index = 0; index < sections.length; index += 1) {
    const analysis = orderedAnalyses[index];
    sectionAnalyses.set(sections[index].sectionId, analysis);
    sections[index] = materializeSectionAnalysis(sections[index], analysis);
  }

  let synthesisInput = synthesisPayload(work, sections, sectionAnalyses, item?.abstract ?? null);
  emit(options, 'synthesizing', 0.64, 'Sintetizando la obra completa…');
  let profile: ProfileSynthesis;
  let extractiveFallback = false;
  let repaired = false;
  try {
    profile = await synthesizeProfileAdaptive(synthesisInput, options);
  } catch (error) {
    if (!recoverablePromptFailure(error)) throw error;
    profile = buildExtractiveProfileFallback(work, sections, sectionAnalyses, 'und');
    extractiveFallback = true;
    repaired = true;
  }
  if (!profile.overview || !profile.fields.length) {
    profile = buildExtractiveProfileFallback(work, sections, sectionAnalyses, profile.source_language);
    extractiveFallback = true;
    repaired = true;
  }
  const initialFieldCount = profile.fields.length;
  profile = retainLiterallySupportedFields(document.text, profile);
  if (!profile.fields.length) {
    profile = buildExtractiveProfileFallback(work, sections, sectionAnalyses, profile.source_language);
    extractiveFallback = true;
    repaired = true;
  }
  repaired = repaired || profile.fields.length !== initialFieldCount;

  profile = withFieldHandles(profile, sourceFingerprint);
  let auditor: AuditResponse | null = null;
  let auditFailureNote: string | null = null;
  let deterministic = deterministicAudit(document.text, sections, profile);
  let auditedRevision: string | null = null;
  let repairDispatches = 0;
  const seenCandidateRevisions = new Set<string>();
  const promptPack = documentProfilePromptPack(options.language ?? getSettings().promptLanguage ?? 'es');

  const mergeAudits = (revision: string, responses: AuditResponse[]): AuditResponse => {
    const scores = responses.map((response) => response.score).filter((score): score is number => score != null);
    const overviews = [...new Set(responses.map((response) => response.overview).filter((value): value is string => !!value))];
    const issues = responses.flatMap((response) => response.issues);
    if (overviews.length > 1) issues.push({
      code: 'conflicting_overview_fixes',
      blocking: true,
      target: { kind: 'overview', handle: null },
      explanation: 'Audit batches proposed conflicting overview replacements.',
    });
    return {
      candidateRevision: revision,
      passed: responses.length > 0 && responses.every((response) => response.passed),
      score: scores.length ? Math.min(...scores) : null,
      issues,
      operations: responses.flatMap((response) => response.operations),
      overview: overviews[0] ?? null,
    };
  };

  const requestAuditUnit = async (
    body: Record<string, unknown>,
    revision: string,
    attempt: number,
    label: string,
  ): Promise<AuditResponse> => normalizeDocumentProfileAuditResponse(await completeJson<AuditResponse>({
    system: promptPack.audit,
    user: JSON.stringify(body),
    temperature: 0,
    maxTokens: 3_000,
    signal: options.signal,
    requestClass: 'background',
    jobId: `${options.jobId}:profile:audit:${attempt}:${label}`,
    perf: options.perf,
  }, isAuditResponse, options.auditorModel), revision, profile.fields.map((field) => field.handle ?? ''));

  const requestAudit = async (attempt: number, forceBatches = false): Promise<AuditResponse> => {
    const revision = profileCandidateRevision(profile, sections);
    const sectionPayloads = Array.isArray(synthesisInput.sections) ? synthesisInput.sections : [];
    const evidence = sectionEvidenceInventory(sections, sectionAnalyses);
    const deterministicPayload = {
      support_coverage: deterministic.supportCoverage,
      structure_coverage: deterministic.structureCoverage,
    };
    const full = {
      candidate_revision: revision,
      profile,
      sections: sectionPayloads,
      evidence,
      deterministic: deterministicPayload,
    };
    if (!forceBatches && promptFits(promptPack.audit, full, options.auditorTokenBudget)) {
      return requestAuditUnit(full, revision, attempt, 'full');
    }

    const responses: AuditResponse[] = [];
    const includedFields = new Set<string>();
    for (let index = 0; index < sections.length; index += 1) {
      const section = sections[index];
      const claims = sectionAnalyses.get(section.sectionId)?.claims ?? [];
      const fields = profile.fields.filter((field) => claims.some((claim) => claim.support_quote.includes(field.support_quote)));
      fields.forEach((field) => includedFields.add(field.handle ?? ''));
      const body = {
        candidate_revision: revision,
        mode: 'complete_section_unit',
        profile: { ...profile, fields },
        sections: [sectionPayloads[index]],
        evidence: evidence.filter((entry) => entry.sectionId === section.sectionId),
        deterministic: deterministicPayload,
      };
      if (!promptFits(promptPack.audit, body, options.auditorTokenBudget)) {
        throw new AiError('La unidad completa de auditoría no cabe en el contexto del modelo.', false, true, 'context_overflow');
      }
      responses.push(await requestAuditUnit(body, revision, attempt, `section-${index}`));
    }
    const remaining = profile.fields.filter((field) => !includedFields.has(field.handle ?? ''));
    if (remaining.length) {
      const body = {
        candidate_revision: revision,
        mode: 'complete_unassigned_fields',
        profile: { ...profile, fields: remaining },
        sections: [],
        evidence,
        deterministic: deterministicPayload,
      };
      if (!promptFits(promptPack.audit, body, options.auditorTokenBudget)) {
        throw new AiError('Los campos completos sin sección no caben en el contexto del modelo.', false, true, 'context_overflow');
      }
      responses.push(await requestAuditUnit(body, revision, attempt, 'unassigned'));
    }
    const global = {
      candidate_revision: revision,
      mode: 'global_consistency_and_omission',
      profile,
      sections: sectionPayloads,
      deterministic: deterministicPayload,
    };
    if (!promptFits(promptPack.audit, global, options.auditorTokenBudget)) {
      throw new AiError('La comprobación global completa no cabe en el contexto del modelo.', false, true, 'context_overflow');
    }
    responses.push(await requestAuditUnit(global, revision, attempt, 'global'));
    return mergeAudits(revision, responses);
  };

  const requestRepairDelta = async (
    issues: AuditIssue[],
    revision: string,
    depth = 0,
  ): Promise<RepairDelta | null> => {
    if (!issues.length || repairDispatches >= 8) return null;
    const fieldHandles = new Set(issues
      .filter((issue) => issue.target.kind === 'field' && issue.target.handle)
      .map((issue) => issue.target.handle as string));
    const sectionHandles = new Set(issues
      .filter((issue) => ['section', 'claim'].includes(issue.target.kind) && issue.target.handle)
      .map((issue) => String(issue.target.handle).split(':claim:')[0]));
    const evidence = sectionEvidenceInventory(sections, sectionAnalyses)
      .filter((entry) => !sectionHandles.size || sectionHandles.has(entry.sectionId));
    const body = {
      base_revision: revision,
      issues,
      overview: issues.some((issue) => issue.target.kind === 'overview') ? profile.overview : undefined,
      fields: profile.fields.filter((field) => fieldHandles.has(field.handle ?? '')),
      evidence,
    };
    repairDispatches += 1;
    try {
      const raw = await completeJson<RepairDelta>({
        system: promptPack.deltaRepair,
        user: JSON.stringify(body),
        temperature: 0,
        maxTokens: issues.length === 1 ? 2_000 : 3_000,
        signal: options.signal,
        requestClass: 'background',
        jobId: `${options.jobId}:profile:delta:${repairDispatches}`,
        perf: options.perf,
      }, isRepairDeltaResponse, options.generatorModel);
      const delta = normalizeRepairDelta(raw);
      return delta?.baseRevision === revision ? delta : null;
    } catch (error) {
      if (!recoverablePromptFailure(error)) throw error;
      if (issues.length > 1 && depth < 4) {
        const middle = Math.ceil(issues.length / 2);
        const left = await requestRepairDelta(issues.slice(0, middle), revision, depth + 1);
        const right = await requestRepairDelta(issues.slice(middle), revision, depth + 1);
        if (!left && !right) return null;
        return {
          baseRevision: revision,
          operations: [...(left?.operations ?? []), ...(right?.operations ?? [])],
          overview: left?.overview ?? right?.overview ?? null,
        };
      }
      if (issues.length === 1 && depth === 0) return requestRepairDelta(issues, revision, 1);
      return null;
    }
  };

  for (let attempt = 0; !extractiveFallback && attempt < 4; attempt += 1) {
    const revision = profileCandidateRevision(profile, sections);
    if (seenCandidateRevisions.has(revision)) {
      auditFailureNote = 'La reparación no cambió la revisión candidata.';
      break;
    }
    seenCandidateRevisions.add(revision);
    emit(options, attempt === 0 ? 'auditing' : 'repairing', 0.7 + Math.min(attempt, 2) * 0.05,
      attempt === 0 ? 'Auditando la ficha contra el texto…' : `Reparando la ficha (${attempt}/3)…`);
    try {
      auditor = await requestAudit(attempt);
      auditedRevision = revision;
    } catch (error) {
      if (!recoverablePromptFailure(error)) throw error;
      try {
        auditor = await requestAudit(attempt, true);
        auditedRevision = revision;
      } catch (batchedError) {
        if (!recoverablePromptFailure(batchedError)) throw batchedError;
        auditFailureNote = describeFailure(batchedError);
        auditor = null;
        break;
      }
    }

    let changed = false;
    const directDelta: RepairDelta = {
      baseRevision: revision,
      operations: auditor.operations,
      overview: auditor.overview,
    };
    if (directDelta.operations.length || directDelta.overview) {
      const applied = applyRepairDelta({
        profile, delta: directDelta, expectedRevision: revision,
        evidence: sectionEvidenceInventory(sections, sectionAnalyses), sections,
      });
      if (applied.errors.length) {
        auditFailureNote = `La reparación propuesta no superó la validación: ${applied.errors.join(', ')}.`;
        break;
      }
      profile = applied.profile;
      changed = applied.changed;
    }

    const sectionIssues = auditor.issues.filter((issue) =>
      issue.blocking && ['section', 'claim'].includes(issue.target.kind) && issue.target.handle
    );
    const issuesBySection = new Map<string, string[]>();
    for (const issue of sectionIssues) {
      const sectionId = String(issue.target.handle).split(':claim:')[0];
      issuesBySection.set(sectionId, [...(issuesBySection.get(sectionId) ?? []), issue.explanation]);
    }
    for (const [sectionId, issues] of [...issuesBySection].slice(0, 4)) {
      const index = sections.findIndex((section) => section.sectionId === sectionId);
      const current = sectionAnalyses.get(sectionId);
      if (index < 0 || !current) continue;
      const next = await auditSectionAnalysis(sections[index].body, current, sections[index].title, options, 0, issues);
      if (JSON.stringify(next) !== JSON.stringify(current)) {
        sectionAnalyses.set(sectionId, next);
        sections[index] = materializeSectionAnalysis(sections[index], next);
        changed = true;
      }
    }

    if (!changed && !auditor.passed) {
      const repairable = auditor.issues.filter((issue) =>
        issue.blocking && ['field', 'overview'].includes(issue.target.kind)
      );
      const delta = await requestRepairDelta(repairable, revision);
      if (delta) {
        const applied = applyRepairDelta({
          profile, delta, expectedRevision: revision,
          evidence: sectionEvidenceInventory(sections, sectionAnalyses), sections,
        });
        if (!applied.errors.length) {
          profile = applied.profile;
          changed = applied.changed;
        } else {
          auditFailureNote = `La reparación dirigida no superó la validación: ${applied.errors.join(', ')}.`;
        }
      }
    }

    if (changed) {
      repaired = true;
      synthesisInput = synthesisPayload(work, sections, sectionAnalyses, item?.abstract ?? null);
      const retainedCount = profile.fields.length;
      profile = retainLiterallySupportedFields(document.text, profile);
      if (profile.fields.length !== retainedCount) repaired = true;
      if (!profile.fields.length) break;
      deterministic = deterministicAudit(document.text, sections, profile);
      auditor = null;
      auditedRevision = null;
      continue;
    }
    deterministic = deterministicAudit(document.text, sections, profile);
    break;
  }
  const finalRevision = profileCandidateRevision(profile, sections);
  if (auditor && auditedRevision !== finalRevision) {
    auditFailureNote = 'La revisión final difiere de la revisión auditada.';
    auditor = null;
  }
  const deterministicComplete = deterministic.supportCoverage === 1 && deterministic.structureCoverage >= 0.95;
  const approved = semanticApproved(auditor);
  // Keep a synthesis as `partial` only when the auditor explicitly approved its
  // semantics but the numeric score missed the preferred acceptance bar. A rejected
  // claim is not made safe merely because its quoted words occur somewhere in the
  // source; explicit rejection therefore degrades to literal extraction below.
  let mode: DocumentProfileFallbackMode | null = null;
  let publishable = false;
  let auditPassed = false;
  if (approved && deterministicComplete) {
    publishable = true;
    auditPassed = true;
  } else if (
    !extractiveFallback
    && auditor?.passed === true
    && deterministicComplete
    && profile.fields.length > 0
  ) {
    mode = 'partial';
    publishable = true;
    repaired = true;
  } else {
    // Last resort for a synthesis that produced nothing usable: rebuild the profile from
    // literal quotes. This is deliberately conservative content, not a failed profile.
    // A profile that already IS that fallback (the synthesis never produced fields) is
    // left as it is: rebuilding it would produce exactly the same quotes again.
    if (!extractiveFallback) {
      profile = buildExtractiveProfileFallback(work, sections, sectionAnalyses, profile.source_language);
      deterministic = deterministicAudit(document.text, sections, profile);
      extractiveFallback = true;
      repaired = true;
    }
    mode = 'extractive';
    // A literal profile is fully supported by construction, so it satisfies the
    // deterministic contract; `fallback` is what tells consumers it is publishable
    // without pretending the semantic auditor approved a synthesis.
    publishable = deterministic.supportCoverage === 1 && deterministic.structureCoverage >= 0.95;
  }
  // How much of the published profile is quotation rather than synthesis. A profile can
  // be approved as a whole while individual sections were degraded, and nothing else in
  // the record would say so.
  const sectionsDegraded = [...sectionAnalyses.values()].filter((analysis) => analysis.degraded).length;
  const audit: DocumentProfileAudit = {
    passed: auditPassed,
    score: auditor?.score ?? null,
    supportCoverage: deterministic.supportCoverage,
    structureCoverage: deterministic.structureCoverage,
    issues: [
      ...(mode === 'extractive' ? ['fallback_extractivo_determinista'] : []),
      ...(auditor?.issues.map((issue) => issue.explanation) ?? []),
      ...(auditFailureNote ? [auditFailureNote] : []),
    ].slice(0, 50),
    repaired: repaired || Boolean(auditor && (auditor.operations.length || auditor.overview)),
    fallback: mode,
    sectionsDegraded,
  };
  if (!publishable) {
    throw new DocumentProfilePublicationError(auditFailureMessage(audit));
  }

  const fields = deterministic.supportedFields.map((field, index) => ({
    fieldId: `field-${sha256(`${sourceFingerprint}|${field.kind}|${index}|${field.text}`).slice(0, 24)}`,
    kind: field.kind, ordinal: deterministic.supportedFields.slice(0, index).filter((prior) => prior.kind === field.kind).length,
    text: field.text, confidence: field.confidence, centrality: CENTRAL_FIELD_KINDS.has(field.kind) ? Math.max(0.75, field.centrality) : field.centrality,
    confidenceSource: field.confidenceSource ?? ('model' as DocumentProfileConfidenceSource),
  }));
  const supports: DocumentProfileSupport[] = [];
  deterministic.supportedFields.forEach((field, index) => {
    const support = supportForQuote({
      nodusId: work.nodus_id, text: document.text, quote: field.support_quote, targetKind: 'field',
      targetId: fields[index].fieldId, sections, confidence: field.confidence,
      passages: preparedPassages.rows, sourceMap,
    });
    if (support) supports.push(support);
  });
  for (const section of sections) {
    const analysis = sectionAnalyses.get(section.sectionId);
    const quote = analysis?.claims?.[0]?.support_quote;
    if (!quote) continue;
    const support = supportForQuote({
      nodusId: work.nodus_id, text: document.text, quote, targetKind: 'section', targetId: section.sectionId,
      sections, confidence: analysis.claims[0].confidence,
      passages: preparedPassages.rows, sourceMap,
    });
    if (support) supports.push(support);
  }

  emit(options, 'embedding', 0.87, 'Creando los vectores documentales…');
  const vectorSources = [
    { kind: 'overview', sourceId: 'overview', text: profile.overview, weight: 1 },
    ...fields.map((field) => ({ kind: field.kind, sourceId: field.fieldId, text: field.text, weight: field.centrality || 0.5 })),
    ...sections.map((section) => ({ kind: 'section', sourceId: section.sectionId, text: `${section.title}\n${section.summary}`, weight: 0.75 })),
  ].filter((source) => source.text.trim());
  const vectorEmbeddingConfig = currentEmbeddingConfig();
  const embeddings = await embedMany(vectorSources.map((source) => source.text), options.signal, {
    perf: options.perf,
    jobId: `${options.jobId}:profile-embeddings`,
  });
  options.signal?.throwIfAborted();
  const vectors = vectorSources.map((source, index) => ({
    ...source,
    embedding: embeddings[index]?.length ? embeddings[index] : null,
    embeddingProvider: vectorEmbeddingConfig.provider,
    embeddingModel: vectorEmbeddingConfig.model,
  }));

  emit(options, 'aligning', 0.94, 'Alineando la estructura con las ideas…');
  const ideaLinks: DocumentIdeaLink[] = alignIdeas(work.nodus_id, vectors);
  emit(options, 'publishing', 0.98, 'Publicando la versión auditada…');
  options.signal?.throwIfAborted();
  // Re-resolve the source at the publication boundary. Database revision guards
  // catch normal Zotero/sync changes; this additionally catches a file replaced
  // externally while a long analysis is running, even before deep_hash changes.
  const latestDocument = await resolveWorkText(
    userId, work.zotero_key, settings.zoteroStoragePath, item?.abstract ?? null, work.doi,
    {
      unpaywallEmail: settings.unpaywallEmail,
      preferZoteroFulltext: settings.preferZoteroFulltext,
      ocr: { enabled: settings.ocrEnabled, languages: settings.ocrLanguages, maxPages: settings.ocrMaxPages },
      signal: options.signal,
    },
    work.item_type,
  );
  options.signal?.throwIfAborted();
  if (latestDocument.sourceType === 'none'
    || latestDocument.sourceType === 'abstract_only'
    || sha256(latestDocument.text) !== sourceFingerprint) {
    throw new Error('DOCUMENT_SOURCE_CHANGED');
  }
  const profileForPublication = {
    ...profile,
    fields: profile.fields.map(({ handle: _handle, ...field }) => field),
    metadata: synthesisInput.metadata,
    fallbackMode: mode,
  };
  const versionId = publishDocumentProfile({
    nodusId: work.nodus_id, sourceFingerprint, pipelineVersion: DOCUMENT_PROFILE_PIPELINE_VERSION,
    schemaVersion: DOCUMENT_PROFILE_SCHEMA_VERSION, sourceLanguage: profile.source_language,
    presentationLanguage: settings.promptLanguage, overview: profile.overview,
    profile: profileForPublication, fields,
    sections: sections.map(({ body: _body, ...section }) => section), supports, ideaLinks,
    resolvedText: latestDocument.text,
    vectors, generatorModel: options.generatorModel, auditorModel: options.auditorModel,
    promptHash: sha256(JSON.stringify(documentProfilePromptPack(options.language ?? settings.promptLanguage ?? 'es'))), audit,
    // Quality is the lowest of the readings, and only exists when the auditor actually
    // produced one. Without a semantic reading the deterministic coverages are all the
    // gate requires (so they would read as a perfect score) and the profile already
    // says it was not approved; reporting "100 %" beside that caveat would be worse
    // than reporting nothing.
    qualityScore: mode === 'extractive' || audit.score == null
      ? null
      : Math.min(audit.score, audit.supportCoverage, audit.structureCoverage),
    expectedWorkRevision: {
      zoteroKey: work.zotero_key,
      zoteroVersion: work.zotero_version,
      title: work.title,
      authorsJson: work.authors_json,
      year: work.year,
      itemType: work.item_type,
      doi: work.doi,
      deepHash: work.deep_hash,
      resolvedTextHash: sourceContentHash,
    },
    passages: preparedPassages.publication,
  });
  // The one green line per indexed document. It carries the numbers a reader cannot get
  // back afterwards — how many sections were extracted and how many vectors were published
  // — and inherits vault/document/job from the queue's log scope.
  logPipelineSuccess({
    subject: 'subjectIndexing',
    message: {
      id: 'documentIndexed',
      params: { title: work.title, sections: sections.length, vectors: vectors.length },
    },
    durationMs: Date.now() - scanStartedAt,
  });
  upsertLibraryAnalysisProvenance({
    workId: work.nodus_id,
    component: 'documentProfile',
    documentFingerprint: sourceContentHash,
    libraryItemId: null,
    libraryRevisionFingerprint: null,
    pipelineVersion: DOCUMENT_PROFILE_PIPELINE_VERSION,
    modelFingerprint: analysisModelFingerprint('documentProfile', settings),
    outputFingerprint: analysisFingerprint({ versionId, sourceFingerprint, audit, overview: profile.overview }),
    sourceVaultId: null,
    sourceWorkId: null,
    updatedAt: new Date().toISOString(),
  });
  clearDocumentCheckpoints(options.jobId);
  return versionId;
}

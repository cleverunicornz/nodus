import { getDb } from './database';
import { planExtractionCacheEviction } from '@shared/extractionCachePrune';
import type { ExtractionControlDiagnostics, PdfAnalysis, SourceType } from '@shared/types';

export const EXTRACTION_CACHE_VERSION = 4;

/** How much extracted text the cache may hold before the oldest entries go. */
const MAX_CACHE_BYTES = 64 * 1024 * 1024;

/**
 * Reading every row's `length(text)` means touching all the cached text, so the
 * prune is throttled rather than run on every extraction. It stays off the app's
 * critical paths entirely: it only ever runs right after a document was extracted,
 * which already cost seconds of PDF parsing.
 */
const PRUNE_INTERVAL_MS = 5 * 60_000;
let lastPruneAt = 0;

interface OcrCacheOptions {
  enabled: boolean;
  languages: string;
  maxPages: number;
}

export interface ExtractionCacheDoc {
  text: string;
  sourceType: SourceType;
  notes: string | null;
  analysis?: PdfAnalysis;
  controlDiagnostics?: ExtractionControlDiagnostics;
}

interface CacheKey {
  filePath: string;
  fileSize: number;
  fileMtimeMs: number;
  ocr: OcrCacheOptions;
}

interface ExtractionCacheRow {
  file_path: string;
  file_size: number;
  file_mtime_ms: number;
  ocr_enabled: number;
  ocr_languages: string;
  ocr_max_pages: number;
  cache_version: number;
  source_type: SourceType;
  text: string;
  notes: string | null;
  analysis_json: string | null;
}

export function getExtractionCache(key: CacheKey): ExtractionCacheDoc | null {
  const row = getDb()
    .prepare(
      `SELECT file_path, file_size, file_mtime_ms, ocr_enabled, ocr_languages, ocr_max_pages,
              cache_version, source_type, text, notes, analysis_json
       FROM extraction_cache
       WHERE file_path = ?
         AND file_size = ?
         AND file_mtime_ms = ?
         AND ocr_enabled = ?
         AND ocr_languages = ?
         AND ocr_max_pages = ?
         AND cache_version = ?`
    )
    .get(
      key.filePath,
      key.fileSize,
      key.fileMtimeMs,
      key.ocr.enabled ? 1 : 0,
      key.ocr.languages,
      key.ocr.maxPages,
      EXTRACTION_CACHE_VERSION
    ) as ExtractionCacheRow | undefined;

  if (!row) return null;
  const cached = parseCachedAnalysis(row.analysis_json);
  return {
    text: row.text,
    sourceType: row.source_type,
    notes: row.notes,
    analysis: cached.analysis,
    controlDiagnostics: cached.controlDiagnostics,
  };
}

export function upsertExtractionCache(key: CacheKey, doc: ExtractionCacheDoc): void {
  const now = new Date().toISOString();
  getDb()
    .prepare(
      `INSERT INTO extraction_cache (
         file_path, file_size, file_mtime_ms, ocr_enabled, ocr_languages, ocr_max_pages,
         cache_version, source_type, text, notes, analysis_json, created_at, updated_at
       )
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(file_path) DO UPDATE SET
         file_size = excluded.file_size,
         file_mtime_ms = excluded.file_mtime_ms,
         ocr_enabled = excluded.ocr_enabled,
         ocr_languages = excluded.ocr_languages,
         ocr_max_pages = excluded.ocr_max_pages,
         cache_version = excluded.cache_version,
         source_type = excluded.source_type,
         text = excluded.text,
         notes = excluded.notes,
         analysis_json = excluded.analysis_json,
         updated_at = excluded.updated_at`
    )
    .run(
      key.filePath,
      key.fileSize,
      key.fileMtimeMs,
      key.ocr.enabled ? 1 : 0,
      key.ocr.languages,
      key.ocr.maxPages,
      EXTRACTION_CACHE_VERSION,
      doc.sourceType,
      doc.text,
      doc.notes,
      doc.analysis || doc.controlDiagnostics
        ? JSON.stringify({ pdf: doc.analysis ?? null, controls: doc.controlDiagnostics ?? null })
        : null,
      now,
      now
    );
  maybePruneExtractionCache();
}

/**
 * Trim the cache to `maxBytes`, keeping the most recently extracted text. Returns
 * what it removed so callers and tests can assert on it.
 */
export function pruneExtractionCache(maxBytes = MAX_CACHE_BYTES): { removed: number; freedBytes: number } {
  const db = getDb();
  const rows = db
    .prepare(`SELECT file_path, length(CAST(text AS BLOB)) AS bytes, updated_at FROM extraction_cache`)
    .all() as { file_path: string; bytes: number | null; updated_at: string }[];
  const plan = planExtractionCacheEviction(
    rows.map((row) => ({ filePath: row.file_path, bytes: row.bytes ?? 0, updatedAt: row.updated_at })),
    { maxBytes }
  );
  if (plan.remove.length === 0) return { removed: 0, freedBytes: 0 };
  const remove = db.prepare(`DELETE FROM extraction_cache WHERE file_path = ?`);
  db.transaction((paths: string[]) => {
    for (const filePath of paths) remove.run(filePath);
  })(plan.remove);
  return { removed: plan.remove.length, freedBytes: plan.freedBytes };
}

function maybePruneExtractionCache(): void {
  const now = Date.now();
  if (now - lastPruneAt < PRUNE_INTERVAL_MS) return;
  lastPruneAt = now;
  try {
    pruneExtractionCache();
  } catch {
    // A cache that failed to shrink must never fail the extraction that filled it.
  }
}

function parseCachedAnalysis(value: string | null): {
  analysis?: PdfAnalysis;
  controlDiagnostics?: ExtractionControlDiagnostics;
} {
  if (!value) return {};
  try {
    const parsed = JSON.parse(value) as Record<string, unknown>;
    // v3 stored PdfAnalysis directly. Keep this reader tolerant for recovery
    // archives even though v4 cache lookup will not select those rows.
    if (typeof parsed.pageCount === 'number') {
      const analysis = parsed as unknown as PdfAnalysis;
      return { analysis, controlDiagnostics: analysis.controlDiagnostics };
    }
    return {
      analysis: parsed.pdf && typeof parsed.pdf === 'object' ? parsed.pdf as PdfAnalysis : undefined,
      controlDiagnostics: parsed.controls && typeof parsed.controls === 'object'
        ? parsed.controls as ExtractionControlDiagnostics
        : undefined,
    };
  } catch {
    return {};
  }
}

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { installRuntimeHooks, requireElectronRuntime } from './lib/tsRuntimeHooks.mjs';

const scriptPath = fileURLToPath(import.meta.url);
if (!requireElectronRuntime(scriptPath, '--electron-canary-corpus-verifier')) process.exit(0);
const repoRoot = path.resolve(path.dirname(scriptPath), '..');
const require = createRequire(import.meta.url);
const profileRoot = path.resolve(arg('profile'));
const expectedDatabase = path.resolve(arg('database'));
const reportPath = arg('report') ? path.resolve(arg('report')) : null;
installRuntimeHooks(profileRoot);
const Database = require('better-sqlite3');
const { activeVault, assertOpenedCanaryDatabase } = await import(`./canary-profile.mjs?v=${Date.now()}`);
const { extractPdfStreaming, planRetrievalChunks } = require(path.join(repoRoot, 'electron/extraction/textExtractor.ts'));
const {
  literalSourceSpanMatches,
  selectLiteralPassageIndex,
  textCanonicallyContainsLiteral,
} = require(path.join(repoRoot, 'electron/extraction/sourceTextRanges.ts'));

const registryPath = path.join(profileRoot, 'vaults.json');
const { vault } = activeVault(registryPath);
assert.equal(fs.realpathSync.native(vault.path), fs.realpathSync.native(expectedDatabase),
  'Canary registry does not resolve to the explicitly expected database');
const db = new Database(expectedDatabase, { readonly: true, fileMustExist: true });
  const preflightPath = path.join(profileRoot, 'preflight-report.json');
  const preflight = fs.existsSync(preflightPath) ? JSON.parse(fs.readFileSync(preflightPath, 'utf8')) : null;
  const preflightAttachments = new Map((preflight?.documents ?? []).flatMap((document) =>
    (document.attachments ?? []).filter((attachment) => attachment.contentType === 'application/pdf').map((attachment) => [
      attachment.itemKey ?? attachment.key,
      {
        nodus_id: document.nodusId,
        source_ref: `zotero:user:0:${attachment.itemKey ?? attachment.key}`,
        attachment_key: attachment.itemKey ?? attachment.key,
      },
    ])
  ));
const report = {
  format: 'nodus.canary-corpus-verification', formatVersion: 1, createdAt: new Date().toISOString(),

  profileRoot: fs.realpathSync.native(profileRoot), registryPath: fs.realpathSync.native(registryPath),
  databasePath: null, integrity: null, foreignKeyViolations: null, profiles: [], documents: [], ok: false,
};
try {
  report.databasePath = assertOpenedCanaryDatabase(db, expectedDatabase);
  report.integrity = db.pragma('integrity_check', { simple: true });
  report.foreignKeyViolations = db.pragma('foreign_key_check').length;
  assert.equal(report.integrity, 'ok');
  const cacheRows = db.prepare(`SELECT file_path,text,analysis_json,cache_version FROM extraction_cache
    WHERE lower(file_path) LIKE '%.pdf' ORDER BY file_path`).all();
  for (const row of cacheRows) {
    let cachedControlDiagnostics = null;
    try {
      const cachedAnalysis = row.analysis_json ? JSON.parse(row.analysis_json) : null;
      cachedControlDiagnostics = cachedAnalysis?.controls
        ?? (typeof cachedAnalysis?.pageCount === 'number' ? cachedAnalysis.controlDiagnostics : null)
        ?? null;
    } catch {
      cachedControlDiagnostics = null;
    }
    const storageKey = path.basename(path.dirname(row.file_path));
    const source = db.prepare(`SELECT nodus_id,source_ref,attachment_key FROM work_text_sources
      WHERE attachment_key=? ORDER BY active DESC LIMIT 1`).get(storageKey) ?? preflightAttachments.get(storageKey);
    if (!source) continue;
    const marked = row.text.replace(/\[\[p\.\s*(\d+)\]\]/gi, '[[src:s1 p.$1]]');
    const sourceMap = { s1: source.source_ref };
    const current = db.prepare(`SELECT current_version_id versionId,pipeline_version pipelineVersion,status
      FROM document_profile_state WHERE nodus_id=? AND current_version_id IS NOT NULL`).get(source.nodus_id);
    if (current) {
      const chunks = planRetrievalChunks(marked, { sourceMap });
      const supports = db.prepare(`SELECT support_id,passage_id,source_ref,page_start_number,page_end_number,
        char_start,char_end,quote,validation_status FROM document_profile_support WHERE version_id=?`).all(current.versionId);
      const passages = new Map(db.prepare(`SELECT passage_id,nodus_id,text,source_ref,page_number
        FROM passages WHERE nodus_id=?`).all(source.nodus_id).map((passage) => [passage.passage_id, passage]));
      const failures = [];
      for (const support of supports) {
        if (support.validation_status !== 'valid'
          || !literalSourceSpanMatches(marked, support.quote, support.char_start, support.char_end)) {
          failures.push({ supportId: support.support_id, reason: 'invalid_range' });
          continue;
        }
        const selected = selectLiteralPassageIndex(
          chunks,
          support.quote,
          { charStart: support.char_start, charEnd: support.char_end, exact: false },
          support.source_ref,
          support.page_start_number,
        );
        const expected = selected == null ? null : `${source.nodus_id}#${selected}`;
        if (support.passage_id !== expected) failures.push({ supportId: support.support_id, reason: 'passage_selection', actual: support.passage_id, expected });
        if (support.passage_id) {
          const passage = passages.get(support.passage_id);
          if (!passage || passage.source_ref !== support.source_ref || !textCanonicallyContainsLiteral(passage.text, support.quote)) {
            failures.push({ supportId: support.support_id, reason: 'passage_containment' });
          }
        }
      }
      assert.deepEqual(failures, []);
      report.profiles.push({ nodusId: source.nodus_id, ...current, supports: supports.length, failures });
    }

    const extracted = await extractPdfStreaming(row.file_path, {
      ocr: { enabled: false, languages: 'spa+eng', maxPages: 0 },
    });
    const disallowed = (extracted.text.match(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g) ?? []).length;
    const rawDisallowed = (row.text.match(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g) ?? []).length;
    const replacementCharacters = (extracted.text.match(/\uFFFD/g) ?? []).length;
    assert.equal(disallowed, 0);
    const expectedReplacements = cachedControlDiagnostics?.replacements ?? rawDisallowed;
    assert.equal(extracted.controlDiagnostics?.replacements ?? 0, expectedReplacements);
    assert.ok(replacementCharacters >= expectedReplacements);
    assert.ok(/\[\[p\.\s*1\]\]/.test(extracted.text), 'page markers survive extraction control cleanup');
    const chunks = planRetrievalChunks(extracted.text, { sourceMap });
    assert.ok(chunks.length > 0);
    report.documents.push({
      nodusId: source.nodus_id,
      attachmentKey: source.attachment_key,
      filePath: row.file_path,
      priorCacheVersion: row.cache_version,
      rawControlCharacters: rawDisallowed,
      cachedControlReplacements: cachedControlDiagnostics?.replacements ?? null,
      replacementCharacters,
      diagnostics: extracted.controlDiagnostics ?? null,
      pages: extracted.analysis?.pageCount ?? null,
      characters: extracted.text.length,
      chunks: chunks.length,
    });
  }
  assert.equal(report.documents.length, 3, 'the complete three-PDF canary corpus must be exercised');
  report.ok = true;
} finally {
  db.close();
}
if (reportPath) fs.writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`, { mode: 0o600 });
process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);

function arg(name) {
  const index = process.argv.indexOf(`--${name}`);
  const value = index >= 0 ? process.argv[index + 1] : null;
  if (!value) throw new Error(`Missing --${name}`);
  return value;
}

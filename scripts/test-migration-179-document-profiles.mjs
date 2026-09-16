import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { createRequire } from 'node:module';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);
const marker = '--electron-migration-179-test';
if (!process.argv.includes(marker)) {
  execFileSync(path.join(repoRoot, 'node_modules/.bin/electron'), [fileURLToPath(import.meta.url), marker], {
    cwd: repoRoot, env: { ...process.env, ELECTRON_RUN_AS_NODE: '1' }, stdio: 'inherit',
  });
  process.exit(0);
}

const root = await mkdtemp(path.join(os.tmpdir(), 'nodus-migration-179-'));
installTsHook();
try {
  const Database = require('better-sqlite3');
  const { migrations, runMigrations, SCHEMA_VERSION } = require(path.join(repoRoot, 'electron/db/migrations.ts'));
  const db = new Database(path.join(root, 'fixture.sqlite'));
  for (const migration of migrations.filter((entry) => entry.version <= 178)) {
    db.transaction(() => {
      db.exec(migration.up);
      migration.after?.(db);
      db.pragma(`user_version = ${migration.version}`);
    })();
  }
  const work = db.prepare(`INSERT INTO works(
    nodus_id,zotero_key,title,authors_json,item_type,source_type,archived,light_status,deep_status,summary_status,
    resolved_source_type,text_block_reason,resolved_text_notes
  ) VALUES(?,?,?,?,?,'pdf',0,'done','done','none',?,?,?)`);
  work.run('valid','Z-valid','Valid profile','[]','journalArticle','pdf',null,null);
  work.run('orphan','Z-orphan','Orphan pointer','[]','journalArticle','pdf',null,null);
  work.run('wrong','Z-wrong','Wrong-work pointer','[]','journalArticle','pdf',null,null);
  work.run('abstract','Z-abstract','Abstract only','[]','journalArticle','abstract_only','abstract_only','Only an abstract exists.');
  work.run('provider','Z-provider','Provider mentioned abstract','[]','journalArticle','pdf',null,null);
  work.run('missing','Z-missing','Missing attachment','[]','journalArticle','abstract_only','file_missing','Attachment missing.');
  const now = new Date().toISOString();
  db.prepare(`INSERT INTO document_profile_versions(
    version_id,nodus_id,state,source_fingerprint,pipeline_version,schema_version,presentation_language,
    overview,profile_json,prompt_hash,quality_score,created_at,published_at
  ) VALUES('version-valid','valid','current','source-valid','document-profile/5',2,'en','Valid','{}','prompt',1,?,?)`).run(now,now);
  const state = db.prepare(`INSERT INTO document_profile_state(
    nodus_id,current_version_id,status,source_fingerprint,pipeline_version,stale_reason,error,updated_at
  ) VALUES(?,?,?,?,?,?,?,?)`);
  state.run('valid','version-valid','failed','source-valid','document-profile/5',null,'refresh failed',now);
  state.run('orphan','version-missing','failed','source-orphan','document-profile/5',null,'old failure',now);
  state.run('wrong','version-valid','failed','source-wrong','document-profile/5',null,'old failure',now);
  state.run('abstract',null,'failed',null,null,null,'Solo abstract disponible.',now);
  state.run('provider',null,'unavailable',null,null,null,'Provider failed while abstracting output.',now);
  state.run('missing',null,'failed',null,null,null,'Archivo adjunto no encontrado.',now);

  runMigrations(db);
  assert.equal(db.pragma('user_version', { simple: true }), SCHEMA_VERSION);
  assert.equal(SCHEMA_VERSION, 180);
  const rows = Object.fromEntries(db.prepare('SELECT nodus_id,current_version_id,status,error FROM document_profile_state').all()
    .map((row) => [row.nodus_id, row]));
  assert.equal(rows.valid.status, 'current', JSON.stringify(rows.valid));
  assert.equal(rows.valid.current_version_id, 'version-valid');
  assert.equal(rows.orphan.status, 'failed');
  assert.equal(rows.orphan.current_version_id, null);
  assert.equal(rows.orphan.error, 'INVALID_CURRENT_PROFILE_POINTER');
  assert.equal(rows.wrong.status, 'failed');
  assert.equal(rows.wrong.current_version_id, null);
  assert.equal(rows.abstract.status, 'unavailable');
  assert.equal(rows.missing.status, 'unavailable');
  assert.equal(rows.provider.status, 'failed', 'structured full text reverses v178 prose-based misclassification');
  db.prepare(`INSERT OR REPLACE INTO library_analysis_freshness(
    work_id,component,freshness,fingerprint,reason,updated_at
  ) VALUES('valid','passages','current','old',NULL,?),
          ('valid','documentProfile','current','old',NULL,?)`).run(now, now);
  db.prepare("UPDATE works SET resolved_text_hash='replacement-source' WHERE nodus_id='valid'").run();
  const stale = db.prepare('SELECT status,stale_reason FROM document_profile_state WHERE nodus_id=?').get('valid');
  assert.deepEqual(stale, { status: 'stale', stale_reason: 'source_changed' });
  assert.deepEqual(
    db.prepare(`SELECT component,freshness,reason FROM library_analysis_freshness
      WHERE work_id='valid' ORDER BY component`).all(),
    [
      { component: 'documentProfile', freshness: 'stale', reason: 'resolved_source_changed' },
      { component: 'passages', freshness: 'stale', reason: 'resolved_source_changed' },
    ],
  );
  assert.equal(db.pragma('integrity_check', { simple: true }), 'ok');
  db.close();
} finally {
  await rm(root, { recursive: true, force: true });
}

function installTsHook() {
  const ts = require('typescript');
  const Module = require('node:module');
  const originalResolveFilename = Module._resolveFilename;
  Module._resolveFilename = function resolveFilename(request, parent, isMain, options) {
    if (request.startsWith('@shared/')) return path.join(repoRoot, `${request.replace('@shared/', 'shared/')}.ts`);
    return originalResolveFilename.call(this, request, parent, isMain, options);
  };
  require.extensions['.ts'] = function loadTs(module, filename) {
    const source = fs.readFileSync(filename, 'utf8');
    module._compile(ts.transpileModule(source, {
      compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true }, fileName: filename,
    }).outputText, filename);
  };
}

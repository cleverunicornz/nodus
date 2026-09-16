import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { createRequire } from 'node:module';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { installRuntimeHooks, requireElectronRuntime } from './lib/tsRuntimeHooks.mjs';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);

if (!requireElectronRuntime(path.join(repoRoot, 'scripts/test-server-snapshot-revision.mjs'), '--electron-snapshot-revision')) {
  process.exit(0);
}

const root = await mkdtemp(path.join(os.tmpdir(), 'nodus-snapshot-revision-'));
installRuntimeHooks(root);
const Database = require('better-sqlite3');
const { runMigrations } = require(path.join(repoRoot, 'electron/db/migrations.ts'));
const { buildServerSnapshot } = require(path.join(repoRoot, 'electron/serverSync/serverSnapshot.ts'));

test('streaming revision matches the former whole-object JSON digest', () => {
  const db = new Database(path.join(root, 'vault.sqlite'));
  try {
    runMigrations(db);
    const vault = { id: 'v1', name: 'Sintética', type: 'academic' };
    const built = buildServerSnapshot(
      vault,
      { nodusServerIncludeUserContent: true, nodusServerIncludePassages: true },
      db,
      null,
    );
    const payload = JSON.parse(built.buffer.toString('utf8'));
    const expected = createHash('sha256').update(JSON.stringify({
      vault: payload.vault,
      schemaVersion: payload.schemaVersion,
      assets: payload.assets,
      library: null,
      tables: payload.tables,
    })).digest('base64url');
    assert.equal(built.revision, expected);

    const second = buildServerSnapshot(
      vault,
      { nodusServerIncludeUserContent: true, nodusServerIncludePassages: true },
      db,
      null,
    );
    assert.equal(second.revision, built.revision, 'generatedAt is excluded from the revision');
  } finally {
    db.close();
  }
});

test('server snapshots publish only the selected current document profile version', () => {
  const db = new Database(path.join(root, 'profile-history.sqlite'));
  try {
    runMigrations(db);
    db.prepare("INSERT INTO works(nodus_id,zotero_key,title) VALUES('w1','Z1','Profile history')").run();
    const insertVersion = db.prepare(`INSERT INTO document_profile_versions(
      version_id,nodus_id,state,source_fingerprint,pipeline_version,schema_version,presentation_language,
      overview,profile_json,prompt_hash,created_at,published_at
    ) VALUES(?,?,?,?,?,2,'en',?,'{}','prompt','2026-01-01','2026-01-01')`);
    insertVersion.run('version-old','w1','superseded','source','document-profile/5','Old overview');
    insertVersion.run('version-current','w1','current','source','document-profile/5','Current overview');
    db.prepare(`INSERT INTO document_profile_state(
      nodus_id,current_version_id,status,source_fingerprint,pipeline_version,updated_at
    ) VALUES('w1','version-current','current','source','document-profile/5','2026-01-01')`).run();
    const insertField = db.prepare(`INSERT INTO document_profile_fields(
      field_id,version_id,nodus_id,kind,ordinal,text,confidence,centrality,created_at
    ) VALUES(?,?,?,?,0,?,1,1,'2026-01-01')`);
    insertField.run('field-old','version-old','w1','thesis','OLD_PROFILE_TEXT');
    insertField.run('field-current','version-current','w1','thesis','CURRENT_PROFILE_TEXT');
    const payload = JSON.parse(buildServerSnapshot(
      { id: 'v1', name: 'Synthetic', type: 'academic' },
      { nodusServerIncludeUserContent: true, nodusServerIncludePassages: false },
      db,
      null,
    ).buffer.toString('utf8'));
    assert.deepEqual(payload.tables.document_profile_versions.map((row) => row.version_id), ['version-current']);
    assert.deepEqual(payload.tables.document_profile_fields.map((row) => row.field_id), ['field-current']);
    assert.doesNotMatch(JSON.stringify(payload.tables), /OLD_PROFILE_TEXT/);
  } finally {
    db.close();
  }
});

test('revision hashing streams values instead of materialising the full object again', () => {
  const source = fs.readFileSync(path.join(repoRoot, 'electron/serverSync/serverSnapshot.ts'), 'utf8');
  assert.match(source, /updateJsonHash\(revisionHash,/);
  assert.doesNotMatch(source, /\.update\(JSON\.stringify\(\{/);
});

test.after(async () => {
  await rm(root, { recursive: true, force: true });
});

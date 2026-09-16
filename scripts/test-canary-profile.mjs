import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { createRequire } from 'node:module';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const marker = '--electron-canary-profile-test';
if (!process.argv.includes(marker)) {
  execFileSync(path.join(repoRoot, 'node_modules/.bin/electron'), [fileURLToPath(import.meta.url), marker], {
    cwd: repoRoot,
    env: { ...process.env, ELECTRON_RUN_AS_NODE: '1' },
    stdio: 'inherit',
  });
  process.exit(0);
}

const require = createRequire(import.meta.url);
const Database = require('better-sqlite3');
const {
  activeVault,
  assertCanaryIsolation,
  assertOpenedCanaryDatabase,
  canaryEnvironment,
  cloneCanaryProfile,
} = await import(`./canary-profile.mjs?v=${Date.now()}`);
const root = await mkdtemp(path.join(os.tmpdir(), 'nodus-canary-profile-'));
test.after(async () => rm(root, { recursive: true, force: true }));

function seedProfile(name) {
  const profile = path.join(root, name);
  fs.mkdirSync(profile, { recursive: true });
  const dbPath = path.join(profile, 'nodus.sqlite');
  const db = new Database(dbPath);
  db.pragma('journal_mode = WAL');
  db.exec(`
    CREATE TABLE proof(id INTEGER PRIMARY KEY,value TEXT);
    CREATE TABLE works(deep_queued INTEGER NOT NULL DEFAULT 0);
    CREATE TABLE document_index_jobs(status TEXT NOT NULL);
    CREATE TABLE study_knowledge_jobs(status TEXT NOT NULL);
  `);
  db.prepare('INSERT INTO proof(value) VALUES(?)').run('committed-wal-row');
  fs.writeFileSync(path.join(profile, 'vaults.json'), `${JSON.stringify({
    formatVersion: 1,
    activeVaultId: 'default',
    vaults: [{ id: 'default', name: 'Principal', path: dbPath, type: 'academic', origin: 'local' }],
  }, null, 2)}\n`);
  fs.writeFileSync(path.join(profile, 'app-prefs.json'), '{"uiLanguage":"en"}\n');
  fs.writeFileSync(path.join(profile, 'preflight-report.json'), '{"format":"fixture"}\n');
  fs.mkdirSync(path.join(profile, 'secrets'), { recursive: true });
  fs.writeFileSync(path.join(profile, 'secrets', 'ai_key_openrouter.bin'), 'must-not-copy');
  fs.writeFileSync(path.join(profile, 'deep-research-queue.v1.json'), '[{"id":"stale"}]\n');
  fs.writeFileSync(path.join(profile, 'unrelated-run-report.json'), '{"stale":true}\n');
  fs.writeFileSync(path.join(profile, 'LOCK'), 'transient');
  return { profile, dbPath, db };
}

test('canary clone snapshots WAL and rewrites one isolated active vault', async () => {
  const source = seedProfile('source');
  const destination = path.join(root, 'clone');
  try {
    const result = await cloneCanaryProfile({
      sourceProfile: source.profile,
      destinationProfile: destination,
      expectedSourceDb: source.dbPath,
      Database,
      target: { itemKey: 'D2JS72TI', nodusId: 'work-1' },
      buildIdentity: { version: 'test', commit: 'fixture' },
      approval: { approvalId: 'approval-fixture', maxRequests: 1 },
    });
    const clone = new Database(result.destinationDb, { readonly: true, fileMustExist: true });
    try {
      assert.equal(clone.prepare('SELECT value FROM proof').get().value, 'committed-wal-row');
      assert.equal(assertOpenedCanaryDatabase(clone, result.destinationDb), result.destinationDb);
    } finally {
      clone.close();
    }
    assert.equal(fs.existsSync(path.join(destination, 'LOCK')), false);
    assert.equal(fs.readFileSync(path.join(destination, 'app-prefs.json'), 'utf8'), '{"uiLanguage":"en"}\n');
    assert.equal(fs.existsSync(path.join(destination, 'secrets')), false);
    assert.equal(fs.existsSync(path.join(destination, 'deep-research-queue.v1.json')), false);
    assert.equal(fs.existsSync(path.join(destination, 'unrelated-run-report.json')), false);
    const { registry, vault } = activeVault(path.join(destination, 'vaults.json'));
    assert.equal(registry.vaults.length, 1);
    assert.equal(vault.path, result.destinationDb);
    const manifest = JSON.parse(fs.readFileSync(result.manifestPath, 'utf8'));
    assert.equal(manifest.databasePath, result.destinationDb);
    assert.equal(manifest.target.itemKey, 'D2JS72TI');
    assert.equal(manifest.approval.approvalId, 'approval-fixture');
    assert.equal(manifest.formatVersion, 2);
    assert.equal(manifest.isolation.automaticQueuesDisabled, true);
    assert.equal(manifest.isolation.credentialsCopied, false);
    assert.equal(manifest.isolation.externalRendererNetworkBlocked, true);
    assert.deepEqual(manifest.isolation.profileAllowlist, ['app-prefs.json', 'preflight-report.json', 'vaults.json']);
    const env = canaryEnvironment(destination, { CUSTOM: 'yes' });
    assert.equal(env.NODUS_USERDATA, result.destinationRoot);
    assert.equal(env.NODUS_STELLAR_PREVIEW, '1');
    assert.equal(env.NODUS_E2E_DISABLE_STUDY_BACKGROUND_AI, '1');
    assert.equal(env.NODUS_CANARY_BLOCK_EXTERNAL_RENDERER_NETWORK, '1');
    assert.equal(env.CUSTOM, 'yes');
  } finally {
    source.db.close();
  }
});

test('canary clone refuses source profiles with active automatic work', async () => {
  const source = seedProfile('source-active-queue');
  try {
    source.db.prepare('INSERT INTO document_index_jobs(status) VALUES(?)').run('running');
    await assert.rejects(cloneCanaryProfile({
      sourceProfile: source.profile,
      destinationProfile: path.join(root, 'clone-active-queue'),
      expectedSourceDb: source.dbPath,
      Database,
      target: { itemKey: 'K4CDC9T5' },
      buildIdentity: { version: 'test', commit: 'fixture' },
    }), /active document index queue item/);
  } finally {
    source.db.close();
  }
});

test('canary guards reject registry aliasing and wrong opened databases', async () => {
  const source = seedProfile('source-wrong-path');
  const cloneRoot = path.join(root, 'manual-clone');
  fs.mkdirSync(cloneRoot, { recursive: true });
  const cloneDbPath = path.join(cloneRoot, 'nodus.sqlite');
  const cloneDb = new Database(cloneDbPath);
  cloneDb.exec('CREATE TABLE proof(id INTEGER PRIMARY KEY);');
  cloneDb.close();
  fs.writeFileSync(path.join(cloneRoot, 'vaults.json'), `${JSON.stringify({
    formatVersion: 1,
    activeVaultId: 'default',
    vaults: [{ id: 'default', name: 'Wrong', path: source.dbPath, type: 'academic', origin: 'local' }],
  })}\n`);
  try {
    assert.throws(() => assertCanaryIsolation({
      sourceProfile: source.profile,
      destinationProfile: cloneRoot,
      expectedSourceDb: source.dbPath,
      expectedDestinationDb: cloneDbPath,
    }), /outside the intended canary database/);
    const openedSource = new Database(source.dbPath, { readonly: true, fileMustExist: true });
    try {
      assert.throws(() => assertOpenedCanaryDatabase(openedSource, cloneDbPath), /not the approved canary database/);
    } finally {
      openedSource.close();
    }
  } finally {
    source.db.close();
  }
});

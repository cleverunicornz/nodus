import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

const SAFE_PROFILE_FILES = [
  'app-prefs.json',
  'preflight-report.json',
  'vaults.json',
];

function copySafeProfileState(sourceRoot, destinationRoot) {
  fs.mkdirSync(destinationRoot, { recursive: true, mode: 0o700 });
  for (const name of SAFE_PROFILE_FILES) {
    const source = path.join(sourceRoot, name);
    if (!fs.existsSync(source)) continue;
    fs.copyFileSync(source, path.join(destinationRoot, name));
  }
}

function assertAutomaticQueuesIdle(db) {
  const checks = [
    ['document index', "SELECT count(*) AS count FROM document_index_jobs WHERE status IN ('pending', 'queued', 'running')"],
    ['deep scan', 'SELECT count(*) AS count FROM works WHERE deep_queued <> 0'],
    ['study knowledge', "SELECT count(*) AS count FROM study_knowledge_jobs WHERE status IN ('pending', 'queued', 'running')"],
  ];
  for (const [name, sql] of checks) {
    const count = Number(db.prepare(sql).get()?.count ?? 0);
    assert.equal(count, 0, `Source profile has ${count} active ${name} queue item(s)`);
  }
}

async function sha256File(file) {
  if (!fs.existsSync(file)) return null;
  const hash = createHash('sha256');
  await new Promise((resolve, reject) => {
    const stream = fs.createReadStream(file);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('error', reject);
    stream.on('end', resolve);
  });
  return hash.digest('hex');
}

async function sourceDatabaseFingerprint(databasePath) {
  return {
    databaseSha256: await sha256File(databasePath),
    walSha256: await sha256File(`${databasePath}-wal`),
  };
}

function canonicalExisting(value) {
  return fs.realpathSync.native(path.resolve(value));
}

function canonicalProspective(value) {
  const absolute = path.resolve(value);
  const parent = canonicalExisting(path.dirname(absolute));
  return path.join(parent, path.basename(absolute));
}

function sameFile(left, right) {
  try {
    const a = fs.statSync(left);
    const b = fs.statSync(right);
    return a.dev === b.dev && a.ino === b.ino;
  } catch {
    return false;
  }
}

export function activeVault(registryPath) {
  const registry = JSON.parse(fs.readFileSync(registryPath, 'utf8'));
  const vault = registry.vaults?.find((entry) => entry.id === registry.activeVaultId);
  assert.ok(vault, `Active vault ${registry.activeVaultId ?? '<missing>'} is absent from ${registryPath}`);
  return { registry, vault };
}

export function rewriteCanaryVaultRegistry(registryPath, expectedSourceDb, destinationDb) {
  const { registry, vault } = activeVault(registryPath);
  assert.equal(canonicalExisting(vault.path), canonicalExisting(expectedSourceDb),
    'Source registry does not address the explicitly approved source database');
  const rewritten = {
    ...registry,
    activeVaultId: vault.id,
    vaults: [{ ...vault, path: canonicalExisting(destinationDb), lastOpenedAt: new Date().toISOString() }],
  };
  fs.writeFileSync(registryPath, `${JSON.stringify(rewritten, null, 2)}\n`, { mode: 0o600 });
  return rewritten;
}

export function assertCanaryIsolation({ sourceProfile, destinationProfile, expectedSourceDb, expectedDestinationDb }) {
  const sourceRoot = canonicalExisting(sourceProfile);
  const destinationRoot = canonicalExisting(destinationProfile);
  const sourceDb = canonicalExisting(expectedSourceDb);
  const destinationDb = canonicalExisting(expectedDestinationDb);
  assert.notEqual(sourceRoot, destinationRoot, 'Canary user-data root aliases the source profile');
  assert.notEqual(sourceDb, destinationDb, 'Canary database aliases the source database');
  assert.equal(sameFile(sourceDb, destinationDb), false, 'Canary and source databases share one inode');
  assert.equal(destinationDb.startsWith(`${destinationRoot}${path.sep}`), true,
    'Canary database is outside its declared isolated user-data root');
  const { registry, vault } = activeVault(path.join(destinationRoot, 'vaults.json'));
  assert.equal(canonicalExisting(vault.path), destinationDb, 'Cloned registry points outside the intended canary database');
  assert.equal(registry.vaults.length, 1, 'Canary registry must expose exactly one intended vault');
  return { sourceRoot, destinationRoot, sourceDb, destinationDb, vaultId: vault.id };
}

export function assertOpenedCanaryDatabase(db, expectedDatabasePath) {
  const rows = db.pragma('database_list');
  const main = rows.find((row) => row.name === 'main');
  assert.ok(main?.file, 'Opened SQLite connection has no main database path');
  assert.equal(canonicalExisting(main.file), canonicalExisting(expectedDatabasePath),
    'Opened SQLite connection is not the approved canary database');
  return canonicalExisting(main.file);
}

export function canaryEnvironment(profileRoot, extra = {}) {
  return {
    ...process.env,
    ...extra,
    NODUS_USERDATA: canonicalExisting(profileRoot),
    NODUS_DISABLE_AUTO_UPDATE: '1',
    NODUS_DISABLE_ANNOUNCEMENTS: '1',
    NODUS_E2E_UPDATE_STATUS: 'not-available',
    NODUS_E2E_DISABLE_STUDY_BACKGROUND_AI: '1',
    NODUS_CANARY_BLOCK_EXTERNAL_RENDERER_NETWORK: '1',
    NODUS_STELLAR_PREVIEW: '1',
  };
}

export async function cloneCanaryProfile({
  sourceProfile,
  destinationProfile,
  expectedSourceDb,
  Database,
  target,
  buildIdentity,
  approval = null,
}) {
  const sourceRoot = canonicalExisting(sourceProfile);
  const destinationRoot = canonicalProspective(destinationProfile);
  const sourceDb = canonicalExisting(expectedSourceDb);
  assert.equal(fs.existsSync(destinationRoot), false, `Canary destination already exists: ${destinationRoot}`);
  assert.equal(sourceDb.startsWith(`${sourceRoot}${path.sep}`), true,
    'Approved source database is outside the source profile');
  copySafeProfileState(sourceRoot, destinationRoot);
  const sourceFingerprintBefore = await sourceDatabaseFingerprint(sourceDb);
  const destinationDb = path.join(destinationRoot, 'nodus.sqlite');
  const source = new Database(sourceDb, { readonly: true, fileMustExist: true });
  try {
    assertAutomaticQueuesIdle(source);
    await source.backup(destinationDb);
  } finally {
    source.close();
  }
  const sourceFingerprintAfter = await sourceDatabaseFingerprint(sourceDb);
  assert.deepEqual(sourceFingerprintAfter, sourceFingerprintBefore,
    'Source database or WAL changed while the canary snapshot was created');
  rewriteCanaryVaultRegistry(path.join(destinationRoot, 'vaults.json'), sourceDb, destinationDb);
  const isolated = assertCanaryIsolation({
    sourceProfile: sourceRoot,
    destinationProfile: destinationRoot,
    expectedSourceDb: sourceDb,
    expectedDestinationDb: destinationDb,
  });
  const clone = new Database(destinationDb, { readonly: true, fileMustExist: true });
  let integrity;
  let openedDatabasePath;
  try {
    openedDatabasePath = assertOpenedCanaryDatabase(clone, destinationDb);
    integrity = clone.pragma('integrity_check', { simple: true });
  } finally {
    clone.close();
  }
  assert.equal(integrity, 'ok', 'Cloned canary database failed integrity_check');
  const manifest = {
    format: 'nodus.canary-run-manifest',
    formatVersion: 2,
    createdAt: new Date().toISOString(),
    sourceProfile: sourceRoot,
    sourceDatabase: sourceDb,
    sourceFingerprint: sourceFingerprintAfter,
    profileRoot: destinationRoot,
    registryPath: path.join(destinationRoot, 'vaults.json'),
    databasePath: openedDatabasePath,
    vaultId: isolated.vaultId,
    target,
    buildIdentity,
    approval,
    isolation: {
      singleVault: true,
      databaseAliasesSource: false,
      automaticQueuesDisabled: true,
      durableQueueFilesCopied: false,
      credentialsCopied: false,
      unrelatedProfileStateCopied: false,
      externalRendererNetworkBlocked: true,
      profileAllowlist: SAFE_PROFILE_FILES,
    },
  };
  const manifestPath = path.join(destinationRoot, 'canary-run-manifest.json');
  fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, { mode: 0o600 });
  return { ...isolated, manifest, manifestPath };
}

function cliArg(name) {
  const at = process.argv.indexOf(`--${name}`);
  return at >= 0 ? process.argv[at + 1] : null;
}
function cliNumber(name) {
  const raw = cliArg(name);
  if (raw === null) return null;
  const value = Number(raw);
  assert.equal(Number.isFinite(value) && value > 0, true, `--${name} must be a positive number`);
  return value;
}
function cliList(name) {
  return (cliArg(name) ?? '').split(',').map((value) => value.trim()).filter(Boolean);
}



if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const require = createRequire(path.join(process.cwd(), 'package.json'));
  const Database = require('better-sqlite3');
  const target = {
    itemKey: cliArg('target'),
    collectionKey: cliArg('collection-key'),
    nodusId: cliArg('nodus-id'),
    providers: cliList('providers'),
    models: cliList('models'),
    requestClasses: cliList('request-classes'),
    limits: {
      concurrency: cliNumber('concurrency'),
      maxRequests: cliNumber('max-requests'),
      maxEstimatedInputTokens: cliNumber('max-input-tokens'),
      maxEstimatedOutputTokens: cliNumber('max-output-tokens'),
      deadlineMinutes: cliNumber('deadline-minutes'),
    },
  };
  assert.ok(target.itemKey, '--target is required');
  assert.ok(target.providers.length, '--providers is required');
  assert.ok(target.models.length, '--models is required');
  assert.ok(target.requestClasses.length, '--request-classes is required');
  const buildIdentity = {
    label: cliArg('build-id'),
    executableSha256: cliArg('build-sha256'),
    appPath: cliArg('build-app'),
  };
  assert.ok(buildIdentity.label, '--build-id is required');
  assert.ok(buildIdentity.executableSha256, '--build-sha256 is required');
  assert.ok(buildIdentity.appPath, '--build-app is required');
  const result = await cloneCanaryProfile({
    sourceProfile: cliArg('source-profile'),
    destinationProfile: cliArg('destination-profile'),
    expectedSourceDb: cliArg('source-db'),
    target,
    buildIdentity,
    approval: cliArg('approval-id') ? { approvalId: cliArg('approval-id'), limits: target.limits } : null,
    Database,
  });
  process.stdout.write(`${JSON.stringify(result.manifest, null, 2)}\n`);
}

import assert from 'node:assert/strict';
import test from 'node:test';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import { mkdtempSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);
const root = mkdtempSync(path.join(os.tmpdir(), 'nodus-paid-approval-'));
const bundle = path.join(root, 'approval.cjs');
execFileSync(path.join(repoRoot, 'node_modules/.bin/esbuild'), [
  'electron/ai/paidOperationApproval.ts', '--bundle', '--platform=node', '--format=cjs', '--target=node22', `--outfile=${bundle}`,
], { cwd: repoRoot, stdio: 'inherit' });
const approval = require(bundle);
const original = {
  required: process.env.NODUS_REQUIRE_AI_APPROVAL,
  config: process.env.NODUS_AI_APPROVAL_JSON,
  ledger: process.env.NODUS_AI_APPROVAL_LEDGER,
};
test.after(() => {
  if (original.required === undefined) delete process.env.NODUS_REQUIRE_AI_APPROVAL; else process.env.NODUS_REQUIRE_AI_APPROVAL = original.required;
  if (original.config === undefined) delete process.env.NODUS_AI_APPROVAL_JSON; else process.env.NODUS_AI_APPROVAL_JSON = original.config;
  if (original.ledger === undefined) delete process.env.NODUS_AI_APPROVAL_LEDGER; else process.env.NODUS_AI_APPROVAL_LEDGER = original.ledger;
  rmSync(root, { recursive: true, force: true });
});

const descriptor = {
  provider: 'openrouter', model: 'z-ai/glm-5.3-flash', credentialScope: 'scope',
  requestClass: 'background', estimatedInputTokens: 100, estimatedOutputTokens: 200, jobId: 'job-1',
};
const configure = (overrides = {}) => {
  const ledger = path.join(root, `${Math.random().toString(36).slice(2)}.json`);
  process.env.NODUS_REQUIRE_AI_APPROVAL = '1';
  process.env.NODUS_AI_APPROVAL_LEDGER = ledger;
  process.env.NODUS_AI_APPROVAL_JSON = JSON.stringify({
    approvalId: 'approval-canary-1', operation: 'zotero-testing-canary',
    providers: ['openrouter'], models: ['z-ai/glm-5.3-flash'], requestClasses: ['background', 'embedding'],
    maxRequests: 2, maxEstimatedInputTokens: 250, maxEstimatedOutputTokens: 450,
    expiresAt: new Date(Date.now() + 60_000).toISOString(), ...overrides,
  });
  return ledger;
};

test('remote requests reserve one durable worst-case budget before dispatch', () => {
  const ledger = configure();
  approval.reservePaidOperationRequest(descriptor);
  approval.reservePaidOperationRequest(descriptor);
  const stored = JSON.parse(fs.readFileSync(ledger, 'utf8'));
  assert.deepEqual(
    { requests: stored.requests, input: stored.estimatedInputTokens, output: stored.estimatedOutputTokens },
    { requests: 2, input: 200, output: 400 },
  );
  assert.throws(() => approval.reservePaidOperationRequest(descriptor), /excede el presupuesto/);
  assert.equal(JSON.parse(fs.readFileSync(ledger, 'utf8')).requests, 2, 'a refused request never consumes or resets the ledger');
});

test('provider, model, class, expiry and malformed ledgers fail closed', () => {
  configure();
  assert.throws(() => approval.reservePaidOperationRequest({ ...descriptor, model: 'another-model' }), /fuera de la aprobación/);
  configure({ expiresAt: new Date(Date.now() - 1_000).toISOString() });
  assert.throws(() => approval.reservePaidOperationRequest(descriptor), /caducado/);
  const ledger = configure();
  fs.writeFileSync(ledger, '{}');
  assert.throws(() => approval.reservePaidOperationRequest(descriptor), /otra aprobación|contadores inválidos/);
});

test('approval-required mode denies absent approval while local inference remains available', () => {
  process.env.NODUS_REQUIRE_AI_APPROVAL = '1';
  delete process.env.NODUS_AI_APPROVAL_JSON;
  delete process.env.NODUS_AI_APPROVAL_LEDGER;
  assert.throws(() => approval.reservePaidOperationRequest(descriptor), /exige una aprobación/);
  assert.doesNotThrow(() => approval.reservePaidOperationRequest({ ...descriptor, provider: 'nodus', model: 'local-model' }));
});

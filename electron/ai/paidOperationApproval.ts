import fs from 'node:fs';
import path from 'node:path';
import type { AiRequestDescriptor } from './aiRequestGate';

interface PaidOperationApproval {
  approvalId: string;
  operation: string;
  providers: string[];
  models: string[];
  requestClasses: AiRequestDescriptor['requestClass'][];
  maxRequests: number;
  maxEstimatedInputTokens: number;
  maxEstimatedOutputTokens: number;
  expiresAt: string;
}

interface PaidOperationLedger {
  approvalId: string;
  operation: string;
  requests: number;
  estimatedInputTokens: number;
  estimatedOutputTokens: number;
  updatedAt: string;
}

export class PaidOperationApprovalError extends Error {
  readonly code = 'auth';
  readonly retriable = false;
  readonly config = true;

  constructor(message: string) {
    super(message);
    this.name = 'PaidOperationApprovalError';
  }
}

const LOCAL_UNBILLED_PROVIDERS: Record<string, true> = {
  nodus: true,
  ollama: true,
  lmstudio: true,
};

function approvalFromEnvironment(): { approval: PaidOperationApproval; ledgerPath: string } {
  const raw = process.env.NODUS_AI_APPROVAL_JSON;
  const ledgerPath = process.env.NODUS_AI_APPROVAL_LEDGER;
  if (!raw || !ledgerPath || !path.isAbsolute(ledgerPath)) {
    throw new PaidOperationApprovalError('Esta ejecución exige una aprobación de gasto explícita y un ledger absoluto; no se envió ninguno válido.');
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new PaidOperationApprovalError('La aprobación de gasto no es JSON válido.');
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new PaidOperationApprovalError('La aprobación de gasto no tiene una estructura válida.');
  }
  const value = parsed as Record<string, unknown>;
  const approval: PaidOperationApproval = {
    approvalId: typeof value.approvalId === 'string' ? value.approvalId : '',
    operation: typeof value.operation === 'string' ? value.operation : '',
    providers: Array.isArray(value.providers) ? value.providers.filter((entry): entry is string => typeof entry === 'string') : [],
    models: Array.isArray(value.models) ? value.models.filter((entry): entry is string => typeof entry === 'string') : [],
    requestClasses: Array.isArray(value.requestClasses)
      ? value.requestClasses.filter((entry): entry is AiRequestDescriptor['requestClass'] =>
        entry === 'interactive' || entry === 'background' || entry === 'fusion' || entry === 'embedding')
      : [],
    maxRequests: Number(value.maxRequests),
    maxEstimatedInputTokens: Number(value.maxEstimatedInputTokens),
    maxEstimatedOutputTokens: Number(value.maxEstimatedOutputTokens),
    expiresAt: typeof value.expiresAt === 'string' ? value.expiresAt : '',
  };
  if (!approval.approvalId || !approval.operation || !approval.providers.length || !approval.models.length
    || !approval.requestClasses.length || !Number.isInteger(approval.maxRequests) || approval.maxRequests < 1
    || !Number.isFinite(approval.maxEstimatedInputTokens) || approval.maxEstimatedInputTokens < 0
    || !Number.isFinite(approval.maxEstimatedOutputTokens) || approval.maxEstimatedOutputTokens < 0
    || !Number.isFinite(Date.parse(approval.expiresAt))) {
    throw new PaidOperationApprovalError('La aprobación de gasto está incompleta o contiene límites inválidos.');
  }
  if (Date.now() >= Date.parse(approval.expiresAt)) {
    throw new PaidOperationApprovalError(`La aprobación de gasto «${approval.approvalId}» ha caducado.`);
  }
  return { approval, ledgerPath };
}

/** Reserve worst-case request budget synchronously before a remote dispatch. Retries and
 * embeddings pass through this same seam, and the ledger survives process restarts. */
export function reservePaidOperationRequest(descriptor: AiRequestDescriptor): void {
  if (process.env.NODUS_REQUIRE_AI_APPROVAL !== '1' || LOCAL_UNBILLED_PROVIDERS[descriptor.provider]) return;
  const { approval, ledgerPath } = approvalFromEnvironment();
  if (!approval.providers.includes(descriptor.provider)
    || !approval.models.includes(descriptor.model)
    || !approval.requestClasses.includes(descriptor.requestClass)) {
    throw new PaidOperationApprovalError(
      `La solicitud ${descriptor.provider}/${descriptor.model} (${descriptor.requestClass}) queda fuera de la aprobación «${approval.approvalId}».`,
    );
  }
  let ledger: PaidOperationLedger = {
    approvalId: approval.approvalId,
    operation: approval.operation,
    requests: 0,
    estimatedInputTokens: 0,
    estimatedOutputTokens: 0,
    updatedAt: new Date().toISOString(),
  };
  if (fs.existsSync(ledgerPath)) {
    let stored: unknown;
    try {
      stored = JSON.parse(fs.readFileSync(ledgerPath, 'utf8'));
    } catch {
      throw new PaidOperationApprovalError('El ledger de gasto existente no se puede validar; la ejecución se detuvo sin reservar otra solicitud.');
    }
    if (!stored || typeof stored !== 'object' || Array.isArray(stored)) {
      throw new PaidOperationApprovalError('El ledger de gasto existente tiene una estructura inválida.');
    }
    const value = stored as Record<string, unknown>;
    if (value.approvalId !== approval.approvalId || value.operation !== approval.operation) {
      throw new PaidOperationApprovalError('El ledger pertenece a otra aprobación u operación; no se puede reiniciar el presupuesto cambiando el proceso.');
    }
    ledger = {
      approvalId: approval.approvalId,
      operation: approval.operation,
      requests: Number(value.requests),
      estimatedInputTokens: Number(value.estimatedInputTokens),
      estimatedOutputTokens: Number(value.estimatedOutputTokens),
      updatedAt: typeof value.updatedAt === 'string' ? value.updatedAt : '',
    };
    if (![ledger.requests, ledger.estimatedInputTokens, ledger.estimatedOutputTokens].every(Number.isFinite)) {
      throw new PaidOperationApprovalError('El ledger de gasto contiene contadores inválidos.');
    }
  }
  const next: PaidOperationLedger = {
    ...ledger,
    requests: ledger.requests + 1,
    estimatedInputTokens: ledger.estimatedInputTokens + Math.max(0, descriptor.estimatedInputTokens ?? 0),
    estimatedOutputTokens: ledger.estimatedOutputTokens + Math.max(0, descriptor.estimatedOutputTokens ?? 0),
    updatedAt: new Date().toISOString(),
  };
  if (next.requests > approval.maxRequests
    || next.estimatedInputTokens > approval.maxEstimatedInputTokens
    || next.estimatedOutputTokens > approval.maxEstimatedOutputTokens) {
    throw new PaidOperationApprovalError(`La solicitud excede el presupuesto aprobado «${approval.approvalId}»; no se envió al proveedor.`);
  }
  fs.mkdirSync(path.dirname(ledgerPath), { recursive: true });
  const temporary = `${ledgerPath}.tmp-${process.pid}`;
  fs.writeFileSync(temporary, `${JSON.stringify(next, null, 2)}\n`, { encoding: 'utf8', mode: 0o600 });
  fs.renameSync(temporary, ledgerPath);
}

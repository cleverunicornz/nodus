const UNKNOWN_CONTENT = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F\uFFFD]/u;
const UNKNOWN_CONTENT_GLOBAL = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F\uFFFD]+/gu;
const graphemes = new Intl.Segmenter('und', { granularity: 'grapheme' });

export interface SourceTextSpan {
  /** Zero-based UTF-16 offsets into the exact marked source string. */
  charStart: number;
  charEnd: number;
  exact: boolean;
}

interface CanonicalUnit {
  start: number;
  end: number;
}

interface CanonicalText {
  text: string;
  units: CanonicalUnit[];
  hasControl: boolean;
}

export interface LiteralPassageCandidate {
  text: string;
  charStart: number;
  charEnd: number;
  sourceRef: string | null;
  pageNumber: number | null;
  chunkIndex?: number;
  passageId?: string;
}

function appendWhitespace(result: CanonicalText, start: number, end: number): void {
  if (result.text.endsWith(' ')) {
    result.units[result.units.length - 1].end = end;
    return;
  }
  result.text += ' ';
  result.units.push({ start, end });
}

function appendRun(result: CanonicalText, value: string, baseOffset: number): void {
  for (const segment of graphemes.segment(value)) {
    const raw = segment.segment;
    const start = baseOffset + segment.index;
    const end = start + raw.length;
    if (UNKNOWN_CONTENT.test(raw)) {
      // Controls are evidence boundaries. Keeping an unmatched sentinel prevents a
      // tolerant comparison from manufacturing a quotation across a damaged glyph.
      result.text += '\u0000';
      result.units.push({ start, end });
      result.hasControl = true;
      continue;
    }
    if (/^\s+$/u.test(raw)) {
      appendWhitespace(result, start, end);
      continue;
    }
    const normalized = raw.normalize('NFC');
    result.text += normalized;
    for (let index = 0; index < normalized.length; index += 1) result.units.push({ start, end });
  }
}

function canonicalize(value: string): CanonicalText {
  const result: CanonicalText = { text: '', units: [], hasControl: false };
  let cursor = 0;
  const markers = value.matchAll(/\[\[(?:src:[^\]\s]+(?:\s+p\.\s*\d+)?|p\.\s*\d+)\]\]/giu);
  for (const marker of markers) {
    const start = marker.index ?? 0;
    if (start > cursor) appendRun(result, value.slice(cursor, start), cursor);
    appendWhitespace(result, start, start + marker[0].length);
    cursor = start + marker[0].length;
  }
  if (cursor < value.length) appendRun(result, value.slice(cursor), cursor);
  return result;
}

function allIndexes(value: string, needle: string): number[] {
  const indexes: number[] = [];
  for (let index = value.indexOf(needle); index >= 0; index = value.indexOf(needle, index + 1)) indexes.push(index);
  return indexes;
}

function boundedScope(text: string, scope?: { charStart: number; charEnd: number } | null): { charStart: number; charEnd: number } {
  if (!scope) return { charStart: 0, charEnd: text.length };
  return {
    charStart: Math.max(0, Math.min(text.length, Math.trunc(scope.charStart))),
    charEnd: Math.max(0, Math.min(text.length, Math.trunc(scope.charEnd))),
  };
}

/**
 * Locate complete literal quotations in one resolved, marker-bearing source.
 *
 * Exact case-sensitive occurrences win. When layout whitespace or Unicode
 * composition differs, the fallback uses NFC plus collapsed whitespace and ignores
 * generated source/page markers. It never folds case, punctuation, accents,
 * mathematical symbols, or controls.
 */
export function findLiteralSourceSpans(
  text: string,
  quote: string,
  scope?: { charStart: number; charEnd: number } | null,
): SourceTextSpan[] {
  if (!quote || UNKNOWN_CONTENT.test(quote)) return [];
  const bounds = boundedScope(text, scope);
  if (bounds.charEnd <= bounds.charStart) return [];

  const exact = allIndexes(text, quote)
    .filter((charStart) => charStart >= bounds.charStart && charStart + quote.length <= bounds.charEnd)
    .map((charStart) => ({ charStart, charEnd: charStart + quote.length, exact: true }));
  if (exact.length) return exact;

  const source = canonicalize(text);
  const canonicalQuote = canonicalize(quote);
  if (canonicalQuote.hasControl) return [];
  const needle = canonicalQuote.text.trim();
  if (!needle) return [];

  const matches = new Map<string, SourceTextSpan>();
  for (const canonicalStart of allIndexes(source.text, needle)) {
    const canonicalEnd = canonicalStart + needle.length;
    const first = source.units[canonicalStart];
    const last = source.units[canonicalEnd - 1];
    if (!first || !last) continue;
    const charStart = first.start;
    const charEnd = last.end;
    if (charStart < bounds.charStart || charEnd > bounds.charEnd) continue;
    matches.set(`${charStart}:${charEnd}`, { charStart, charEnd, exact: false });
  }
  return [...matches.values()].sort((left, right) => left.charStart - right.charStart || left.charEnd - right.charEnd);
}

export function literalSourceSpanMatches(text: string, quote: string, charStart: number, charEnd: number): boolean {
  return findLiteralSourceSpans(text, quote, { charStart, charEnd })
    .some((span) => span.charStart === charStart && span.charEnd === charEnd);
}

export function textCanonicallyContainsLiteral(text: string, quote: string): boolean {
  return findLiteralSourceSpans(text, quote).length > 0;
}
/** Select one continuous excerpt that contains no unknown extracted glyphs. */
export function intactSourceExcerpt(value: string, maxLength: number): string {
  if (maxLength <= 0) return '';
  const segments = value.split(UNKNOWN_CONTENT_GLOBAL)
    .map((text, index) => ({ text: text.trim(), index }))
    .filter((entry) => entry.text.length > 0)
    .sort((left, right) => right.text.length - left.text.length || left.index - right.index);
  return (segments[0]?.text ?? '').slice(0, maxLength).trim();
}


/** Choose a deterministic, same-source passage that fully contains one occurrence. */
export function selectLiteralPassageIndex(
  candidates: LiteralPassageCandidate[],
  quote: string,
  span: SourceTextSpan,
  sourceRef: string | null,
  pageNumber: number | null,
): number | null {
  const eligible = candidates.map((candidate, index) => ({ candidate, index })).filter(({ candidate }) =>
    candidate.sourceRef === sourceRef
    && candidate.charStart <= span.charStart
    && candidate.charEnd >= span.charEnd
    && textCanonicallyContainsLiteral(candidate.text, quote)
  );
  eligible.sort((left, right) => {
    const leftPage = pageNumber != null && left.candidate.pageNumber === pageNumber ? 0 : 1;
    const rightPage = pageNumber != null && right.candidate.pageNumber === pageNumber ? 0 : 1;
    return leftPage - rightPage
      || (left.candidate.chunkIndex ?? left.index) - (right.candidate.chunkIndex ?? right.index)
      || left.candidate.charStart - right.candidate.charStart
      || (left.candidate.passageId ?? '').localeCompare(right.candidate.passageId ?? '');
  });
  return eligible[0]?.index ?? null;
}

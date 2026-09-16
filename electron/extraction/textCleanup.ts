export interface ExtractionControlDiagnostics {
  replacements: number;
  preexistingReplacementCharacters: number;
}

const DISALLOWED_CONTROL = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g;
const REPLACEMENT_CHARACTER = /\uFFFD/g;

export function replaceDisallowedControls(value: string): { text: string; diagnostics: ExtractionControlDiagnostics } {
  const replacements = (value.match(DISALLOWED_CONTROL) ?? []).length;
  const preexistingReplacementCharacters = (value.match(REPLACEMENT_CHARACTER) ?? []).length;
  return {
    text: replacements ? value.replace(DISALLOWED_CONTROL, '\uFFFD') : value,
    diagnostics: { replacements, preexistingReplacementCharacters },
  };
}

export function mergeControlDiagnostics(
  ...values: Array<ExtractionControlDiagnostics | null | undefined>
): ExtractionControlDiagnostics {
  return values.reduce<ExtractionControlDiagnostics>((total, value) => ({
    replacements: total.replacements + (value?.replacements ?? 0),
    preexistingReplacementCharacters: total.preexistingReplacementCharacters
      + (value?.preexistingReplacementCharacters ?? 0),
  }), { replacements: 0, preexistingReplacementCharacters: 0 });
}

/** Shared, deterministic cleanup for extracted prose. Keep source/page markers
 * outside this function so provenance tokens can never be rewritten. */
export function cleanInlineText(value: string): string {
  return replaceDisallowedControls(value).text
    .normalize('NFC')
    .replace(/\u00ad/g, '')
    .replace(/\u00a0/g, ' ')
    .replace(/[\t ]+/g, ' ')
    .replace(/[-‐‑‒–—]{2,}/g, '-')
    // NO LEXICAL REPAIR HERE, DELIBERATELY. Three rules used to guess at OCR damage —
    // rejoining a standalone `fi`/`fl`, and gluing an isolated accented vowel to its
    // neighbours. Measured over the raw text of a real 368-page Spanish book: 21 firings,
    // every single one of them corruption and not one repair. `nació y` became `nacióy`
    // (19 times, the rule never met a genuine split), `empezó á depender` became one
    // word, and `Wi fi network` became two of them.
    .replace(/\s+([,.;:!?%)\]}»”])/g, '$1')
    .replace(/([¿¡([{«“])\s+/g, '$1')
    .trim();
}

export function dehyphenatingJoin(left: string, right: string): string {
  const first = left.trimEnd();
  const second = right.trimStart();
  if (!first) return cleanInlineText(second);
  if (!second) return cleanInlineText(first);
  if (/\d-$/u.test(first) && /^\d/u.test(second)) return cleanInlineText(`${first}${second}`);
  if (/\p{L}{2,}-$/u.test(first) && /^\p{Ll}/u.test(second)) return cleanInlineText(`${first.slice(0, -1)}${second}`);
  return cleanInlineText(`${first} ${second}`);
}

export function cleanExtractedTextWithDiagnostics(value: string): {
  text: string;
  diagnostics: ExtractionControlDiagnostics;
} {
  const sanitized = replaceDisallowedControls(value);
  return {
    text: sanitized.text
      .replace(/\r\n?/g, '\n')
      .split(/\n\s*\n+/)
      .map((paragraph) => paragraph.split('\n').reduce(dehyphenatingJoin, ''))
      .map(cleanInlineText)
      .filter(Boolean)
      .join('\n\n'),
    diagnostics: sanitized.diagnostics,
  };
}

export function cleanExtractedText(value: string): string {
  return cleanExtractedTextWithDiagnostics(value).text;
}

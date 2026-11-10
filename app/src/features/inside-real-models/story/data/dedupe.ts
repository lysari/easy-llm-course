export const SHINGLE_SIZE = 8;
export const NEAR_COPY_SHARE = 0.5;

const FNV_OFFSET = 0x811c9dc5;
const FNV_PRIME = 0x01000193;

export type CopyVerdict = "copy" | "near-copy" | "new";

export interface CopyCheck {
  verdict: CopyVerdict;
  fingerprint: string;
  match: number | null;
  share: number;
  shingleCount: number;
}

export const normalWords = (text: string) => text.toLowerCase().match(/[\p{L}\p{M}\p{N}]+/gu) ?? [];

export function fingerprint(text: string) {
  let hash = FNV_OFFSET;
  for (const byte of new TextEncoder().encode(normalWords(text).join(" "))) hash = Math.imul(hash ^ byte, FNV_PRIME) >>> 0;
  return hash.toString(16).padStart(8, "0");
}

export function shinglesOf(text: string) {
  const words = normalWords(text);
  return new Set(words.slice(0, Math.max(0, words.length - SHINGLE_SIZE + 1)).map((_, i) => words.slice(i, i + SHINGLE_SIZE).join(" ")));
}

function sharedShare(candidate: Set<string>, kept: Set<string>) {
  if (candidate.size === 0) return 0;
  return [...candidate].filter((shingle) => kept.has(shingle)).length / candidate.size;
}

export function checkCopy(candidate: string, pile: string[]): CopyCheck {
  const print = fingerprint(candidate);
  const shingles = shinglesOf(candidate);
  const exact = pile.findIndex((kept) => fingerprint(kept) === print);
  if (exact >= 0) return { verdict: "copy", fingerprint: print, match: exact, share: 1, shingleCount: shingles.size };
  const shares = pile.map((kept) => sharedShare(shingles, shinglesOf(kept)));
  const best = shares.reduce((top, share, i) => (share > shares[top] ? i : top), 0);
  const share = shares[best] ?? 0;
  return { verdict: share >= NEAR_COPY_SHARE ? "near-copy" : "new", fingerprint: print, match: pile.length > 0 ? best : null, share, shingleCount: shingles.size };
}

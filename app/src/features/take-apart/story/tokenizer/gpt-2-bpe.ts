const GPT_2_SPLIT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export const END_OF_TEXT = "<|endoftext|>";

export const MAX_TOKENIZE_LENGTH = 400;

const PRINTABLE_BYTES = [...range(0x21, 0x7e), ...range(0xa1, 0xac), ...range(0xae, 0xff)];

function range(from: number, to: number) {
  return Array.from({ length: to - from + 1 }, (_, i) => from + i);
}

const BYTE_ORDER = [...PRINTABLE_BYTES, ...range(0, 255).filter((byte) => !PRINTABLE_BYTES.includes(byte))];

const BYTE_CHARS: string[] = [];
BYTE_ORDER.forEach((byte, i) => {
  BYTE_CHARS[byte] = String.fromCodePoint(i < PRINTABLE_BYTES.length ? byte : 256 + i - PRINTABLE_BYTES.length);
});

export const BASE_PIECES = BYTE_ORDER.map((byte) => BYTE_CHARS[byte]);

export type Pair = [string, string];

export interface Gpt2Vocab {
  merges: Pair[];
  ranks: Map<string, number>;
  ids: Map<string, number>;
  pieces: string[];
}

export interface MergeStep {
  pieces: string[];
  rule: Pair | null;
  rank: number | null;
}

export interface Token {
  piece: string;
  id: number;
}

export interface EncodedWord {
  text: string;
  steps: MergeStep[];
  tokens: Token[];
}

export const pairKey = ([a, b]: Pair) => `${a} ${b}`;

export const toBytePieces = (text: string) => [...new TextEncoder().encode(text)].map((byte) => BYTE_CHARS[byte]);

export const splitWords = (text: string) => text.match(GPT_2_SPLIT) ?? [];

export function readVocab(bpeText: string): Gpt2Vocab {
  const merges = bpeText
    .split("\n")
    .slice(1)
    .filter(Boolean)
    .map((line): Pair => {
      const [a, b] = line.split(" ");
      return [a, b];
    });
  const pieces = [...BASE_PIECES, ...merges.map(([a, b]) => a + b), END_OF_TEXT];
  return {
    merges,
    ranks: new Map(merges.map((pair, rank) => [pairKey(pair), rank])),
    ids: new Map(pieces.map((piece, id) => [piece, id])),
    pieces,
  };
}

export function mergePair(pieces: string[], [a, b]: Pair): string[] {
  const out: string[] = [];
  for (let i = 0; i < pieces.length; i++) {
    if (pieces[i] === a && pieces[i + 1] === b) {
      out.push(a + b);
      i++;
    } else out.push(pieces[i]);
  }
  return out;
}

function lowestRule(pieces: string[], ranks: Map<string, number>) {
  let best: { rule: Pair; rank: number } | null = null;
  for (let i = 0; i < pieces.length - 1; i++) {
    const rule: Pair = [pieces[i], pieces[i + 1]];
    const rank = ranks.get(pairKey(rule));
    if (rank !== undefined && (!best || rank < best.rank)) best = { rule, rank };
  }
  return best;
}

export function mergeSteps(word: string, vocab: Gpt2Vocab): MergeStep[] {
  let pieces = toBytePieces(word);
  const steps: MergeStep[] = [{ pieces, rule: null, rank: null }];
  for (let best = lowestRule(pieces, vocab.ranks); best; best = lowestRule(pieces, vocab.ranks)) {
    pieces = mergePair(pieces, best.rule);
    steps.push({ pieces, rule: best.rule, rank: best.rank });
  }
  return steps;
}

export function encodeText(text: string, vocab: Gpt2Vocab): EncodedWord[] {
  return splitWords(text).map((word) => {
    const steps = mergeSteps(word, vocab);
    const pieces = steps[steps.length - 1].pieces;
    return { text: word, steps, tokens: pieces.map((piece) => ({ piece, id: vocab.ids.get(piece) ?? -1 })) };
  });
}

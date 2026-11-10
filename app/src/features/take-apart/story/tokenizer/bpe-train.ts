import { type Pair, mergePair, pairKey, splitWords, toBytePieces } from "./gpt-2-bpe";

export interface TrainingWord {
  pieces: string[];
  times: number;
}

export interface PairCount {
  pair: Pair;
  count: number;
}

export interface MergeRound {
  rule: Pair;
  count: number;
  pieceCount: number;
}

export interface PairPlace {
  word: number;
  at: number;
}

export function trainingWords(text: string): TrainingWord[] {
  const times = new Map<string, number>();
  for (const word of splitWords(text)) times.set(word, (times.get(word) ?? 0) + 1);
  return [...times].map(([word, seen]) => ({ pieces: toBytePieces(word), times: seen }));
}

export function countPairs(words: TrainingWord[]): PairCount[] {
  const counts = new Map<string, PairCount>();
  for (const { pieces, times } of words) {
    for (let i = 0; i < pieces.length - 1; i++) {
      const pair: Pair = [pieces[i], pieces[i + 1]];
      const key = pairKey(pair);
      const found = counts.get(key);
      if (found) found.count += times;
      else counts.set(key, { pair, count: times });
    }
  }
  return [...counts.values()].sort((a, b) => b.count - a.count);
}

export const pieceCount = (words: TrainingWord[]) => words.reduce((sum, { pieces, times }) => sum + pieces.length * times, 0);

export function learnMerges(start: TrainingWord[], rounds: number) {
  let words = start;
  const learned: MergeRound[] = [];
  for (let i = 0; i < rounds; i++) {
    const top = countPairs(words)[0];
    if (!top) break;
    words = words.map(({ pieces, times }) => ({ pieces: mergePair(pieces, top.pair), times }));
    learned.push({ rule: top.pair, count: top.count, pieceCount: pieceCount(words) });
  }
  return { learned, words };
}

export function pairPlaces(pieceRows: string[][]): PairPlace[] {
  return pieceRows.flatMap((pieces, word) => pieces.slice(1).map((_, at) => ({ word, at })));
}

export function countFirstPairs(pieceRows: string[][], places: PairPlace[]): PairCount[] {
  return countPairs(places.map(({ word, at }) => ({ pieces: pieceRows[word].slice(at, at + 2), times: 1 })));
}

import { VOCABULARY } from "@/core/model/corpus";
import { count } from "@/shared/utils/format";

export interface ChanceRow {
  word: string;
  id: number;
  score: number;
  power: number;
  chance: number;
}

export interface Softmax {
  rows: ChanceRow[];
  total: number;
}

export function softmaxOf(scores: number[]): Softmax {
  const powers = scores.map(Math.exp);
  const total = powers.reduce((sum, power) => sum + power, 0);
  const rows = VOCABULARY.map((word, id) => ({ word, id, score: scores[id], power: powers[id], chance: powers[id] / total }));
  return { rows: rows.sort((a, b) => b.score - a.score), total };
}

export const showPower = (value: number) => (value >= 1000 ? count(Math.round(value)) : value.toFixed(3));

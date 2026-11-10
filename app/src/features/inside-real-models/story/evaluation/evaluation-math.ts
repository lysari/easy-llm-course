import { TRAINING_LINES } from "@/core/model/corpus";
import { nextOptions, writeText } from "@/core/model/generate";
import { seededRandom } from "@/core/model/random";
import { allTables, copyWeights } from "@/core/model/tiny-gpt";
import type { TinyGptWeights } from "@/shared/interfaces/model.interface";

export const SAFETY_TRIES = 20;
export const BIT_CHOICES = [8, 4, 2];
export const TINY_BITS = 64;

export interface TinyQuestion {
  words: string[];
  answer: string;
}

export interface MarkedQuestion extends TinyQuestion {
  top: string;
  chance: number;
  isRight: boolean;
}

export const TINY_QUESTIONS: TinyQuestion[] = [
  ...new Map(
    TRAINING_LINES.map(({ text }) => {
      const words = text.split(" ").slice(0, -1);
      return [words.join(" "), { words: words.slice(0, -1), answer: words[words.length - 1] }] as const;
    }),
  ).values(),
];

export function markQuestion(question: TinyQuestion, weights: TinyGptWeights): MarkedQuestion {
  const options = nextOptions(question.words, weights);
  return { ...question, top: options[0].token, chance: options.find((option) => option.token === question.answer)?.chance ?? 0, isRight: options[0].token === question.answer };
}

export const scoreOf = (weights: TinyGptWeights) => TINY_QUESTIONS.filter((question) => markQuestion(question, weights).isRight).length / TINY_QUESTIONS.length;

export const writeOnce = (prompt: string[], weights: TinyGptWeights, seed: number) => writeText(prompt, weights, seededRandom(seed)).slice(prompt.length);

export const breaksRule = (written: string[], banned: string) => written.includes(banned);

export function redTeam(prompts: string[][], banned: string, weights: TinyGptWeights, tries = SAFETY_TRIES) {
  return prompts.map((prompt) => ({ prompt, fails: Array.from({ length: tries }, (_, seed) => breaksRule(writeOnce(prompt, weights, seed + 1), banned)).filter(Boolean).length }));
}

export const levelsFor = (bits: number) => 2 ** (bits - 1) - 1;

export const scaleFor = (values: Float64Array, bits: number) => (values.reduce((top, value) => Math.max(top, Math.abs(value)), 0) || 1) / levelsFor(bits);

export function roundNumber(weight: number, scale: number, bits: number) {
  const level = Math.max(-levelsFor(bits), Math.min(levelsFor(bits), Math.round(weight / scale)));
  return { level, back: level * scale, error: Math.abs(level * scale - weight) };
}

export function quantizeWeights(weights: TinyGptWeights, bits: number) {
  const copy = copyWeights(weights);
  let biggestError = 0;
  for (const { tensor } of allTables(copy)) {
    const scale = scaleFor(tensor.data, bits);
    for (let i = 0; i < tensor.data.length; i++) {
      const rounded = roundNumber(tensor.data[i], scale, bits);
      biggestError = Math.max(biggestError, rounded.error);
      tensor.data[i] = rounded.back;
    }
  }
  return { weights: copy, biggestError };
}

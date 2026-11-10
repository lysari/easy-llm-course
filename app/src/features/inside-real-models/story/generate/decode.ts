import { END_TOKEN, MAX_CONTEXT } from "@/core/model/corpus";
import { applySampling, nextOptions, pickToken } from "@/core/model/generate";
import { seededRandom } from "@/core/model/random";
import type { RankedToken, Sampling, TinyGptWeights } from "@/shared/interfaces/model.interface";

export type StopReason = "stop-word" | "your-stop" | "full";

export interface WriteRound {
  words: string[];
  options: RankedToken[];
  roll: number | null;
  picked: RankedToken;
  stop: StopReason | null;
}

export interface RollStep {
  option: RankedToken;
  left: number;
}

export const isGreedy = (sampling: Sampling) => sampling.temperature === 0;

export function chancesFor(words: string[], weights: TinyGptWeights, sampling: Sampling) {
  const options = nextOptions(words, weights);
  return isGreedy(sampling) ? options : applySampling(options, sampling);
}

export function rollWalk(options: RankedToken[], roll: number): RollStep[] {
  const steps: RollStep[] = [];
  let left = roll;
  for (const option of options) {
    left -= option.chance;
    steps.push({ option, left });
    if (left <= 0) break;
  }
  return steps;
}

export function writeRounds(start: string[], weights: TinyGptWeights, sampling: Sampling, seed: number, yourStop: string | null): WriteRound[] {
  const random = seededRandom(seed);
  const words = [...start];
  const rounds: WriteRound[] = [];
  while (words.length < MAX_CONTEXT) {
    const options = chancesFor(words, weights, sampling);
    const roll = isGreedy(sampling) ? null : random();
    const picked = roll === null ? options[0] : pickToken(options, () => roll);
    const stop: StopReason | null = picked.token === END_TOKEN ? "stop-word" : picked.token === yourStop ? "your-stop" : words.length + 1 >= MAX_CONTEXT ? "full" : null;
    rounds.push({ words: [...words], options, roll, picked, stop });
    if (stop === "stop-word" || stop === "your-stop") break;
    words.push(picked.token);
  }
  return rounds;
}

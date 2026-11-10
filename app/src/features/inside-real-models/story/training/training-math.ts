import { EPSILON, LEARNING_RATE, MOMENTUM_DECAY, VARIANCE_DECAY } from "@/core/model/adam";
import { END_TOKEN, TRAINING_SEQUENCES, VOCABULARY } from "@/core/model/corpus";
import { copyWeights, lossAndGradients, measureLoss, traceForward } from "@/core/model/tiny-gpt";
import type { TinyGptWeights } from "@/shared/interfaces/model.interface";

export const NUDGE = 1e-5;

export interface WordChance {
  word: string;
  chance: number;
  loss: number;
}

export interface AdamTrace {
  round: number;
  momentum: number;
  variance: number;
  step: number;
}

export const START_TRACE: AdamTrace = { round: 0, momentum: 0, variance: 0, step: 0 };

export function wordChances(words: string[], weights: TinyGptWeights): WordChance[] {
  const ids = [...words, END_TOKEN].map((word) => VOCABULARY.indexOf(word));
  const { logits } = traceForward(ids.slice(0, -1), weights);
  return logits.map((row, i) => {
    const top = Math.max(...row);
    const total = row.reduce((sum, score) => sum + Math.exp(score - top), 0);
    const chance = Math.exp(row[ids[i + 1]] - top) / total;
    return { word: VOCABULARY[ids[i + 1]], chance, loss: -Math.log(chance) };
  });
}

export function slopeCheck(weights: TinyGptWeights, index = 0) {
  const copy = copyWeights(weights);
  lossAndGradients(TRAINING_SEQUENCES, copy);
  const table = copy.tokenTable.data;
  const weight = table[index];
  table[index] = weight + NUDGE;
  const lossUp = measureLoss(TRAINING_SEQUENCES, copy);
  table[index] = weight - NUDGE;
  const lossDown = measureLoss(TRAINING_SEQUENCES, copy);
  return { weight, slope: copy.tokenTable.grad[index], measured: (lossUp - lossDown) / (2 * NUDGE), lossUp, lossDown };
}

export function adamStep(trace: AdamTrace, gradient: number): AdamTrace {
  const round = trace.round + 1;
  const momentum = MOMENTUM_DECAY * trace.momentum + (1 - MOMENTUM_DECAY) * gradient;
  const variance = VARIANCE_DECAY * trace.variance + (1 - VARIANCE_DECAY) * gradient * gradient;
  const step = (LEARNING_RATE * (momentum / (1 - MOMENTUM_DECAY ** round))) / (Math.sqrt(variance / (1 - VARIANCE_DECAY ** round)) + EPSILON);
  return { round, momentum, variance, step };
}

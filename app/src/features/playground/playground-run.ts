import { Adam, LEARNING_RATE } from "@/core/model/adam";
import { TRAINING_LINES, toSequences } from "@/core/model/corpus";
import { TARGET_ROUNDS } from "@/core/model/lab";
import { allTables, LAYER_COUNT, countWeights, createWeights, lossAndGradients, VECTOR_SIZE } from "@/core/model/tiny-gpt";
import type { PlaygroundSettings } from "@/features/playground/interfaces/playground.interface";
import type { LossPoint, TinyGptWeights } from "@/shared/interfaces/model.interface";

export const TEXT_GROUPS = [
  { name: "hungry dog", match: "my dog is hungry" },
  { name: "hungry cat", match: "my cat is hungry" },
  { name: "bored dog", match: "my dog is bored" },
  { name: "bored cat", match: "my cat is bored" },
  { name: "likes", match: " likes " },
  { name: "chases", match: " chases " },
];

export const defaultSettings = (start: string): PlaygroundSettings => ({
  textGroups: TEXT_GROUPS.map((group) => group.name),
  isEvenAnswers: false,
  vectorSize: VECTOR_SIZE,
  layers: Array.from({ length: LAYER_COUNT }, () => ({ hasAttention: true, hasFeedForward: true })),
  rounds: TARGET_ROUNDS,
  learningRate: LEARNING_RATE,
  optimizer: "adam",
  start,
  temperature: 1,
  topK: 0,
  topP: 1,
});

export function sequencesFor({ textGroups, isEvenAnswers }: { textGroups: string[]; isEvenAnswers: boolean }): number[][] {
  const matches = TEXT_GROUPS.filter((group) => textGroups.includes(group.name)).map((group) => group.match);
  const lines = TRAINING_LINES.filter((line) => matches.some((match) => line.text.includes(match)));
  return toSequences(isEvenAnswers ? lines.map((line) => ({ ...line, times: 1 })) : lines);
}

export class PlaygroundRun {
  readonly weights: TinyGptWeights;
  readonly numberCount: number;
  readonly history: LossPoint[] = [];
  private readonly sequences: number[][];
  private readonly rounds: number;
  private readonly adam: Adam | null;
  private readonly plainStep: number;

  constructor(settings: PlaygroundSettings) {
    this.weights = createWeights(undefined, settings);
    this.numberCount = countWeights(this.weights);
    this.sequences = sequencesFor(settings);
    this.rounds = settings.rounds;
    this.adam = settings.optimizer === "adam" ? new Adam(this.weights, settings.learningRate) : null;
    this.plainStep = settings.learningRate;
  }

  get latestLoss() {
    return this.history[this.history.length - 1]?.loss ?? null;
  }

  get isBroken() {
    return this.latestLoss !== null && !Number.isFinite(this.latestLoss);
  }

  get isDone() {
    return this.isBroken || this.history.length >= this.rounds;
  }

  trainRound() {
    const loss = lossAndGradients(this.sequences, this.weights);
    this.history.push({ round: this.history.length + 1, loss });
    if (!Number.isFinite(loss)) return;
    if (this.adam) {
      this.adam.update();
      return;
    }
    for (const { tensor } of allTables(this.weights)) {
      for (let i = 0; i < tensor.data.length; i++) tensor.data[i] -= this.plainStep * tensor.grad[i];
    }
  }

  trainFor(milliseconds: number) {
    const deadline = performance.now() + milliseconds;
    do {
      if (this.isDone) return;
      this.trainRound();
    } while (performance.now() < deadline);
  }
}

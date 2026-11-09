import type { LossPoint, TinyGptWeights } from "@/shared/interfaces/model.interface";
import { Adam } from "./adam";
import { TRAINING_SEQUENCES } from "./corpus";
import { copyWeights, createWeights, lossAndGradients } from "./tiny-gpt";

export const TARGET_ROUNDS = 200;

export class ModelLab {
  readonly untrained: TinyGptWeights = createWeights();
  live: TinyGptWeights = copyWeights(this.untrained);
  history: LossPoint[] = [];
  private optimizer = new Adam(this.live);

  get round() {
    return this.history.length;
  }

  get isTrained() {
    return this.round >= TARGET_ROUNDS;
  }

  get latestLoss() {
    return this.history[this.history.length - 1]?.loss ?? null;
  }

  trainRounds(count: number) {
    for (let i = 0; i < count && !this.isTrained; i++) {
      const loss = lossAndGradients(TRAINING_SEQUENCES, this.live);
      this.optimizer.update();
      this.history.push({ round: this.round + 1, loss });
    }
  }

  reset() {
    this.live = copyWeights(this.untrained);
    this.optimizer = new Adam(this.live);
    this.history = [];
  }
}

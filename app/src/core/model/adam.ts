import type { TinyGptWeights } from "@/shared/interfaces/model.interface";
import { allTables } from "./tiny-gpt";

export const LEARNING_RATE = 0.004;
export const MOMENTUM_DECAY = 0.9;
export const VARIANCE_DECAY = 0.999;
const EPSILON = 1e-8;

export class Adam {
  private round = 0;
  private readonly momentum: Float64Array[];
  private readonly variance: Float64Array[];

  constructor(
    private readonly weights: TinyGptWeights,
    private readonly learningRate = LEARNING_RATE,
  ) {
    const tables = allTables(weights);
    this.momentum = tables.map(({ tensor }) => new Float64Array(tensor.data.length));
    this.variance = tables.map(({ tensor }) => new Float64Array(tensor.data.length));
  }

  update() {
    this.round++;
    const momentumFix = 1 - MOMENTUM_DECAY ** this.round;
    const varianceFix = 1 - VARIANCE_DECAY ** this.round;
    allTables(this.weights).forEach(({ tensor }, t) => {
      const momentum = this.momentum[t];
      const variance = this.variance[t];
      for (let i = 0; i < tensor.data.length; i++) {
        const gradient = tensor.grad[i];
        momentum[i] = MOMENTUM_DECAY * momentum[i] + (1 - MOMENTUM_DECAY) * gradient;
        variance[i] = VARIANCE_DECAY * variance[i] + (1 - VARIANCE_DECAY) * gradient * gradient;
        tensor.data[i] -= (this.learningRate * (momentum[i] / momentumFix)) / (Math.sqrt(variance[i] / varianceFix) + EPSILON);
      }
    });
  }
}

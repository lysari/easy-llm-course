import { SALARIES, YEARS } from "./line-model";
import { seededRandom } from "./random";

export const LESSON_ROUNDS = 50000;
export const NETWORK_TOTAL_ROUNDS = 500000;
export const NETWORK_NUMBER_COUNT = 10;
const LEARNING_RATE = 0.05;
const DEFAULT_SEED = 1;
const TOP_SALARY = Math.max(...SALARIES);

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const slopeOf = (s: number) => s * (1 - s);

export class SmallNetwork {
  rounds = 0;
  private readonly w: number[];
  private readonly b = [0, 0, 0];
  private readonly wOut: number[];
  private bOut = 0;

  constructor(seed = DEFAULT_SEED) {
    const random = seededRandom(seed);
    this.w = [random() - 0.5, random() - 0.5, random() - 0.5];
    this.wOut = [random() - 0.5, random() - 0.5, random() - 0.5];
  }

  private forward(x: number) {
    const hidden = this.w.map((w, i) => sigmoid(w * x + this.b[i]));
    const out = sigmoid(hidden.reduce((sum, h, i) => sum + this.wOut[i] * h, this.bOut));
    return { hidden, out };
  }

  explain(index: number) {
    const x = YEARS[index];
    const { hidden, out } = this.forward(x);
    const target = SALARIES[index] / TOP_SALARY;
    const error = target - out;
    const deltaOut = error * slopeOf(out);
    return {
      x,
      hidden,
      out,
      target,
      error,
      deltaOut,
      outputChange: LEARNING_RATE * deltaOut * hidden[0],
      deltaHidden: deltaOut * this.wOut[0] * slopeOf(hidden[0]),
    };
  }

  predict(x: number) {
    return this.forward(x).out * TOP_SALARY;
  }

  loss() {
    return YEARS.reduce((sum, x, i) => sum + (this.predict(x) - SALARIES[i]) ** 2, 0) / YEARS.length;
  }

  learnFrom(index: number) {
    const x = YEARS[index];
    const { hidden, out } = this.forward(x);
    const outputBlame = (SALARIES[index] / TOP_SALARY - out) * slopeOf(out);
    this.wOut.forEach((_, i) => {
      this.wOut[i] += LEARNING_RATE * outputBlame * hidden[i];
    });
    this.bOut += LEARNING_RATE * outputBlame;
    hidden.forEach((h, i) => {
      const hiddenBlame = outputBlame * this.wOut[i] * slopeOf(h);
      this.w[i] += LEARNING_RATE * hiddenBlame * x;
      this.b[i] += LEARNING_RATE * hiddenBlame;
    });
  }

  trainRounds(count: number) {
    for (let round = 0; round < count; round++) {
      for (let i = 0; i < YEARS.length; i++) this.learnFrom(i);
      this.rounds++;
    }
  }
}

export class Tensor {
  grad: Float64Array;
  backward: (() => void) | null = null;

  constructor(
    readonly rows: number,
    readonly cols: number,
    readonly data: Float64Array,
    readonly tape: Tensor[] | null,
  ) {
    this.grad = new Float64Array(data.length);
    tape?.push(this);
  }

  row(index: number): number[] {
    return Array.from(this.data.subarray(index * this.cols, (index + 1) * this.cols));
  }

  toRows(): number[][] {
    return Array.from({ length: this.rows }, (_, r) => this.row(r));
  }
}

export function parameter(rows: number, cols: number, data: Float64Array): Tensor {
  return new Tensor(rows, cols, data, null);
}

const resultOf = (rows: number, cols: number, tape: Tensor[]) => new Tensor(rows, cols, new Float64Array(rows * cols), tape);

export function lookupRows(table: Tensor, indexes: number[], tape: Tensor[]): Tensor {
  const out = resultOf(indexes.length, table.cols, tape);
  indexes.forEach((index, r) => {
    out.data.set(table.data.subarray(index * table.cols, (index + 1) * table.cols), r * table.cols);
  });
  out.backward = () => {
    indexes.forEach((index, r) => {
      for (let c = 0; c < table.cols; c++) table.grad[index * table.cols + c] += out.grad[r * table.cols + c];
    });
  };
  return out;
}

export function add(a: Tensor, b: Tensor, tape: Tensor[]): Tensor {
  const out = resultOf(a.rows, a.cols, tape);
  for (let i = 0; i < out.data.length; i++) out.data[i] = a.data[i] + b.data[i];
  out.backward = () => {
    for (let i = 0; i < out.data.length; i++) {
      a.grad[i] += out.grad[i];
      b.grad[i] += out.grad[i];
    }
  };
  return out;
}

export function multiply(a: Tensor, b: Tensor, tape: Tensor[]): Tensor {
  const out = resultOf(a.rows, b.cols, tape);
  for (let r = 0; r < a.rows; r++) {
    for (let k = 0; k < a.cols; k++) {
      const value = a.data[r * a.cols + k];
      for (let c = 0; c < b.cols; c++) out.data[r * b.cols + c] += value * b.data[k * b.cols + c];
    }
  }
  out.backward = () => {
    for (let r = 0; r < a.rows; r++) {
      for (let k = 0; k < a.cols; k++) {
        for (let c = 0; c < b.cols; c++) {
          const g = out.grad[r * b.cols + c];
          a.grad[r * a.cols + k] += g * b.data[k * b.cols + c];
          b.grad[k * b.cols + c] += g * a.data[r * a.cols + k];
        }
      }
    }
  };
  return out;
}

export function matchScores(queries: Tensor, keys: Tensor, scale: number, tape: Tensor[]): Tensor {
  const out = resultOf(queries.rows, keys.rows, tape);
  const width = queries.cols;
  for (let r = 0; r < queries.rows; r++) {
    for (let c = 0; c < keys.rows; c++) {
      let sum = 0;
      for (let k = 0; k < width; k++) sum += queries.data[r * width + k] * keys.data[c * width + k];
      out.data[r * keys.rows + c] = sum * scale;
    }
  }
  out.backward = () => {
    for (let r = 0; r < queries.rows; r++) {
      for (let c = 0; c < keys.rows; c++) {
        const g = out.grad[r * keys.rows + c] * scale;
        for (let k = 0; k < width; k++) {
          queries.grad[r * width + k] += g * keys.data[c * width + k];
          keys.grad[c * width + k] += g * queries.data[r * width + k];
        }
      }
    }
  };
  return out;
}

export function causalSoftmax(scores: Tensor, tape: Tensor[]): Tensor {
  const out = resultOf(scores.rows, scores.cols, tape);
  for (let r = 0; r < scores.rows; r++) {
    let top = -Infinity;
    for (let c = 0; c <= r; c++) top = Math.max(top, scores.data[r * scores.cols + c]);
    let total = 0;
    for (let c = 0; c <= r; c++) total += Math.exp(scores.data[r * scores.cols + c] - top);
    for (let c = 0; c <= r; c++) out.data[r * scores.cols + c] = Math.exp(scores.data[r * scores.cols + c] - top) / total;
  }
  out.backward = () => {
    for (let r = 0; r < scores.rows; r++) {
      let mean = 0;
      for (let c = 0; c <= r; c++) mean += out.grad[r * scores.cols + c] * out.data[r * scores.cols + c];
      for (let c = 0; c <= r; c++) {
        scores.grad[r * scores.cols + c] += out.data[r * scores.cols + c] * (out.grad[r * scores.cols + c] - mean);
      }
    }
  };
  return out;
}

export function relu(a: Tensor, tape: Tensor[]): Tensor {
  const out = resultOf(a.rows, a.cols, tape);
  for (let i = 0; i < a.data.length; i++) out.data[i] = Math.max(0, a.data[i]);
  out.backward = () => {
    for (let i = 0; i < a.data.length; i++) if (a.data[i] > 0) a.grad[i] += out.grad[i];
  };
  return out;
}

export function layerNorm(a: Tensor, tape: Tensor[]): Tensor {
  const EPSILON = 1e-5;
  const out = resultOf(a.rows, a.cols, tape);
  const means = new Float64Array(a.rows);
  const spreads = new Float64Array(a.rows);
  for (let r = 0; r < a.rows; r++) {
    let mean = 0;
    for (let c = 0; c < a.cols; c++) mean += a.data[r * a.cols + c];
    mean /= a.cols;
    let variance = 0;
    for (let c = 0; c < a.cols; c++) variance += (a.data[r * a.cols + c] - mean) ** 2;
    const spread = Math.sqrt(variance / a.cols + EPSILON);
    means[r] = mean;
    spreads[r] = spread;
    for (let c = 0; c < a.cols; c++) out.data[r * a.cols + c] = (a.data[r * a.cols + c] - mean) / spread;
  }
  out.backward = () => {
    for (let r = 0; r < a.rows; r++) {
      let gradMean = 0;
      let gradDotCentered = 0;
      for (let c = 0; c < a.cols; c++) {
        gradMean += out.grad[r * a.cols + c];
        gradDotCentered += out.grad[r * a.cols + c] * (a.data[r * a.cols + c] - means[r]);
      }
      gradMean /= a.cols;
      const spreadCubed = spreads[r] ** 3;
      for (let c = 0; c < a.cols; c++) {
        const centered = a.data[r * a.cols + c] - means[r];
        a.grad[r * a.cols + c] += (out.grad[r * a.cols + c] - gradMean) / spreads[r] - (gradDotCentered * centered) / (a.cols * spreadCubed);
      }
    }
  };
  return out;
}

export function crossEntropy(logits: Tensor, targets: number[], tape: Tensor[]): Tensor {
  const out = resultOf(1, 1, tape);
  const probabilities = new Float64Array(logits.data.length);
  let total = 0;
  for (let r = 0; r < logits.rows; r++) {
    let top = -Infinity;
    for (let c = 0; c < logits.cols; c++) top = Math.max(top, logits.data[r * logits.cols + c]);
    let sum = 0;
    for (let c = 0; c < logits.cols; c++) sum += Math.exp(logits.data[r * logits.cols + c] - top);
    for (let c = 0; c < logits.cols; c++) probabilities[r * logits.cols + c] = Math.exp(logits.data[r * logits.cols + c] - top) / sum;
    total -= Math.log(probabilities[r * logits.cols + targets[r]]);
  }
  out.data[0] = total / logits.rows;
  out.backward = () => {
    const g = out.grad[0] / logits.rows;
    for (let r = 0; r < logits.rows; r++) {
      for (let c = 0; c < logits.cols; c++) {
        const isTarget = c === targets[r] ? 1 : 0;
        logits.grad[r * logits.cols + c] += g * (probabilities[r * logits.cols + c] - isTarget);
      }
    }
  };
  return out;
}

export function runBackward(loss: Tensor, tape: Tensor[]) {
  loss.grad[0] = 1;
  for (let i = tape.length - 1; i >= 0; i--) tape[i].backward?.();
}

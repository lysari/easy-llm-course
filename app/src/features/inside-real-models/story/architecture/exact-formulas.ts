const GPT_2_NORM_EPSILON = 1e-5;
const LLAMA_3_NORM_EPSILON = 1e-5;
const LLAMA_3_ROPE_THETA = 500_000;

const GELU_SCALE = Math.sqrt(2 / Math.PI);
const GELU_CUBE = 0.044715;

export const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(GELU_SCALE * (x + GELU_CUBE * x ** 3)));

export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export const silu = (x: number) => x * sigmoid(x);

const average = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;

export function layerNormParts(values: number[], epsilon = GPT_2_NORM_EPSILON) {
  const mean = average(values);
  const spread = Math.sqrt(average(values.map((value) => (value - mean) ** 2)) + epsilon);
  return { mean, spread, out: values.map((value) => (value - mean) / spread) };
}

export function rmsNormParts(values: number[], epsilon = LLAMA_3_NORM_EPSILON) {
  const rms = Math.sqrt(average(values.map((value) => value * value)) + epsilon);
  return { rms, out: values.map((value) => value / rms) };
}

export const ropeAngle = (pair: number, size: number, place: number, theta = LLAMA_3_ROPE_THETA) => place * theta ** (-(2 * pair) / size);

export function rope(values: number[], place: number, theta = LLAMA_3_ROPE_THETA) {
  return values.map((_, i) => {
    const pair = Math.floor(i / 2);
    const angle = ropeAngle(pair, values.length, place, theta);
    const [x, y] = [values[pair * 2], values[pair * 2 + 1]];
    return i % 2 === 0 ? x * Math.cos(angle) - y * Math.sin(angle) : x * Math.sin(angle) + y * Math.cos(angle);
  });
}

export const dot = (a: number[], b: number[]) => a.reduce((sum, value, i) => sum + value * b[i], 0);


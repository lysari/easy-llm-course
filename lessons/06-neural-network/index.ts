// Multi-Layer Perceptron with vector input
// Input: number[] (multiple features), not just one number

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

function relu(z: number): number {
  return Math.max(0, z);
}

// One neuron: weighted sum of inputs + bias → activation
function neuron(inputs: number[], weights: number[], bias: number): number {
  const z = inputs.reduce((sum, xi, i) => sum + xi * (weights[i] ?? 0), bias);
  return sigmoid(z);
}

// One layer: multiple neurons in parallel
function layer(inputs: number[], weights: number[][], biases: number[]): number[] {
  return weights.map((w, i) => neuron(inputs, w, biases[i] ?? 0));
}

// Network: input(2) → hidden(4) → output(1)
// Weights shape: hidden layer = (4 × 2), output layer = (1 × 4)
const hiddenWeights = [
  [0.5, -0.3],
  [-0.2, 0.8],
  [0.1, 0.4],
  [-0.6, 0.2],
];
const hiddenBiases = [0.1, -0.1, 0.05, 0.0];

const outputWeights = [[0.3, -0.4, 0.6, 0.2]];
const outputBiases = [0.0];

function forward(input: number[]): number[] {
  const hidden = layer(input, hiddenWeights, hiddenBiases);
  const output = layer(hidden, outputWeights, outputBiases);
  return output;
}

// Test with different inputs
const inputs = [
  [1.0, 0.5],
  [0.0, 0.0],
  [2.0, -1.0],
  [-0.5, 1.5],
];

console.log("=== Forward Pass ===");
for (const input of inputs) {
  const output = forward(input);
  console.log(`input=${JSON.stringify(input)} → output=${output.map(v => v.toFixed(4))}`);
}

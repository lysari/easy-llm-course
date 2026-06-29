// RL-03 demo: attention is just dot products and weighted sums
//
// Run: npx ts-node reverse-lessons/RL-03-attention-is-arithmetic/demo.ts

function dot(a: number[], b: number[]): number {
  return a.reduce((sum, ai, i) => sum + ai * b[i], 0);
}

function softmax(scores: number[]): number[] {
  const max = Math.max(...scores);
  const exps = scores.map(s => Math.exp(s - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function matMul(vector: number[], matrix: number[][]): number[] {
  return matrix.map(row => dot(vector, row));
}

// Tiny 3-token example: "cat sat mat"
// Each token has a 4-dimensional embedding
const tokens = ["cat", "sat", "mat"];
const embeddings = [
  [ 0.9,  0.1, -0.2,  0.7],   // cat
  [-0.3,  0.8,  0.4, -0.5],   // sat
  [-0.4,  0.6,  0.3, -0.5],   // mat  (similar to "sat" → both short words that rhyme)
];

// Weight matrices (tiny: 4→4). In a real model these are 768×64 or larger.
// These are fixed here to show the mechanism. In training, they're learned.
const W_Q = [
  [0.1, 0.2, 0.3, 0.4],
  [0.5, 0.6, 0.7, 0.8],
  [0.9, 0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6, 0.7],
];
const W_K = [
  [0.7, 0.6, 0.5, 0.4],
  [0.3, 0.2, 0.1, 0.9],
  [0.8, 0.7, 0.6, 0.5],
  [0.4, 0.3, 0.2, 0.1],
];
const W_V = [
  [0.5, 0.5, 0.5, 0.5],
  [0.2, 0.8, 0.2, 0.8],
  [0.9, 0.1, 0.9, 0.1],
  [0.3, 0.7, 0.3, 0.7],
];

// Compute Q, K, V for each token
const queries = embeddings.map(e => matMul(e, W_Q));
const keys    = embeddings.map(e => matMul(e, W_K));
const values  = embeddings.map(e => matMul(e, W_V));

const dim = queries[0]!.length;
const scale = Math.sqrt(dim);

console.log("=== Single-head attention for 'cat sat mat' ===\n");

// Compute attention output for each token
for (let i = 0; i < tokens.length; i++) {
  const token = tokens[i]!;
  const q = queries[i]!;

  // Compute raw scores against all keys
  const rawScores = keys.map(k => dot(q, k) / scale);

  // Convert to attention weights via softmax
  const weights = softmax(rawScores);

  // Weighted sum of values
  const output = new Array(dim).fill(0);
  for (let j = 0; j < tokens.length; j++) {
    for (let d = 0; d < dim; d++) {
      output[d] += weights[j]! * values[j]![d]!;
    }
  }

  console.log(`Token: "${token}"`);
  console.log(`  Query:           [${q.map(x => x.toFixed(2)).join(", ")}]`);
  console.log(`  Raw scores vs all keys: [${rawScores.map(x => x.toFixed(3)).join(", ")}]`);
  console.log(`  Attention weights:      [${weights.map(x => x.toFixed(3)).join(", ")}]`);
  console.log(`  (weights for:           [${tokens.join(",  ")}])`);
  console.log(`  Output vector:   [${output.map(x => x.toFixed(3)).join(", ")}]`);
  console.log();
}

console.log("=== What just happened? ===");
console.log("For each token, we:");
console.log("  1. Multiplied its embedding by W_Q to get a query vector");
console.log("  2. Multiplied all embeddings by W_K to get key vectors");
console.log("  3. Computed dot products (query · key) for all pairs");
console.log("  4. Applied softmax to get attention weights (sum = 1.0)");
console.log("  5. Computed a weighted average of value vectors");
console.log();
console.log("That is ALL attention is.");
console.log("Dot products → softmax → weighted sum.");
console.log("No understanding. No grammar. No semantics.");
console.log("Just arithmetic on vectors.");

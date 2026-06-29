// Matrix math: the foundation of every layer in a neural network
// A layer is just: output = activation(X @ W + b)

function dotProduct(a: number[], b: number[]): number {
  return a.reduce((sum, ai, i) => sum + ai * (b[i] ?? 0), 0);
}

function matmul(A: number[][], B: number[][]): number[][] {
  const rows = A.length;
  const cols = B[0]?.length ?? 0;
  const inner = B.length;
  const C = Array.from({ length: rows }, () => Array<number>(cols).fill(0));
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++)
      for (let k = 0; k < inner; k++)
        C[i]![j]! += (A[i]?.[k] ?? 0) * (B[k]?.[j] ?? 0);
  return C;
}

function transpose(A: number[][]): number[][] {
  return (A[0] ?? []).map((_, j) => A.map(row => row[j] ?? 0));
}

function addBias(A: number[][], bias: number[]): number[][] {
  return A.map(row => row.map((v, j) => v + (bias[j] ?? 0)));
}

function applyRelu(A: number[][]): number[][] {
  return A.map(row => row.map(v => Math.max(0, v)));
}

function applySigmoid(A: number[][]): number[][] {
  return A.map(row => row.map(v => 1 / (1 + Math.exp(-v))));
}

// ── Test dot product ──
console.log("=== Dot product ===");
console.log("[1,2,3] · [4,5,6] =", dotProduct([1, 2, 3], [4, 5, 6])); // 32

// ── Test matmul ──
console.log("\n=== Matrix multiply (2×3) @ (3×2) = (2×2) ===");
const A = [[1, 2, 3], [4, 5, 6]];
const B = [[7, 8], [9, 10], [11, 12]];
const C = matmul(A, B);
console.log(C); // [[58, 64], [139, 154]]

// ── Test transpose ──
console.log("\n=== Transpose ===");
console.log(transpose([[1, 2, 3], [4, 5, 6]])); // [[1,4],[2,5],[3,6]]

// ── Forward pass of one dense layer ──
console.log("\n=== Dense layer: X(3×2) @ W(2×4) + b → relu ===");
const X = [[0.5, 1.0], [1.5, 2.0], [2.5, 0.5]]; // 3 samples, 2 features
const W = [[0.2, -0.3, 0.5, 0.1], [-0.1, 0.4, -0.2, 0.3]]; // 2 inputs, 4 outputs
const biasVec = [0.1, 0.1, 0.1, 0.1];

const Z = addBias(matmul(X, W), biasVec);
const H = applyRelu(Z);

console.log("Output shape:", H.length, "×", H[0]?.length);
H.forEach((row, i) => console.log(`  sample ${i}: [${row.map(v => v.toFixed(3)).join(", ")}]`));

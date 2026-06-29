export {}; // isolate this file's scope from other lessons

// MLP Sorter — learn to sort [a, b, c] → [min, mid, max]
// Architecture: Input(3) → Hidden(16, relu) → Hidden(8, relu) → Output(3, linear)
// This is regression, not classification: the network predicts actual values, not classes

// ── Matrix helpers (same as lesson 08) ──
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

function addBias(Z: number[][], b: number[]): number[][] {
  return Z.map(row => row.map((v, j) => v + (b[j] ?? 0)));
}

// Element-wise multiply — used in backprop to cancel out where relu was 0
function hadamard(A: number[][], B: number[][]): number[][] {
  return A.map((row, i) => row.map((v, j) => v * (B[i]?.[j] ?? 0)));
}

// Derivative of relu: 1 if the neuron fired, 0 if it didn't
function reluMask(Z: number[][]): number[][] {
  return Z.map(row => row.map(v => (v > 0 ? 1 : 0)));
}

// ── Weight initialization (He init — designed for relu networks) ──
function initWeights(rows: number, cols: number): number[][] {
  const scale = Math.sqrt(2 / rows);
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (Math.random() - 0.5) * scale)
  );
}

// W shape: [inputs × outputs] so we can do matmul(X, W) where X is [batch × inputs]
let W1 = initWeights(3, 16);
let b1 = Array<number>(16).fill(0);
let W2 = initWeights(16, 8);
let b2 = Array<number>(8).fill(0);
let W3 = initWeights(8, 3);
let b3 = Array<number>(3).fill(0);

// ── Forward pass ──
function forward(X: number[][]) {
  const Z1 = addBias(matmul(X, W1), b1);                  // [batch × 16]
  const A1 = Z1.map(row => row.map(v => Math.max(0, v))); // relu
  const Z2 = addBias(matmul(A1, W2), b2);                 // [batch × 8]
  const A2 = Z2.map(row => row.map(v => Math.max(0, v))); // relu
  const Z3 = addBias(matmul(A2, W3), b3);                 // [batch × 3] — no activation (linear)
  return { Z1, A1, Z2, A2, Z3 };
}

// ── MSE loss — how far off are the predictions? ──
function mseLoss(pred: number[][], target: number[][]): number {
  let sum = 0, count = 0;
  pred.forEach((row, i) => row.forEach((v, j) => {
    sum += (v - (target[i]?.[j] ?? 0)) ** 2;
    count++;
  }));
  return sum / count;
}

// ── Backward pass ──
// Chain rule, starting from the output and working back to the input
function backward(
  X: number[][],
  target: number[][],
  fwd: ReturnType<typeof forward>,
  lr: number
): void {
  const { Z1, A1, Z2, A2, Z3 } = fwd;
  const n = X.length;

  // dL/dZ3 — output has no activation, so gradient is just the MSE derivative
  const dZ3 = Z3.map((row, i) =>
    row.map((v, j) => (2 / n) * (v - (target[i]?.[j] ?? 0)))
  );

  // W3 gradient = A2ᵀ @ dZ3, bias gradient = sum over batch
  const dW3 = matmul(transpose(A2), dZ3);
  const db3new = (dZ3[0] ?? []).map((_, j) =>
    dZ3.reduce((s, row) => s + (row[j] ?? 0), 0)
  );

  // Backprop through relu in layer 2
  const dA2 = matmul(dZ3, transpose(W3));
  const dZ2 = hadamard(dA2, reluMask(Z2));

  const dW2 = matmul(transpose(A1), dZ2);
  const db2new = (dZ2[0] ?? []).map((_, j) =>
    dZ2.reduce((s, row) => s + (row[j] ?? 0), 0)
  );

  // Backprop through relu in layer 1
  const dA1 = matmul(dZ2, transpose(W2));
  const dZ1 = hadamard(dA1, reluMask(Z1));

  const dW1 = matmul(transpose(X), dZ1);
  const db1new = (dZ1[0] ?? []).map((_, j) =>
    dZ1.reduce((s, row) => s + (row[j] ?? 0), 0)
  );

  // SGD: move weights in the opposite direction of the gradient
  W3 = W3.map((row, i) => row.map((w, j) => w - lr * (dW3[i]?.[j] ?? 0)));
  b3 = b3.map((v, j) => v - lr * (db3new[j] ?? 0));
  W2 = W2.map((row, i) => row.map((w, j) => w - lr * (dW2[i]?.[j] ?? 0)));
  b2 = b2.map((v, j) => v - lr * (db2new[j] ?? 0));
  W1 = W1.map((row, i) => row.map((w, j) => w - lr * (dW1[i]?.[j] ?? 0)));
  b1 = b1.map((v, j) => v - lr * (db1new[j] ?? 0));
}

// ── Training data ──
// Numbers are normalized to 0–1 range so the network doesn't have to deal with scale
function generateBatch(n: number): { X: number[][]; Y: number[][] } {
  const X: number[][] = [], Y: number[][] = [];
  for (let i = 0; i < n; i++) {
    const raw = [Math.random() * 9 + 1, Math.random() * 9 + 1, Math.random() * 9 + 1];
    X.push(raw.map(v => v / 10));
    Y.push([...raw].sort((a, b) => a - b).map(v => v / 10));
  }
  return { X, Y };
}

// ── Training loop ──
const LR = 0.01;
const EPOCHS = 3000;
const BATCH_SIZE = 64;

console.log("=== Training MLP Sorter ===");
for (let epoch = 0; epoch <= EPOCHS; epoch++) {
  const { X, Y } = generateBatch(BATCH_SIZE);
  const fwd = forward(X);
  const loss = mseLoss(fwd.Z3, Y);
  backward(X, Y, fwd, LR);

  if (epoch % 500 === 0)
    console.log(`Epoch ${String(epoch).padStart(4)} — Loss: ${loss.toFixed(6)}`);
}

// ── Test ──
console.log("\n=== Test ===");
const tests = [[10,3, 10.1, 1, 2], [9, 5, 7], [2, 8, 4], [6, 6, 1], [4, 4, 4]];

for (const raw of tests) {
  const fwd = forward([raw.map(v => v / 10)]);
  const predicted = (fwd.Z3[0] ?? []).map(v => +(v * 10).toFixed(1));
  const expected = [...raw].sort((a, b) => a - b);
  console.log(`[${raw}] → predicted: [${predicted}] | expected: [${expected}]`);
}

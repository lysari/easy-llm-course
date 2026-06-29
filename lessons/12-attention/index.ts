// Scaled dot-product attention: the core operation of every transformer / LLM

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function matmul(A: number[][], B: number[][]): number[][] {
  const rows = A.length, cols = B[0]?.length ?? 0, inner = B.length;
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

// Learned projection matrices (normally trained via backprop)
function linearProject(X: number[][], W: number[][]): number[][] {
  return matmul(X, W);
}

function attention(
  Q: number[][], // (T × dk) — what each token is looking for
  K: number[][], // (T × dk) — what each token advertises
  V: number[][], // (T × dv) — what each token contains
  causal = false // true = each token only sees past tokens (LM mode)
): { output: number[][]; weights: number[][] } {
  const T = Q.length;
  const dk = Q[0]?.length ?? 1;

  // Score: how relevant is each key to each query?
  const scores = matmul(Q, transpose(K)).map(row => row.map(s => s / Math.sqrt(dk)));

  // Causal mask: hide future tokens
  if (causal) {
    for (let i = 0; i < T; i++)
      for (let j = i + 1; j < T; j++)
        scores[i]![j] = -Infinity;
  }

  // Turn scores into probabilities
  const weights = scores.map(row => softmax(row));

  // Weighted sum of values
  const output = matmul(weights, V);
  return { output, weights };
}

// ── Demo: 5 tokens, embed dim = 4 ──
const T = 5, d = 4;
// Fake token embeddings (normally come from embedding table)
const X: number[][] = [
  [1.0, 0.5, -0.2, 0.8],
  [0.3, 1.2, 0.4, -0.5],
  [-0.1, 0.7, 1.0, 0.2],
  [0.6, -0.3, 0.5, 1.1],
  [0.2, 0.9, -0.4, 0.6],
];

// Fake projection weights (in a real model these are learned)
const randW = (r: number, c: number) =>
  Array.from({ length: r }, () => Array.from({ length: c }, () => (Math.random() - 0.5) * 0.5));

const Wq = randW(d, d), Wk = randW(d, d), Wv = randW(d, d);
const Q = linearProject(X, Wq);
const K = linearProject(X, Wk);
const V = linearProject(X, Wv);

const { output, weights } = attention(Q, K, V, true);

console.log(`=== Attention (${T} tokens, d=${d}, causal=true) ===`);
console.log("Output shape:", output.length, "×", output[0]?.length);

console.log("\nAttention weights (each row sums to 1.0):");
weights.forEach((row, i) => {
  console.log(`  token ${i}: [${row.map(w => w.toFixed(3)).join(", ")}]`);
});

// Token 0 should attend only to itself (causal mask blocks future)
console.log("\nToken 0 attends only to token 0:", weights[0]?.[0]?.toFixed(3), "(should be ~1.000)");

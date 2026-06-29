// One transformer block: the repeating unit inside every LLM
// Block = LayerNorm → Attention → residual → LayerNorm → FFN → residual

// ── Helpers ──
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

// ── Layer Normalization (per token, across embedding dim) ──
function layerNorm(x: number[], gamma: number[], beta: number[], eps = 1e-5): number[] {
  const mean = x.reduce((s, v) => s + v, 0) / x.length;
  const variance = x.reduce((s, v) => s + (v - mean) ** 2, 0) / x.length;
  return x.map((v, i) => ((v - mean) / Math.sqrt(variance + eps)) * (gamma[i] ?? 1) + (beta[i] ?? 0));
}

// ── Residual add ──
function addResidual(x: number[][], delta: number[][]): number[][] {
  return x.map((row, i) => row.map((v, j) => v + (delta[i]?.[j] ?? 0)));
}

// ── Scaled dot-product attention ──
function attention(Q: number[][], K: number[][], V: number[][], causal: boolean): number[][] {
  const dk = Q[0]?.length ?? 1;
  const scores = matmul(Q, transpose(K)).map(row => row.map(s => s / Math.sqrt(dk)));
  if (causal) {
    for (let i = 0; i < scores.length; i++)
      for (let j = i + 1; j < scores.length; j++)
        scores[i]![j] = -Infinity;
  }
  return matmul(scores.map(row => softmax(row)), V);
}

// ── Feed-Forward Network (applied per token) ──
function relu(z: number): number { return Math.max(0, z); }

function ffn(x: number[], W1: number[][], b1: number[], W2: number[][], b2: number[]): number[] {
  // x: (d,) → hidden: (4d,) → out: (d,)
  const hidden = W1.map((row, i) =>
    relu(x.reduce((s, v, j) => s + v * (row[j] ?? 0), 0) + (b1[i] ?? 0))
  );
  return W2.map((row, i) =>
    hidden.reduce((s, v, j) => s + v * (row[j] ?? 0), 0) + (b2[i] ?? 0)
  );
}

// ── Transformer block ──
interface BlockParams {
  Wq: number[][], Wk: number[][], Wv: number[][], Wo: number[][];
  ln1_gamma: number[], ln1_beta: number[];
  ln2_gamma: number[], ln2_beta: number[];
  ffnW1: number[][], ffnB1: number[], ffnW2: number[][], ffnB2: number[];
}

function transformerBlock(X: number[][], p: BlockParams, causal = true): number[][] {
  const d = X[0]?.length ?? 1;

  // Sublayer 1: LayerNorm → Self-Attention → residual
  const normed1 = X.map(row => layerNorm(row, p.ln1_gamma, p.ln1_beta));
  const Q = matmul(normed1, p.Wq);
  const K = matmul(normed1, p.Wk);
  const V = matmul(normed1, p.Wv);
  const attnOut = matmul(attention(Q, K, V, causal), p.Wo);
  const X2 = addResidual(X, attnOut);

  // Sublayer 2: LayerNorm → FFN → residual
  const normed2 = X2.map(row => layerNorm(row, p.ln2_gamma, p.ln2_beta));
  const ffnOut = normed2.map(row => ffn(row, p.ffnW1, p.ffnB1, p.ffnW2, p.ffnB2));
  return addResidual(X2, ffnOut);
}

// ── Test with random weights ──
const T = 4, d = 8, dff = 32;
const rand = (r: number, c: number) =>
  Array.from({ length: r }, () => Array.from({ length: c }, () => (Math.random() - 0.5) * 0.1));

const X: number[][] = Array.from({ length: T }, () =>
  Array.from({ length: d }, () => Math.random() - 0.5)
);

const params: BlockParams = {
  Wq: rand(d, d), Wk: rand(d, d), Wv: rand(d, d), Wo: rand(d, d),
  ln1_gamma: Array(d).fill(1), ln1_beta: Array(d).fill(0),
  ln2_gamma: Array(d).fill(1), ln2_beta: Array(d).fill(0),
  ffnW1: rand(dff, d), ffnB1: Array(dff).fill(0),
  ffnW2: rand(d, dff), ffnB2: Array(d).fill(0),
};

const output = transformerBlock(X, params);

console.log(`=== Transformer Block ===`);
console.log(`Input shape:  ${T} × ${d}`);
console.log(`Output shape: ${output.length} × ${output[0]?.length}`);
console.log("Input  token 0:", X[0]?.map(v => v.toFixed(3)));
console.log("Output token 0:", output[0]?.map(v => v.toFixed(3)));
console.log("\nShape is preserved — block transforms without changing dimensions.");

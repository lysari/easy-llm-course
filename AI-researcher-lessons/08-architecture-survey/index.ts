// Why transformers won, measured: a real tiny-RNN forward pass, then the
// vanishing-gradient effect — multiply the RNN's step Jacobians across 50
// timesteps and watch the gradient signal die — vs attention's 1-step path.
//
// Run: npx ts-node index.ts

// ── Seeded PRNG (mulberry32): same numbers every run ──
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Small matrix helpers (number[][] everywhere, guarded accesses) ──
function matmul(A: number[][], B: number[][]): number[][] {
  const rows = A.length, cols = B[0]?.length ?? 0, inner = B.length;
  const C = Array.from({ length: rows }, () => Array<number>(cols).fill(0));
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++)
      for (let k = 0; k < inner; k++)
        C[i]![j]! += (A[i]?.[k] ?? 0) * (B[k]?.[j] ?? 0);
  return C;
}
function matVec(A: number[][], v: number[]): number[] {
  return A.map(row => row.reduce((s, a, j) => s + a * (v[j] ?? 0), 0));
}
// Frobenius norm: overall "size" of a matrix (√ of sum of squared entries).
// If this shrinks toward 0, any gradient passing through the matrix shrinks too.
function frobNorm(A: number[][]): number {
  let s = 0;
  for (const row of A) for (const a of row) s += a * a;
  return Math.sqrt(s);
}
function randMat(r: number, c: number, scale: number, rand: () => number): number[][] {
  return Array.from({ length: r }, () =>
    Array.from({ length: c }, () => (rand() - 0.5) * 2 * scale));
}
const fmtE = (v: number) => (isFinite(v) ? v.toExponential(3) : "overflow");

// ════════════════════════════════════════════════════════════════════
// PART 1 — A real tiny RNN forward pass
//   h_t = tanh(W_hh · h_{t-1} + W_xh · x_t)
// ════════════════════════════════════════════════════════════════════
const H = 4;              // hidden size (the "notepad" has 4 numbers)
const T = 50;             // sequence length
const rand = mulberry32(7);

const Whh = randMat(H, H, 0.5, rand);   // notepad → notepad (recurrent weights)
const Wxh = randMat(H, H, 0.5, rand);   // token   → notepad (input weights)
const xs: number[][] = Array.from({ length: T }, () =>
  Array.from({ length: H }, () => (rand() - 0.5) * 2)); // 50 fake token vectors

let h: number[] = Array<number>(H).fill(0);
const hs: number[][] = [];              // keep every hidden state for Part 2
console.log("=== PART 1: RNN forward pass (hidden size 4, 50 timesteps) ===\n");
for (let t = 0; t < T; t++) {
  const rec = matVec(Whh, h);           // what survives from the old notepad
  const inp = matVec(Wxh, xs[t] ?? []); // what the new token writes
  h = rec.map((r, i) => Math.tanh(r + (inp[i] ?? 0)));
  hs.push(h);
  if (t < 4 || t === T - 1)
    console.log(`  step ${String(t + 1).padStart(2)}: h = [${h.map(v => v.toFixed(3)).join(", ")}]`);
  if (t === 4) console.log("  ...");
}
console.log("\nOne 4-number notepad carries EVERYTHING the RNN knows about the past.");

// ════════════════════════════════════════════════════════════════════
// PART 2 — The vanishing gradient, measured
//
// How much does token 1 still influence the state at step t?
// Backprop multiplies one Jacobian per step:
//   J_t = ∂h_t/∂h_{t-1} = diag(1 − h_t²) · W_hh     (tanh′(z) = 1 − tanh(z)²)
// The gradient from step t back to step 1 is the PRODUCT J_t·J_{t-1}·...·J_2.
// We compute that product and print its norm as t grows.
// ════════════════════════════════════════════════════════════════════
function stepJacobian(hNew: number[], W: number[][]): number[][] {
  // diag(1 − h²) · W  — each row of W scaled by that unit's tanh slope
  return W.map((row, i) => {
    const slope = 1 - (hNew[i] ?? 0) * (hNew[i] ?? 0);
    return row.map(w => slope * w);
  });
}

console.log("\n=== PART 2: gradient of h_t with respect to h_1 (product of Jacobians) ===\n");
console.log("  steps back | ‖∏ J‖ (tanh RNN, W scale 0.5) ");
console.log("  -----------|------------------------------");
let P: number[][] = Array.from({ length: H }, (_, i) =>
  Array.from({ length: H }, (_, j) => (i === j ? 1 : 0))); // identity: 0 steps back
for (let t = 1; t < T; t++) {
  P = matmul(stepJacobian(hs[t] ?? [], Whh), P);            // one more step of chain rule
  if (t % 10 === 0 || t === 1 || t === T - 1)
    console.log(`     ${String(t).padStart(6)}    | ${fmtE(frobNorm(P))}`);
}
const vanished = frobNorm(P);
console.log(`\n  After 49 steps the gradient factor is ~${fmtE(vanished)}.`);
console.log("  Token 1's influence on learning at step 50 is numerically GONE.");

// The razor's edge: same experiment with a LINEAR recurrence (no tanh),
// isolating the pure matrix effect at three weight scales.
console.log("\n=== PART 2b: the razor's edge (linear recurrence, ‖W^t‖) ===\n");
console.log("  steps | W scale 0.3    | W scale 0.9    | W scale 1.3");
console.log("  ------|----------------|----------------|---------------");
const Ws = [0.3, 0.9, 1.3].map(s => randMat(H, H, s, mulberry32(11)));
let Ps: number[][][] = Ws.map(() =>
  Array.from({ length: H }, (_, i) => Array.from({ length: H }, (_, j) => (i === j ? 1 : 0))));
for (let t = 1; t <= T; t++) {
  Ps = Ps.map((Pk, k) => matmul(Ws[k] ?? [], Pk));
  if (t % 10 === 0)
    console.log(`    ${String(t).padStart(3)} | ${Ps.map(Pk => fmtE(frobNorm(Pk)).padStart(14)).join(" | ")}`);
}
console.log("\n  Too small → vanishes exponentially. Too big → explodes. Only a knife-edge");
console.log("  of scales (here ~0.9) survives 50 multiplications — the RNN's structural flaw.");

// ════════════════════════════════════════════════════════════════════
// PART 3 — Attention's answer: path length 1
//
// In a transformer, position 50 reads position 1 DIRECTLY:
//   output_50 = Σ_j weight_j · V_j   →   ∂output_50/∂V_1 = weight_1 · I
// The gradient back to token 1 passes through ONE factor, not 49.
// ════════════════════════════════════════════════════════════════════
console.log("\n=== PART 3: attention connects step 50 to step 1 in ONE hop ===\n");

// Toy attention scores from token 50's query against all 50 keys
const rand3 = mulberry32(99);
const scores = Array.from({ length: T }, () => (rand3() - 0.5) * 2);
const maxS = Math.max(...scores);
const exps = scores.map(s => Math.exp(s - maxS));
const Z = exps.reduce((a, b) => a + b, 0);
const weights = exps.map(e => e / Z);
const w1 = weights[0] ?? 0;

console.log(`  attention weight from position 50 to position 1: ${w1.toFixed(4)}`);
console.log(`  gradient factor through attention: ONE multiplication ≈ ${fmtE(w1)}`);
console.log(`  gradient factor through the RNN:   49 multiplications ≈ ${fmtE(vanished)}`);
console.log(`  ratio: attention's signal is ~${fmtE(w1 / Math.max(vanished, 1e-300))} times stronger\n`);

console.log("Scorecard:");
console.log("  RNN        : path length t−1, sequential training, gradient dies with distance");
console.log("  Attention  : path length 1,   parallel training,   gradient independent of distance");
console.log("  Price paid : attention computes all 50×50 = 2500 pairs (quadratic in length).");
console.log("  With enough data + hardware, that trade wins — and that's why transformers won.");

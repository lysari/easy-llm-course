// Generalization & overfitting: the polynomial-fitting laboratory.
// Fit degrees 1..15 to noisy points, watch train error fall forever while
// test error makes a U-turn — then rescue the overfit model with L2.
//
// Run: npx ts-node index.ts

// ── Seeded PRNG (mulberry32): same data every run ──
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
function gaussian(rand: () => number): number {
  let s = 0;
  for (let i = 0; i < 12; i++) s += rand();
  return s - 6;
}

// ── The ground truth the model is trying to discover ──
const trueFn = (x: number) => Math.sin(2 * x);
const NOISE = 0.15; // irreducible noise: no model can beat this floor

// ── Generate data: 16 training points, 40 test points, both noisy ──
const rand = mulberry32(1234);
function makeData(n: number): { xs: number[]; ys: number[] } {
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = -1 + (2 * i) / (n - 1); // evenly spread on [-1, 1]
    xs.push(x);
    ys.push(trueFn(x) + NOISE * gaussian(rand));
  }
  return { xs, ys };
}
const train = makeData(16);
const test = makeData(40); // different points AND different noise draws

// ── Least-squares polynomial fit (normal equations + Gaussian elimination) ──
// Model: ŷ = w0 + w1·x + w2·x² + ... + wd·x^d
// We minimize  Σ (ŷᵢ − yᵢ)²  +  λ·Σ wⱼ²   (λ = 0 → plain least squares)

// Design matrix row: [1, x, x², ..., x^d]
function features(x: number, degree: number): number[] {
  const row: number[] = [1];
  for (let j = 1; j <= degree; j++) row.push((row[j - 1] ?? 0) * x);
  return row;
}

// Solve the linear system A·w = b (Gaussian elimination, partial pivoting)
function solve(A: number[][], b: number[]): number[] {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i] ?? 0]); // augmented matrix
  for (let col = 0; col < n; col++) {
    // pick the row with the largest entry in this column (stability)
    let best = col;
    for (let r = col + 1; r < n; r++)
      if (Math.abs(M[r]?.[col] ?? 0) > Math.abs(M[best]?.[col] ?? 0)) best = r;
    const tmp = M[col]!; M[col] = M[best]!; M[best] = tmp;
    const pivot = M[col]?.[col] ?? 0;
    if (Math.abs(pivot) < 1e-300) continue; // singular direction: leave weight 0
    for (let r = col + 1; r < n; r++) {
      const f = (M[r]?.[col] ?? 0) / pivot;
      for (let c = col; c <= n; c++) M[r]![c] = (M[r]?.[c] ?? 0) - f * (M[col]?.[c] ?? 0);
    }
  }
  const w = Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i]?.[n] ?? 0;
    for (let c = i + 1; c < n; c++) s -= (M[i]?.[c] ?? 0) * (w[c] ?? 0);
    const piv = M[i]?.[i] ?? 0;
    w[i] = Math.abs(piv) < 1e-300 ? 0 : s / piv;
  }
  return w;
}

function fitPolynomial(xs: number[], ys: number[], degree: number, lambda = 0): number[] {
  const p = degree + 1;
  // Normal equations: (ΦᵀΦ + λI) w = Φᵀ y
  const A = Array.from({ length: p }, () => Array<number>(p).fill(0));
  const b = Array<number>(p).fill(0);
  for (let i = 0; i < xs.length; i++) {
    const phi = features(xs[i] ?? 0, degree);
    for (let r = 0; r < p; r++) {
      b[r] = (b[r] ?? 0) + (phi[r] ?? 0) * (ys[i] ?? 0);
      for (let c = 0; c < p; c++) A[r]![c] = (A[r]?.[c] ?? 0) + (phi[r] ?? 0) * (phi[c] ?? 0);
    }
  }
  for (let r = 0; r < p; r++) A[r]![r] = (A[r]?.[r] ?? 0) + lambda; // the L2 penalty
  return solve(A, b);
}

function predict(w: number[], x: number): number {
  const phi = features(x, w.length - 1);
  let y = 0;
  for (let j = 0; j < w.length; j++) y += (w[j] ?? 0) * (phi[j] ?? 0);
  return y;
}

function mse(w: number[], xs: number[], ys: number[]): number {
  let s = 0;
  for (let i = 0; i < xs.length; i++) {
    const e = predict(w, xs[i] ?? 0) - (ys[i] ?? 0);
    s += e * e;
  }
  return s / xs.length;
}

const maxAbsWeight = (w: number[]) => Math.max(...w.map(Math.abs));
const fmt = (v: number, digits = 5) =>
  Math.abs(v) >= 1e5 || (v !== 0 && Math.abs(v) < 1e-4) ? v.toExponential(2) : v.toFixed(digits);

// ════════════════════════════════════════════════════════════════════
// PART 1 — The U-curve: train vs test error as capacity grows
// ════════════════════════════════════════════════════════════════════
console.log("=== PART 1: fit degree 1..15 to 16 noisy points from y = sin(2x) ===");
console.log(`(noise σ = ${NOISE} → the best possible test MSE is about σ² = ${(NOISE * NOISE).toFixed(4)})\n`);
console.log("degree | train MSE   | test MSE    | max |weight| | verdict");
console.log("-------|-------------|-------------|--------------|---------");

let bestDeg = 1, bestTest = Infinity;
const results: { deg: number; tr: number; te: number; w: number[] }[] = [];
for (let deg = 1; deg <= 15; deg++) {
  const w = fitPolynomial(train.xs, train.ys, deg);
  const tr = mse(w, train.xs, train.ys);
  const te = mse(w, test.xs, test.ys);
  results.push({ deg, tr, te, w });
  if (te < bestTest) { bestTest = te; bestDeg = deg; }
}
for (const r of results) {
  const verdict =
    r.deg <= 2 ? "underfit" :
    r.deg === bestDeg ? "← best test error" :
    r.te > 3 * bestTest ? "OVERFIT" : "";
  console.log(
    `  ${String(r.deg).padStart(4)} | ${fmt(r.tr).padStart(11)} | ${fmt(r.te).padStart(11)} |` +
    ` ${fmt(maxAbsWeight(r.w), 2).padStart(12)} | ${verdict}`
  );
}
console.log("\nRead the columns:");
console.log("  train MSE only ever falls as degree grows (more capacity = better memorizing).");
console.log("  test MSE falls, bottoms out around degree", bestDeg + ",", "then climbs: the U-curve.");
console.log("  And look at max |weight|: wild swings need huge coefficients.");

// ════════════════════════════════════════════════════════════════════
// PART 2 — L2 regularization rescues the degree-15 model
// ════════════════════════════════════════════════════════════════════
console.log("\n=== PART 2: keep degree 15, add L2 penalty λ·Σw² ===\n");
console.log("     λ     | train MSE   | test MSE    | max |weight|");
console.log("-----------|-------------|-------------|-------------");
for (const lambda of [0, 1e-6, 1e-4, 1e-2, 1, 100]) {
  const w = fitPolynomial(train.xs, train.ys, 15, lambda);
  console.log(
    `${lambda.toExponential(0).padStart(10)} | ${fmt(mse(w, train.xs, train.ys)).padStart(11)} |` +
    ` ${fmt(mse(w, test.xs, test.ys)).padStart(11)} | ${fmt(maxAbsWeight(w), 2).padStart(11)}`
  );
}
console.log("\n  λ = 0    : memorizes the noise — tiny train error, terrible test error.");
console.log("  λ medium : same 16-coefficient model, but forced to be smooth —");
console.log("             test error drops near the best unregularized degree.");
console.log("  λ huge   : weights crushed toward 0 → the model underfits again.");
console.log("  Regularization is a capacity dial you can turn continuously.");

// ════════════════════════════════════════════════════════════════════
// PART 3 — What the fits actually predict between the training points
// ════════════════════════════════════════════════════════════════════
console.log("\n=== PART 3: predictions at x-values BETWEEN training points ===\n");
const w3 = fitPolynomial(train.xs, train.ys, 3);
const w15 = fitPolynomial(train.xs, train.ys, 15);
const w15reg = fitPolynomial(train.xs, train.ys, 15, 1e-2);
console.log("    x    |  true y  | deg 3    | deg 15 (λ=0) | deg 15 (λ=0.01)");
console.log("---------|----------|----------|--------------|----------------");
for (const x of [-0.97, -0.5, -0.03, 0.29, 0.61, 0.97]) {
  console.log(
    `  ${x.toFixed(2).padStart(5)}  | ${trueFn(x).toFixed(4).padStart(8)} |` +
    ` ${predict(w3, x).toFixed(4).padStart(8)} | ${fmt(predict(w15, x), 4).padStart(12)} |` +
    ` ${predict(w15reg, x).toFixed(4).padStart(10)}`
  );
}
console.log("\nThe unregularized degree-15 model can be far off BETWEEN the points it");
console.log("nailed exactly — that gap between memorizing and understanding is the");
console.log("whole subject of generalization.");

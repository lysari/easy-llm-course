// Efficiency research, two experiments:
//   1) QUANTIZATION — float32 → int8/int4 with scale + zero-point.
//      Per-weight reconstruction error, output drift through a tiny network,
//      and the outlier problem that made quantization a research field.
//   2) DISTILLATION — a student trained on a teacher's SOFT targets
//      generalizes better than the same student trained on hard labels.
//
// Run: npx ts-node AI-researcher-lessons/24-efficiency-research/index.ts

// ── helpers ─────────────────────────────────────────────────────────────────

// mulberry32 — seeded RNG so results are reproducible
function rng(seed: number): () => number {
  let t = seed;
  return () => {
    t = (t + 0x6d2b79f5) | 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

// ════════════════════════════════════════════════════════════════════════════
// PART 1 — Quantization: float32 → int8 → int4
// ════════════════════════════════════════════════════════════════════════════

// Affine quantization of a whole matrix with ONE scale + zero-point:
//   q = round(w/s) + z   stored as an integer in [qmin, qmax]
//   ŵ = s · (q − z)      reconstructed float
function quantize(W: number[][], bits: number) {
  const flat = W.flat();
  const min = Math.min(...flat), max = Math.max(...flat);
  const qmin = -(2 ** (bits - 1)), qmax = 2 ** (bits - 1) - 1; // int8: [-128,127]
  const s = (max - min) / (qmax - qmin) || 1e-12;              // real size of 1 step
  const z = Math.round(qmin - min / s);                        // integer that means 0.0
  const Q = W.map(row =>
    row.map(w => Math.max(qmin, Math.min(qmax, Math.round(w / s) + z)))
  );
  const What = Q.map(row => row.map(q => s * (q - z)));        // dequantized
  return { Q, What, s, z };
}

function reconErr(W: number[][], What: number[][]) {
  let sum = 0, max = 0, n = 0;
  for (let i = 0; i < W.length; i++)
    for (let j = 0; j < (W[i]?.length ?? 0); j++) {
      const e = Math.abs((W[i]?.[j] ?? 0) - (What[i]?.[j] ?? 0));
      sum += e; max = Math.max(max, e); n++;
    }
  return { mean: sum / n, max };
}

const rand = rng(42);
const gauss = () => {
  // Box–Muller: realistic bell-shaped weights, mostly in [-0.3, 0.3]
  const u = Math.max(rand(), 1e-12), v = rand();
  return 0.1 * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};
const mat = (r: number, c: number) =>
  Array.from({ length: r }, () => Array.from({ length: c }, gauss));

console.log("═══ Part 1: quantization (scale + zero-point) ═══\n");

const W1 = mat(8, 4);  // tiny 2-layer net: 4 inputs → 8 hidden → 3 outputs
const W2 = mat(3, 8);

for (const bits of [8, 4]) {
  const q1 = quantize(W1, bits);
  const e = reconErr(W1, q1.What);
  console.log(
    `int${bits}: scale=${q1.s.toFixed(5)} zero-point=${q1.z}  ` +
    `per-weight error: mean=${e.mean.toFixed(5)} max=${e.max.toFixed(5)}` +
    `  (${bits === 8 ? "255 buckets" : "15 buckets"} to cover the range)`
  );
}

// One weight end-to-end, so the formula is concrete:
const w0 = W1[0]?.[0] ?? 0;
const { Q, What, s, z } = quantize(W1, 8);
console.log(
  `\none weight walked through int8: w=${w0.toFixed(4)} → q=round(w/s)+z=${Q[0]?.[0]} ` +
  `→ ŵ=s·(q−z)=${(What[0]?.[0] ?? 0).toFixed(4)}  (4 bytes → 1 byte)`
);

// But per-weight error is not the number that matters — OUTPUT drift is.
// Run the tiny network full-precision vs quantized and compare outputs.
function forward(Wa: number[][], Wb: number[][], x: number[]): number[] {
  const h = Wa.map(row => Math.max(0, row.reduce((a, w, j) => a + w * (x[j] ?? 0), 0))); // ReLU
  return Wb.map(row => row.reduce((a, w, j) => a + w * (h[j] ?? 0), 0));
}
const argmax = (v: number[]) => v.indexOf(Math.max(...v));

console.log("\nOutput drift of the 2-layer net (100 random inputs):");
for (const bits of [8, 4]) {
  const W1q = quantize(W1, bits).What, W2q = quantize(W2, bits).What;
  let drift = 0, flips = 0, scale = 0;
  for (let t = 0; t < 100; t++) {
    const x = Array.from({ length: 4 }, () => rand() * 2 - 1);
    const y = forward(W1, W2, x), yq = forward(W1q, W2q, x);
    for (let k = 0; k < y.length; k++) {
      drift += Math.abs((y[k] ?? 0) - (yq[k] ?? 0));
      scale += Math.abs(y[k] ?? 0);
    }
    if (argmax(y) !== argmax(yq)) flips++;
  }
  console.log(
    `  int${bits}: mean |Δoutput| = ${(drift / 300).toFixed(5)} ` +
    `(${((100 * drift) / scale).toFixed(2)}% of output magnitude), ` +
    `argmax flips: ${flips}/100`
  );
}

// The outlier problem: ONE big weight stretches [min,max], so the shared scale
// s gets huge and every NORMAL weight loses precision. This single effect is
// why naive int8 broke >6B-param transformers (LLM.int8(), 2022).
const Wout = W1.map(row => row.slice());
Wout[0]![0] = 8.0; // inject one outlier (typical weights are ~±0.3)
const qNorm = quantize(W1, 8), qOut = quantize(Wout, 8);
// measure error on the NORMAL weights only (skip the outlier itself)
const normalErr = (W: number[][], What: number[][]) => {
  let sum = 0, n = 0;
  for (let i = 0; i < W.length; i++)
    for (let j = 0; j < (W[i]?.length ?? 0); j++) {
      if (i === 0 && j === 0) continue;
      sum += Math.abs((W[i]?.[j] ?? 0) - (What[i]?.[j] ?? 0)); n++;
    }
  return sum / n;
};
console.log("\nThe outlier problem (int8, one weight set to 8.0 among ~±0.3 weights):");
console.log(`  error on normal weights, no outlier:   ${normalErr(W1, qNorm.What).toFixed(5)}`);
console.log(`  error on normal weights, WITH outlier: ${normalErr(Wout, qOut.What).toFixed(5)}  ← ~${
  Math.round(normalErr(Wout, qOut.What) / normalErr(W1, qNorm.What))}× worse`);
console.log("  One outlier stretched the scale; everyone else got crushed into");
console.log("  a few buckets. Fixes (per-channel scales, outlier channels in");
console.log("  fp16, rotations) are exactly what quantization papers are about.");

// ════════════════════════════════════════════════════════════════════════════
// PART 2 — Distillation: soft targets beat hard labels
// ════════════════════════════════════════════════════════════════════════════
//
// Teacher: a fixed "big model" that knows a smooth circular boundary —
//   p(class 1 | x) = sigmoid(4·(x₁² + x₂² − 1))    (radius-1 circle)
// Students: two IDENTICAL tiny MLPs (2 → 8 → 1), same init, same 20 training
// points, same optimizer. Only the target differs:
//   hard student: trained on round(p)  ∈ {0, 1}     (what a dataset gives you)
//   soft student: trained on p itself  ∈ (0, 1)     (what the teacher gives you)
// Test: 900 grid points, scored against the teacher's true class.

const teacherP = (x1: number, x2: number) => sigmoid(4 * (x1 * x1 + x2 * x2 - 1));

// 20 training points
const rand2 = rng(7);
const train: { x: number[]; p: number }[] = [];
for (let i = 0; i < 20; i++) {
  const x = [rand2() * 3 - 1.5, rand2() * 3 - 1.5];
  train.push({ x, p: teacherP(x[0] ?? 0, x[1] ?? 0) });
}

// tiny MLP: 2 → H(tanh) → 1(sigmoid), trained with full-batch gradient descent
const H = 8;
type Net = { w1: number[][]; b1: number[]; w2: number[]; b2: number };
function initNet(seed: number): Net {
  const r = rng(seed);
  const g = () => (r() * 2 - 1) * 0.8;
  return {
    w1: Array.from({ length: H }, () => [g(), g()]),
    b1: Array.from({ length: H }, g),
    w2: Array.from({ length: H }, g),
    b2: 0,
  };
}
function predict(net: Net, x: number[]): { h: number[]; y: number } {
  const h = net.w1.map((row, i) =>
    Math.tanh((row[0] ?? 0) * (x[0] ?? 0) + (row[1] ?? 0) * (x[1] ?? 0) + (net.b1[i] ?? 0))
  );
  const y = sigmoid(h.reduce((a, hi, i) => a + hi * (net.w2[i] ?? 0), 0) + net.b2);
  return { h, y };
}
// binary cross-entropy works for soft targets too: −t·log y − (1−t)·log(1−y);
// its gradient at the logit is simply (y − t) in both cases.
function trainNet(net: Net, targets: number[], epochs: number, lr: number): void {
  for (let e = 0; e < epochs; e++) {
    for (let n = 0; n < train.length; n++) {
      const { x } = train[n]!;
      const t = targets[n] ?? 0;
      const { h, y } = predict(net, x);
      const dLogit = y - t; // d(BCE)/d(output logit)
      for (let i = 0; i < H; i++) {
        const dHi = dLogit * (net.w2[i] ?? 0) * (1 - (h[i] ?? 0) ** 2); // through tanh
        net.w2[i]! -= lr * dLogit * (h[i] ?? 0);
        net.w1[i]![0]! -= lr * dHi * (x[0] ?? 0);
        net.w1[i]![1]! -= lr * dHi * (x[1] ?? 0);
        net.b1[i]! -= lr * dHi;
      }
      net.b2 -= lr * dLogit;
    }
  }
}

const hardStudent = initNet(99);
const softStudent = initNet(99); // identical initialization — only targets differ
trainNet(hardStudent, train.map(d => Math.round(d.p)), 2000, 0.1);
trainNet(softStudent, train.map(d => d.p), 2000, 0.1);

// evaluate on a 30×30 grid against the teacher's true class
let hardOK = 0, softOK = 0, total = 0, hardMSE = 0, softMSE = 0;
for (let i = 0; i < 30; i++)
  for (let j = 0; j < 30; j++) {
    const x = [-1.5 + (3 * i) / 29, -1.5 + (3 * j) / 29];
    const p = teacherP(x[0] ?? 0, x[1] ?? 0);
    const truth = p > 0.5 ? 1 : 0;
    const yh = predict(hardStudent, x).y, ys = predict(softStudent, x).y;
    if ((yh > 0.5 ? 1 : 0) === truth) hardOK++;
    if ((ys > 0.5 ? 1 : 0) === truth) softOK++;
    hardMSE += (yh - p) ** 2; softMSE += (ys - p) ** 2;
    total++;
  }

console.log("\n═══ Part 2: distillation — soft targets vs hard labels ═══\n");
console.log("Teacher knows a circular boundary. Both students: same net, same");
console.log("init, same 20 training points. Only the training target differs.\n");
console.log(`  hard-label student: ${((100 * hardOK) / total).toFixed(1)}% test accuracy,  MSE vs teacher prob: ${(hardMSE / total).toFixed(4)}`);
console.log(`  soft-target student: ${((100 * softOK) / total).toFixed(1)}% test accuracy,  MSE vs teacher prob: ${(softMSE / total).toFixed(4)}`);
console.log("\nWhy: a hard label says only which side of the boundary a point is");
console.log("on. The soft target ALSO says how far from the boundary it is —");
console.log("teacher confidence is extra supervision on every single example.");
console.log("That per-example 'dark knowledge' is why distilled minis punch");
console.log("above their parameter count.");

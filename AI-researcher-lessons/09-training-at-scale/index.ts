// Training at scale, in numbers:
//   PART 1 — a training-memory calculator (params + grads + Adam state)
//   PART 2 — gradient accumulation proved equivalent to full-batch training
//   PART 3 — why fp32 master weights exist (fp16-style update rounding)
//   PART 4 — gradient clipping
//
// Run: npx ts-node index.ts

// ════════════════════════════════════════════════════════════════════
// PART 1 — The memory calculator
//
// Training state has four big slots:
//   weights, gradients, Adam m (momentum), Adam v (variance)
// Mixed precision adds a fifth: fp32 "master" weights that updates
// accumulate into safely.
// ════════════════════════════════════════════════════════════════════

interface MemoryBreakdown {
  label: string;
  slots: { name: string; bytes: number }[];
}

// dtype sizes in bytes per number
const FP32 = 4, FP16 = 2; // (bf16 is also 2 bytes — same arithmetic here)

function fp32AdamMemory(params: number): MemoryBreakdown {
  return {
    label: "fp32 + Adam",
    slots: [
      { name: "weights (fp32)", bytes: params * FP32 },
      { name: "gradients (fp32)", bytes: params * FP32 },
      { name: "Adam m (fp32)", bytes: params * FP32 },
      { name: "Adam v (fp32)", bytes: params * FP32 },
    ], // total: 16 bytes/param
  };
}

function mixedPrecisionAdamMemory(params: number): MemoryBreakdown {
  return {
    label: "mixed precision + Adam",
    slots: [
      { name: "weights (fp16)", bytes: params * FP16 },
      { name: "gradients (fp16)", bytes: params * FP16 },
      { name: "master weights (fp32)", bytes: params * FP32 },
      { name: "Adam m (fp32)", bytes: params * FP32 },
      { name: "Adam v (fp32)", bytes: params * FP32 },
    ], // total: 16 bytes/param — same total, but the fp16 halves buy FAST math
  };
}

function inferenceMemory(params: number): MemoryBreakdown {
  return {
    label: "inference only (fp16)",
    slots: [{ name: "weights (fp16)", bytes: params * FP16 }],
  };
}

const GB = 1024 ** 3;
const GPU_GB = 80; // one NVIDIA A100/H100-class device
const fmtGB = (b: number) =>
  b / GB >= 100 ? (b / GB).toFixed(0) : (b / GB).toFixed(2);

const models: { name: string; params: number }[] = [
  { name: "GPT-2 small", params: 124e6 },
  { name: "GPT-2 medium", params: 355e6 },
  { name: "GPT-2 large", params: 774e6 },
  { name: "GPT-2 XL", params: 1.5e9 },
  { name: "GPT-3", params: 175e9 },
];

console.log("=== PART 1: training memory calculator ===");
console.log(`(model state only — activations come on top; 1 GPU = ${GPU_GB} GB)\n`);

for (const m of models) {
  console.log(`── ${m.name}  (${(m.params / 1e6).toFixed(0)}M params) ──`);
  for (const plan of [fp32AdamMemory(m.params), mixedPrecisionAdamMemory(m.params), inferenceMemory(m.params)]) {
    const total = plan.slots.reduce((s, x) => s + x.bytes, 0);
    const gpus = Math.ceil(total / (GPU_GB * GB));
    const parts = plan.slots.map(s => `${s.name} ${fmtGB(s.bytes)}`).join(" + ");
    console.log(`  ${plan.label.padEnd(23)}: ${fmtGB(total).padStart(8)} GB  (${gpus} GPU${gpus > 1 ? "s" : ""})`);
    if (m.params === 175e9 && plan.label !== "inference only (fp16)")
      console.log(`      breakdown (GB): ${parts}`);
  }
  console.log();
}
console.log("Rule of thumb: Adam training ≈ 16 bytes per parameter — 8× the fp16");
console.log("inference size. GPT-3's training STATE alone needs 33 × 80GB GPUs,");
console.log("before a single activation is stored. Hence: parallelism + sharding.\n");

// ════════════════════════════════════════════════════════════════════
// PART 2 — Gradient accumulation === full batch (proved on a tiny model)
//
// Model: ŷ = w·x + b, loss = mean squared error over the batch.
// Gradients (calculus from ../../lessons/02-loss-function + 03-gradient-descent):
//   ∂L/∂w = mean over batch of  2·(ŷ − y)·x
//   ∂L/∂b = mean over batch of  2·(ŷ − y)
// ════════════════════════════════════════════════════════════════════
console.log("=== PART 2: gradient accumulation equivalence ===\n");

const w0 = 0.7, b0 = -0.2; // fixed starting weights for both methods
const batch: { x: number; y: number }[] = [
  { x: 1.0, y: 3.1 },
  { x: 2.0, y: 5.2 },
  { x: -1.5, y: -1.9 },
  { x: 0.5, y: 2.05 },
]; // roughly y = 2x + 1 with noise

function gradOnExamples(w: number, b: number, ex: { x: number; y: number }[]) {
  let gw = 0, gb = 0;
  for (const { x, y } of ex) {
    const err = w * x + b - y;
    gw += (2 * err * x) / ex.length;
    gb += (2 * err) / ex.length;
  }
  return { gw, gb };
}

// Method A: one full batch of 4
const full = gradOnExamples(w0, b0, batch);

// Method B: 4 microbatches of 1, gradients ACCUMULATED (no update in between)
let accW = 0, accB = 0;
for (const ex of batch) {
  const g = gradOnExamples(w0, b0, [ex]); // microbatch of size 1
  accW += g.gw; // just add — no weight update yet!
  accB += g.gb;
}
accW /= batch.length; // divide by microbatch count at the end
accB /= batch.length;

console.log(`  full batch of 4 :  dL/dw = ${full.gw.toFixed(16)}   dL/db = ${full.gb.toFixed(16)}`);
console.log(`  4 accumulated   :  dL/dw = ${accW.toFixed(16)}   dL/db = ${accB.toFixed(16)}`);
console.log(`  difference      :  ${Math.abs(full.gw - accW).toExponential(2)}, ${Math.abs(full.gb - accB).toExponential(2)}`);
console.log("  → identical (floating-point dust at worst). Addition is associative,");
console.log("    so you can buy a huge batch with a small GPU — paying in time.\n");

// ════════════════════════════════════════════════════════════════════
// PART 3 — Why fp32 master weights exist: update rounding
//
// fp16 has ~3 decimal digits of precision near 1.0. Simulate a weight at 1.0
// receiving a small update per step in "fp16" (rounded to steps of 2⁻¹⁰,
// fp16's spacing near 1.0) vs fp32.
// ════════════════════════════════════════════════════════════════════
console.log("=== PART 3: the fp16 update that rounds to nothing ===\n");

const roundFp16Near1 = (v: number) => Math.round(v * 1024) / 1024; // spacing 2⁻¹⁰ ≈ 0.001

const update = 0.0004; // a perfectly ordinary small gradient step
let wFp16 = 1.0, wFp32 = 1.0;
for (let step = 1; step <= 1000; step++) {
  wFp16 = roundFp16Near1(wFp16 + update); // 1.0 + 0.0004 rounds back to 1.0
  wFp32 = wFp32 + update;
}
console.log(`  per-step update: ${update}   (1000 steps)`);
console.log(`  fp16-style weight after 1000 steps: ${wFp16}   ← never moved!`);
console.log(`  fp32 master weight after 1000 steps: ${wFp32.toFixed(4)}`);
console.log("  → in half precision the update is smaller than the representable gap,");
console.log("    so learning silently stops. Fix: accumulate updates in an fp32 master");
console.log("    copy, round to fp16 only for the fast forward-pass math.\n");

// ════════════════════════════════════════════════════════════════════
// PART 4 — Gradient clipping: the seatbelt
// ════════════════════════════════════════════════════════════════════
console.log("=== PART 4: gradient clipping (max norm c = 1.0) ===\n");

function clipByNorm(g: number[], c: number): { clipped: number[]; norm: number } {
  const norm = Math.sqrt(g.reduce((s, v) => s + v * v, 0));
  if (norm <= c) return { clipped: g, norm };
  return { clipped: g.map(v => (v * c) / norm), norm };
}

for (const g of [[0.3, -0.2, 0.1], [40, -30, 20]]) {
  const { clipped, norm } = clipByNorm(g, 1.0);
  const newNorm = Math.sqrt(clipped.reduce((s, v) => s + v * v, 0));
  console.log(`  gradient [${g.join(", ")}]  ‖g‖ = ${norm.toFixed(3)}`);
  console.log(`    → after clip: [${clipped.map(v => v.toFixed(4)).join(", ")}]  ‖g‖ = ${newNorm.toFixed(3)}`);
}
console.log("\n  Direction preserved, length capped: one freak batch can no longer");
console.log("  fling 175B parameters into the wilderness. (GPT-3 used c = 1.0.)");

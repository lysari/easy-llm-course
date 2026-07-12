// Reproducibility, end to end:
//   PART 1 — mulberry32: a seedable PRNG you can read in full
//   PART 2 — same seed → IDENTICAL training runs; different seed → different
//   PART 3 — a mini experiment tracker (JSON lines) + a 5-seed study
//
// Run: npx ts-node index.ts     (writes runs.jsonl next to this file)

import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";

// ════════════════════════════════════════════════════════════════════
// PART 1 — A seedable PRNG (mulberry32)
//
// Math.random() cannot be seeded — so results using it can never be
// reproduced. mulberry32 is a tiny high-quality PRNG: 32 bits of state,
// marched forward and bit-scrambled each call.
// ════════════════════════════════════════════════════════════════════
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;                      // the PRNG's ENTIRE memory: 32 bits
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;        // march the state forward
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);      // scramble: xor-shift + multiply
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); // more scrambling
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296; // map 32-bit int → [0, 1)
  };
}

console.log("=== PART 1: a seedable PRNG ===\n");
const r42a = mulberry32(42);
const r42b = mulberry32(42);
const r43 = mulberry32(43);
const take = (r: () => number, n: number) =>
  Array.from({ length: n }, () => r().toFixed(4)).join(", ");
console.log(`  seed 42, first call : ${take(r42a, 5)}`);
console.log(`  seed 42, fresh again: ${take(r42b, 5)}   ← identical sequence`);
console.log(`  seed 43             : ${take(r43, 5)}   ← different universe`);

// ════════════════════════════════════════════════════════════════════
// PART 2 — Seeded training runs of a tiny model
//
// Model: ŷ = w·x + b fit by SGD to noisy data y = 2x + 1 + noise.
// The seed controls EVERY random choice: init, data noise, shuffling.
// ════════════════════════════════════════════════════════════════════
interface Config { lr: number; epochs: number; nPoints: number; noise: number }

function trainRun(seed: number, cfg: Config): { finalLoss: number; w: number; b: number } {
  const rand = mulberry32(seed);
  const gauss = () => { let s = 0; for (let i = 0; i < 12; i++) s += rand(); return s - 6; };

  // randomness door 1: the dataset's noise draws
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < cfg.nPoints; i++) {
    const x = -1 + (2 * i) / (cfg.nPoints - 1);
    xs.push(x);
    ys.push(2 * x + 1 + cfg.noise * gauss());
  }
  // randomness door 2: weight initialization
  let w = (rand() - 0.5) * 2, b = (rand() - 0.5) * 2;

  for (let epoch = 0; epoch < cfg.epochs; epoch++) {
    // randomness door 3: data shuffling (Fisher–Yates with OUR rng)
    const order = Array.from({ length: cfg.nPoints }, (_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const t = order[i] ?? 0; order[i] = order[j] ?? 0; order[j] = t;
    }
    for (const idx of order) { // SGD: one example at a time
      const x = xs[idx] ?? 0, y = ys[idx] ?? 0;
      const err = w * x + b - y;
      w -= cfg.lr * 2 * err * x;
      b -= cfg.lr * 2 * err;
    }
  }
  let loss = 0;
  for (let i = 0; i < cfg.nPoints; i++) {
    const e = w * (xs[i] ?? 0) + b - (ys[i] ?? 0);
    loss += (e * e) / cfg.nPoints;
  }
  return { finalLoss: loss, w, b };
}

const config: Config = { lr: 0.05, epochs: 30, nPoints: 40, noise: 0.2 };

console.log("\n=== PART 2: same seed → identical run; different seed → different ===\n");
const runA = trainRun(42, config);
const runB = trainRun(42, config);
const runC = trainRun(7, config);
console.log(`  seed 42, run 1: finalLoss = ${runA.finalLoss.toFixed(15)}  w = ${runA.w.toFixed(15)}`);
console.log(`  seed 42, run 2: finalLoss = ${runB.finalLoss.toFixed(15)}  w = ${runB.w.toFixed(15)}`);
console.log(`  identical to every bit? ${runA.finalLoss === runB.finalLoss && runA.w === runB.w && runA.b === runB.b}`);
console.log(`\n  seed  7, run 1: finalLoss = ${runC.finalLoss.toFixed(15)}  w = ${runC.w.toFixed(15)}`);
console.log("  → same code + same config + different seed = a DIFFERENT result.");
console.log("    One run is one sample from a distribution, not 'the' result.");
console.log("    (On GPUs even fixed seeds can differ bitwise — float addition");
console.log("     order is nondeterministic — so log seeds AND expect spread.)");

// ════════════════════════════════════════════════════════════════════
// PART 3 — The mini experiment tracker + a 5-seed study
//
// One JSON object per line, appended to runs.jsonl IN THIS FOLDER.
// Each record carries everything needed to recreate the number.
// ════════════════════════════════════════════════════════════════════
interface RunRecord {
  runId: string;
  timestamp: string;
  gitHash: string;
  seed: number;
  config: Config;
  finalLoss: number;
}

const LOG_PATH = path.join(__dirname, "runs.jsonl");

function currentGitHash(): string {
  try {
    return execSync("git rev-parse --short HEAD", { cwd: __dirname }).toString().trim();
  } catch {
    return "no-git";
  }
}

function logRun(record: RunRecord): void {
  fs.appendFileSync(LOG_PATH, JSON.stringify(record) + "\n");
}

console.log("\n=== PART 3: 'it worked once' vs 'it works' — a 5-seed study ===\n");
fs.writeFileSync(LOG_PATH, ""); // fresh log each demo run (real trackers append forever)

const gitHash = currentGitHash();
const seeds = [1, 2, 3, 4, 5];
const losses: number[] = [];

for (const seed of seeds) {
  const { finalLoss } = trainRun(seed, config);
  losses.push(finalLoss);
  const record: RunRecord = {
    runId: `run-${seed}-${Date.now().toString(36)}`,
    timestamp: new Date().toISOString(),
    gitHash,
    seed,
    config,
    finalLoss,
  };
  logRun(record);
  console.log(`  seed ${seed}: finalLoss = ${finalLoss.toFixed(6)}   → logged to runs.jsonl`);
}

// mean ± std (std: how big is seed luck?)
const mean = losses.reduce((a, b) => a + b, 0) / losses.length;
const variance = losses.reduce((s, l) => s + (l - mean) * (l - mean), 0) / losses.length;
const std = Math.sqrt(variance);
const min = Math.min(...losses), max = Math.max(...losses);

console.log(`\n  result over ${seeds.length} seeds: finalLoss = ${mean.toFixed(6)} ± ${std.toFixed(6)}  (mean ± std)`);
console.log(`  best seed: ${min.toFixed(6)}   worst seed: ${max.toFixed(6)}   spread: ${(max - min).toExponential(2)}`);
console.log(`\n  "It worked once"  = reporting ${min.toFixed(6)} (the luckiest seed).`);
console.log(`  "It works"        = reporting ${mean.toFixed(6)} ± ${std.toFixed(6)} over seeds ${seeds.join(",")}.`);
console.log("  Any claimed improvement SMALLER than the ± is indistinguishable");
console.log("  from seed luck — run more seeds or make a humbler claim.");
console.log(`\n  Full audit trail written to: ${LOG_PATH}`);
console.log(`  (each line: runId, timestamp, gitHash=${gitHash}, seed, full config, finalLoss)`);

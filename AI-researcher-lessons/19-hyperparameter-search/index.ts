// Hyperparameter search: grid vs random vs random + successive halving,
// under the SAME total budget.
//
// Instead of really training models we use a synthetic "training outcome"
// function whose true optimum we know — so we can score each search strategy
// by how close it got. The function is rigged the way real problems usually
// are: the learning rate matters a LOT, weight decay barely matters.
//
//   trueScore(lr, wd) — peak 0.90 at lr = 3e-3, wd = 1e-4
//     • sensitive along log10(lr):  width 0.4 decades
//     • insensitive along log10(wd): width 3.0 decades
//
// Evaluating a config for e "epochs" returns trueScore + noise/√e :
// cheap evaluations are noisy, long evaluations are reliable.

// ── Seeded RNG ──
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
function gauss(rand: () => number): number {
  const u = Math.max(rand(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

// ── The rigged objective (unknown to the searcher, known to us) ──
const LR_OPT = 3e-3;
const WD_OPT = 1e-4;
function trueScore(lr: number, wd: number): number {
  const dx = (Math.log10(lr) - Math.log10(LR_OPT)) / 0.4; // lr: narrow ridge
  const dy = (Math.log10(wd) - Math.log10(WD_OPT)) / 3.0; // wd: nearly flat
  return 0.9 * Math.exp(-0.5 * dx * dx - 0.5 * dy * dy);
}

// One noisy "training run" of e epochs. More epochs → less noise.
const NOISE = 0.05;
function evaluate(lr: number, wd: number, epochs: number, rand: () => number): number {
  return trueScore(lr, wd) + (NOISE / Math.sqrt(epochs)) * gauss(rand);
}

// Search space: lr ∈ 10^[-4, 0], wd ∈ 10^[-6, -2]
const LR_EXP: [number, number] = [-4, 0];
const WD_EXP: [number, number] = [-6, -2];
const TOTAL_BUDGET = 100; // total epochs each strategy may spend

interface Config { lr: number; wd: number }
interface Result { name: string; best: Config; bestTrue: number; distinctLr: number; spent: number }

// ── Strategy 1: grid search — 25 points × 4 epochs = 100 epochs ──
function gridSearch(): Result {
  const rand = mulberry32(101);
  const lrs = [0, 1, 2, 3, 4].map(i => Math.pow(10, LR_EXP[0] + i)); // 1e-4 … 1e0
  const wds = [0, 1, 2, 3, 4].map(i => Math.pow(10, WD_EXP[0] + i)); // 1e-6 … 1e-2
  let best: Config = { lr: lrs[0] ?? 1e-4, wd: wds[0] ?? 1e-6 };
  let bestObs = -Infinity;
  let spent = 0;
  for (const lr of lrs)
    for (const wd of wds) {
      const obs = evaluate(lr, wd, 4, rand);
      spent += 4;
      if (obs > bestObs) { bestObs = obs; best = { lr, wd }; }
    }
  return { name: "grid  (5×5, 4 epochs each)", best, bestTrue: trueScore(best.lr, best.wd), distinctLr: lrs.length, spent };
}

// Log-uniform sample: uniform in the exponent, so each decade is equally likely
function logUniform(range: [number, number], rand: () => number): number {
  return Math.pow(10, range[0] + (range[1] - range[0]) * rand());
}

// ── Strategy 2: random search — 25 samples × 4 epochs = 100 epochs ──
function randomSearch(): Result {
  const rand = mulberry32(202);
  let best: Config = { lr: 1e-4, wd: 1e-6 };
  let bestObs = -Infinity;
  let spent = 0;
  for (let i = 0; i < 25; i++) {
    const c = { lr: logUniform(LR_EXP, rand), wd: logUniform(WD_EXP, rand) };
    const obs = evaluate(c.lr, c.wd, 4, rand);
    spent += 4;
    if (obs > bestObs) { bestObs = obs; best = c; }
  }
  return { name: "random (25 samples, 4 epochs)", best, bestTrue: trueScore(best.lr, best.wd), distinctLr: 25, spent };
}

// ── Strategy 3: random + successive halving, same 100-epoch budget ──
// 24 configs × 1 epoch → keep 8 × 3 more → keep 3 × 8 more → keep 1 × 28 more
function successiveHalving(): Result {
  const rand = mulberry32(303);
  let pool: Array<{ c: Config; score: number; epochs: number }> = [];
  for (let i = 0; i < 24; i++)
    pool.push({ c: { lr: logUniform(LR_EXP, rand), wd: logUniform(WD_EXP, rand) }, score: 0, epochs: 0 });
  let spent = 0;

  const rounds: Array<{ keep: number; addEpochs: number }> = [
    { keep: 24, addEpochs: 1 }, // screen everyone cheaply
    { keep: 8, addEpochs: 3 }, // survivors get more epochs
    { keep: 3, addEpochs: 8 },
    { keep: 1, addEpochs: 28 }, // finalist trained long (reliable estimate)
  ];
  for (const { keep, addEpochs } of rounds) {
    pool.sort((a, b) => b.score - a.score);
    pool = pool.slice(0, keep);
    for (const entry of pool) {
      entry.epochs += addEpochs;
      // re-evaluate at the larger cumulative budget (less noisy now)
      entry.score = evaluate(entry.c.lr, entry.c.wd, entry.epochs, rand);
      spent += addEpochs;
    }
  }
  const winner = pool[0]?.c ?? { lr: 1e-4, wd: 1e-6 };
  return { name: "random + successive halving", best: winner, bestTrue: trueScore(winner.lr, winner.wd), distinctLr: 24, spent };
}

// ── Run all three under equal budget and compare against the known optimum ──
const results = [gridSearch(), randomSearch(), successiveHalving()];
const optimum = trueScore(LR_OPT, WD_OPT);

console.log("=== Searching lr ∈ [1e-4, 1] × wd ∈ [1e-6, 1e-2], budget = 100 epochs each ===");
console.log(`True optimum: score ${optimum.toFixed(3)} at lr=${LR_OPT}, wd=${WD_OPT}`);
console.log("(lr matters a lot: width 0.4 decades. wd barely matters: width 3 decades.)\n");

console.log("strategy                        distinct lr   epochs   best lr    best TRUE score");
console.log("------------------------------  -----------  -------  ---------  ---------------");
for (const r of results) {
  console.log(
    `${r.name.padEnd(32)}${String(r.distinctLr).padStart(6)}       ${String(r.spent).padStart(5)}   ${r.best.lr.toExponential(1).padStart(8)}   ${r.bestTrue.toFixed(3)}  (${((100 * r.bestTrue) / optimum).toFixed(0)}% of optimum)`
  );
}

console.log("\n=== Why the ranking comes out this way ===");
console.log("• Grid spent 100 epochs but tried only 5 DISTINCT learning rates —");
console.log("  every lr was repeated at 5 wd values that changed almost nothing.");
console.log("  Its best possible lr is a full half-decade from the optimum.");
console.log("• Random tried 25 distinct lrs for the same price, so some landed");
console.log("  inside the narrow good ridge. (Bergstra & Bengio's argument.)");
console.log("• Successive halving screened 24 configs at 1 epoch, then spent the");
console.log("  saved budget training survivors longer — breadth AND a reliable");
console.log("  final estimate, under the exact same 100-epoch budget.");
console.log("\nBudget honesty: whichever strategy you use, its total cost");
console.log(`(${TOTAL_BUDGET} epochs here) is part of your method's cost. Report it.`);

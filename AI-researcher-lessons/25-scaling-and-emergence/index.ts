// Scaling & emergence, two experiments:
//   1) THE COMPUTE-OPTIMAL FRONTIER — take the Chinchilla loss formula
//      L(N,D) = E + A/N^α + B/D^β, sweep compute budgets, numerically find
//      the best (N, D) split at each budget, and watch the ~20 tokens/param
//      rule fall out of the fitted constants.
//   2) THE EMERGENCE MIRAGE — one smoothly improving skill, scored with three
//      different metrics. One metric shows a sudden "emergent" jump; the
//      others show the smooth truth. (Schaeffer et al., 2023)
//
// Run: npx ts-node AI-researcher-lessons/25-scaling-and-emergence/index.ts

// ════════════════════════════════════════════════════════════════════════════
// PART 1 — The compute-optimal frontier
// ════════════════════════════════════════════════════════════════════════════

// Chinchilla-style constants (Hoffmann et al. 2022, as refit by the
// Epoch AI replication, Besiroglu et al. 2024 — the refit is the version
// consistent with the famous "≈20 tokens per parameter" rule):
const E = 1.82;      // irreducible loss: the entropy of language itself
const A = 482.01;    // } "model too small" tax:  A / N^alpha
const alpha = 0.3478;
const B = 2085.43;   // } "data too little" tax:  B / D^beta
const beta = 0.3658;

// L(N, D): predicted final cross-entropy loss (nats/token)
const loss = (N: number, D: number) => E + A / Math.pow(N, alpha) + B / Math.pow(D, beta);

// Training compute accounting: C ≈ 6·N·D FLOPs (2 forward + 4 backward
// FLOPs per parameter per token). Fix C, then D is determined by N.
const dataForBudget = (C: number, N: number) => C / (6 * N);

// Numerically find the loss-minimizing N for a budget C: fine grid search
// over log10(N) — no calculus needed, and you can verify the answer is a
// true minimum by looking at the neighbors.
function optimalSplit(C: number): { N: number; D: number; L: number } {
  let best = { N: 0, D: 0, L: Infinity };
  for (let lg = 5; lg <= 14; lg += 0.001) {
    const N = Math.pow(10, lg);
    const D = dataForBudget(C, N);
    const L = loss(N, D);
    if (L < best.L) best = { N, D, L };
  }
  return best;
}

const fmt = (x: number): string => {
  if (x >= 1e12) return (x / 1e12).toFixed(2) + "T";
  if (x >= 1e9) return (x / 1e9).toFixed(2) + "B";
  if (x >= 1e6) return (x / 1e6).toFixed(2) + "M";
  return x.toExponential(1);
};

console.log("═══ Part 1: the compute-optimal frontier ═══\n");
console.log("L(N,D) = E + A/N^α + B/D^β, budget C = 6·N·D, solve min L per C:\n");
console.log("  budget C (FLOPs)   optimal N   optimal D    loss    tokens/param");
console.log("  ────────────────  ──────────  ──────────  ───────  ────────────");
for (let exp = 18; exp <= 26; exp++) {
  const C = Math.pow(10, exp);
  const { N, D, L } = optimalSplit(C);
  console.log(
    `  1e${exp}             ${fmt(N).padStart(9)}  ${fmt(D).padStart(9)}  ` +
    `${L.toFixed(3).padStart(7)}  ${(D / N).toFixed(1).padStart(9)}`
  );
}
console.log("\nThe ratio hovers around ~20 tokens per parameter across 8 orders");
console.log("of magnitude — that is the Chinchilla rule. (It drifts slowly");
console.log("because α ≠ β in the fit; it is a rule of thumb, not a constant.)");

// Sanity check against history: Chinchilla itself, and GPT-3's allocation.
const cChin = 6 * 70e9 * 1.4e12;
console.log(`\nChinchilla the model (N=70B, D=1.4T, C=${cChin.toExponential(1)}):`);
const optChin = optimalSplit(cChin);
console.log(
  `  optimal split at that budget: N=${fmt(optChin.N)}, D=${fmt(optChin.D)}` +
  `  → the real run sits almost exactly on the frontier.`
);

const cGpt3 = 6 * 175e9 * 300e9;
const lGpt3 = loss(175e9, 300e9);
const optG = optimalSplit(cGpt3);
console.log(`\nGPT-3's allocation (N=175B, D=300B — 1.7 tokens/param):`);
console.log(`  predicted loss of GPT-3 split:      ${lGpt3.toFixed(3)}`);
console.log(`  optimal split, SAME compute (${fmt(optG.N)}, ${fmt(optG.D)}): ${optG.L.toFixed(3)}`);
console.log("  Same money, lower loss, 4× smaller model — Kaplan-era scaling");
console.log("  had the field building models ~10× too big for their data.");

// ════════════════════════════════════════════════════════════════════════════
// PART 2 — The emergence mirage
// ════════════════════════════════════════════════════════════════════════════
//
// Ground truth: PER-DIGIT accuracy p on 5-digit arithmetic improves smoothly
// with scale (a gentle logistic in log-compute — no jumps anywhere).
// We score the SAME underlying skill three ways:
//   metric A: per-digit accuracy         = p            (smooth — the truth)
//   metric B: ≥4 of 5 digits correct     = binomial     (steep-ish)
//   metric C: EXACT MATCH, all 5 digits  = p^5          (a cliff)
// Metric C is how benchmarks actually score arithmetic. Watch it "emerge".

const sigmoidFn = (x: number) => 1 / (1 + Math.exp(-x));
const perDigit = (logC: number) => 0.02 + 0.96 * sigmoidFn(0.7 * (logC - 21)); // smooth!

const binom = (n: number, k: number): number => {
  let r = 1;
  for (let i = 0; i < k; i++) r = (r * (n - i)) / (i + 1);
  return r;
};
const atLeast = (n: number, k: number, p: number): number => {
  let s = 0;
  for (let j = k; j <= n; j++) s += binom(n, j) * Math.pow(p, j) * Math.pow(1 - p, n - j);
  return s;
};

console.log("\n═══ Part 2: the emergence mirage ═══\n");
console.log("One skill, improving smoothly. Three ways to score it:\n");
console.log("  scale     per-digit p   ≥4-of-5    exact-match p⁵");
console.log("  ────────  ───────────  ─────────  ──────────────");

const bar = (v: number) => "█".repeat(Math.round(v * 12)).padEnd(12, "·");
const rows: { lg: number; a: number; b: number; c: number }[] = [];
for (let lg = 17; lg <= 25; lg += 0.5) {
  const p = perDigit(lg);
  rows.push({ lg, a: p, b: atLeast(5, 4, p), c: Math.pow(p, 5) });
}
for (const r of rows) {
  console.log(
    `  1e${r.lg.toFixed(1).padEnd(4)}   ` +
    `${(100 * r.a).toFixed(1).padStart(5)}% ${bar(r.a)}` +
    `  ${(100 * r.b).toFixed(1).padStart(5)}%` +
    `  ${(100 * r.c).toFixed(1).padStart(5)}% ${bar(r.c)}`
  );
}

// Quantify "suddenness" two ways per metric:
//   max jump  — biggest gain in one scale step
//   dead zone — how much of the sweep passes before the metric reaches
//               even 10% of its final value ("looks like nothing happens")
function suddenness(get: (r: { a: number; b: number; c: number }) => number) {
  let maxJump = 0;
  for (let i = 1; i < rows.length; i++)
    maxJump = Math.max(maxJump, get(rows[i]!) - get(rows[i - 1]!));
  const final = get(rows[rows.length - 1]!);
  const idx = rows.findIndex(r => get(r) >= 0.1 * final);
  return { maxJump, deadFrac: idx / (rows.length - 1) };
}
console.log("\nHow sudden does each metric make the SAME smooth progress look?");
console.log("                              biggest 1-step jump   flat 'dead zone' first");
const label = ["per-digit (smooth metric) ", "≥4-of-5                   ", "exact-match (hard metric) "];
[
  suddenness(r => r.a),
  suddenness(r => r.b),
  suddenness(r => r.c),
].forEach((s, i) => {
  console.log(
    `  ${label[i]}   +${(100 * s.maxJump).toFixed(1).padStart(4)} points` +
    `          ${(100 * s.deadFrac).toFixed(0).padStart(3)}% of the sweep` +
    (i === 2 ? '   ← "emergence!"' : "")
  );
});

console.log("\nSame model at every scale — the underlying skill never jumped.");
console.log("Exact-match awards zero credit until per-digit accuracy is already");
console.log("high, then pays out all at once: the discontinuity lives in the");
console.log("METRIC, not (necessarily) in the model. Schaeffer et al. (2023)");
console.log("showed most claimed emergent abilities soften under smooth metrics.");
console.log("But note both truths: for deployment, exact-match may be what");
console.log("reality grades you on — a threshold crossed is still a threshold.");

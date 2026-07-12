// Statistical significance from scratch: seeds as repeated experiments,
// mean ± std, standard error, Welch's t-test, p-values — and the
// multiple-comparisons trap that manufactures fake discoveries.
//
// Part 1: two methods whose TRUE means really do differ by a small margin.
//         5 seeds each → compute every statistic by hand → verdict.
// Part 2: twenty variants with NO true difference at all. Watch at least
//         one of them "win significantly" by pure seed luck.

// ── Seeded RNG + gaussian noise ──
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

// One "training run": true skill of the method + seed-to-seed wobble.
// (In real life this line is hours of GPU time. Here it is a gaussian.)
const SEED_NOISE = 0.5; // std dev of accuracy across seeds, in points
function runExperiment(trueMean: number, rand: () => number): number {
  return trueMean + SEED_NOISE * gauss(rand);
}

// ── Basic statistics, from scratch ──
function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}
function sampleVariance(xs: number[]): number {
  const m = mean(xs);
  return xs.reduce((a, x) => a + (x - m) * (x - m), 0) / (xs.length - 1);
}

// ── Welch's t-test ──
function welch(a: number[], b: number[]): { t: number; df: number; p: number } {
  const va = sampleVariance(a) / a.length; // s²_A / n_A
  const vb = sampleVariance(b) / b.length; // s²_B / n_B
  const t = (mean(b) - mean(a)) / Math.sqrt(va + vb);
  // Welch–Satterthwaite degrees of freedom
  const df =
    ((va + vb) * (va + vb)) /
    ((va * va) / (a.length - 1) + (vb * vb) / (b.length - 1));
  return { t, df, p: twoSidedP(t, df) };
}

// p-value = P(|T| ≥ |t|) under the t-distribution with df degrees of freedom.
// We integrate the t density numerically (Simpson's rule) — no lookup tables.
function lgamma(x: number): number {
  // Lanczos approximation
  const g = [76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let ser = 1.000000000190015;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  for (let j = 0; j < 6; j++) ser += (g[j] ?? 0) / (x + 1 + j);
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}
function tPdf(x: number, df: number): number {
  const logC = lgamma((df + 1) / 2) - lgamma(df / 2) - 0.5 * Math.log(df * Math.PI);
  return Math.exp(logC - ((df + 1) / 2) * Math.log(1 + (x * x) / df));
}
function twoSidedP(t: number, df: number): number {
  const T = Math.abs(t);
  // integrate pdf from 0 to T with Simpson's rule
  const steps = 2000;
  const h = T / steps;
  let area = tPdf(0, df) + tPdf(T, df);
  for (let i = 1; i < steps; i++) area += tPdf(i * h, df) * (i % 2 === 1 ? 4 : 2);
  area *= h / 3;
  return Math.max(0, Math.min(1, 1 - 2 * area)); // 2 × tail mass
}

const fmt = (x: number, d = 2) => x.toFixed(d);

// ════════════════════════════════════════════════════════════════════
// PART 1 — a real (but small) difference, measured properly
// ════════════════════════════════════════════════════════════════════
const TRUE_BASELINE = 81.5; // the truth, never visible in real life:
const TRUE_METHOD = 82.4; //   the new method really is +0.9 better

const rand1 = mulberry32(17);
const N_SEEDS = 5;
const baseline: number[] = [];
const method: number[] = [];
for (let s = 0; s < N_SEEDS; s++) baseline.push(runExperiment(TRUE_BASELINE, rand1));
for (let s = 0; s < N_SEEDS; s++) method.push(runExperiment(TRUE_METHOD, rand1));

console.log("=== Part 1: is the new method really better? (5 seeds each) ===\n");
console.log(`baseline runs:   [${baseline.map(x => fmt(x, 1)).join(", ")}]`);
console.log(`new method runs: [${method.map(x => fmt(x, 1)).join(", ")}]\n`);

const mA = mean(baseline), mB = mean(method);
const sA = Math.sqrt(sampleVariance(baseline)), sB = Math.sqrt(sampleVariance(method));
const seA = sA / Math.sqrt(N_SEEDS), seB = sB / Math.sqrt(N_SEEDS);

console.log(`baseline:   mean ${fmt(mA)} ± std ${fmt(sA)},  SE = ${fmt(sA)}/√5 = ${fmt(seA)}`);
console.log(`new method: mean ${fmt(mB)} ± std ${fmt(sB)},  SE = ${fmt(sB)}/√5 = ${fmt(seB)}`);
console.log(`~95% CI baseline:   [${fmt(mA - 2 * seA)}, ${fmt(mA + 2 * seA)}]`);
console.log(`~95% CI new method: [${fmt(mB - 2 * seB)}, ${fmt(mB + 2 * seB)}]\n`);

const w = welch(baseline, method);
console.log(`observed gap:      ${fmt(mB - mA)} points`);
console.log(`Welch t statistic: ${fmt(w.t)}   (gap ÷ expected wobble of the gap)`);
console.log(`degrees of freedom (Welch–Satterthwaite): ${fmt(w.df, 1)}`);
console.log(`two-sided p-value: ${w.p.toFixed(4)}`);
console.log(
  w.p < 0.05
    ? `\nVERDICT: p < 0.05 — a gap this large arises by seed luck only ${(100 * w.p).toFixed(1)}%`
    : `\nVERDICT: p ≥ 0.05 — this gap is within what seed luck produces; NOT supported`
);
if (w.p < 0.05)
  console.log("of the time, so we call the improvement significant. (True gap: +0.9.)");

// ════════════════════════════════════════════════════════════════════
// PART 2 — the multiple-comparisons trap
// ════════════════════════════════════════════════════════════════════
console.log("\n=== Part 2: run 20 USELESS variants, one 'wins' by luck ===\n");
console.log("Every variant below has TRUE mean exactly equal to the baseline");
console.log(`(${TRUE_BASELINE}). Any 'significant win' is pure seed noise.\n`);

const rand2 = mulberry32(2);
const base2: number[] = [];
for (let s = 0; s < N_SEEDS; s++) base2.push(runExperiment(TRUE_BASELINE, rand2));

let luckyName = "";
let luckyP = 1;
let luckyGap = 0;
let significant = 0;
for (let v = 1; v <= 20; v++) {
  const scores: number[] = [];
  for (let s = 0; s < N_SEEDS; s++) scores.push(runExperiment(TRUE_BASELINE, rand2));
  const res = welch(base2, scores);
  const gap = mean(scores) - mean(base2);
  const flag = res.p < 0.05 && gap > 0 ? "  ← 'significant'!" : "";
  if (res.p < 0.05 && gap > 0) significant++;
  if (res.p < luckyP && gap > 0) { luckyP = res.p; luckyGap = gap; luckyName = `variant-${v}`; }
  console.log(
    `variant-${String(v).padStart(2)}: gap ${gap >= 0 ? "+" : ""}${fmt(gap)}  p=${res.p.toFixed(3)}${flag}`
  );
}

console.log(`\n${significant} of 20 no-difference variants crossed p < 0.05.`);
console.log(`Best looker: ${luckyName}, gap +${fmt(luckyGap)}, p=${luckyP.toFixed(3)}.`);
console.log("With 20 tries, P(at least one fluke) = 1 − 0.95^20 ≈ 64%. The");
console.log("trap: write the paper about that variant and quietly forget the 19.");

// The cheap, brutal defense: rerun the winner on FRESH seeds.
const fresh: number[] = [];
const freshBase: number[] = [];
for (let s = 0; s < N_SEEDS; s++) freshBase.push(runExperiment(TRUE_BASELINE, rand2));
for (let s = 0; s < N_SEEDS; s++) fresh.push(runExperiment(TRUE_BASELINE, rand2)); // true mean unchanged!
const confirmTest = welch(freshBase, fresh);
console.log(`\nConfirmation run of ${luckyName} on 5 fresh seeds:`);
console.log(`gap ${mean(fresh) - mean(freshBase) >= 0 ? "+" : ""}${fmt(mean(fresh) - mean(freshBase))},  p=${confirmTest.p.toFixed(3)} — the 'discovery' evaporates.`);
console.log("A lucky winner regresses to nothing; a real effect survives.");
console.log("Always re-verify your best variant on fresh randomness.");

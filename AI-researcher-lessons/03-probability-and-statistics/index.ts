// Probability from scratch: sampling, Box-Muller Gaussians, the
// Central Limit Theorem with real dice, Bayes' medical test, and
// fitting a coin by maximum likelihood (the same math as LLM training).

// ── Helpers: mean, variance, histogram ───────────────────────────

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

// Variance = average squared distance from the mean.
function variance(xs: number[]): number {
  const m = mean(xs);
  return xs.reduce((s, x) => s + (x - m) ** 2, 0) / xs.length;
}

// Text histogram: count samples per bin, draw bars with █.
function histogram(xs: number[], lo: number, hi: number, bins: number, width = 40): void {
  const counts = Array<number>(bins).fill(0);
  for (const x of xs) {
    const b = Math.min(bins - 1, Math.max(0, Math.floor(((x - lo) / (hi - lo)) * bins)));
    counts[b] = (counts[b] ?? 0) + 1;
  }
  const peak = Math.max(...counts);
  for (let b = 0; b < bins; b++) {
    const c = counts[b] ?? 0;
    const bar = "█".repeat(Math.round((c / peak) * width));
    const label = (lo + ((b + 0.5) * (hi - lo)) / bins).toFixed(2).padStart(6);
    console.log(`  ${label} │${bar}`);
  }
}

// ── 1. Sampling a categorical distribution ───────────────────────
// This is literally what an LLM does to pick its next token:
// walk the cumulative probabilities until you pass a uniform random r.
function sampleCategorical(probs: number[]): number {
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i] ?? 0;
    if (r < cum) return i;
  }
  return probs.length - 1; // guard against float round-off
}

console.log("=== 1. Categorical sampling (how an LLM picks a token) ===");
const tokenProbs = [0.5, 0.3, 0.15, 0.05];
const tokenNames = ["the", "a", "cat", "zygote"];
const counts = [0, 0, 0, 0];
const N1 = 100_000;
for (let i = 0; i < N1; i++) counts[sampleCategorical(tokenProbs)]!++;
console.log(`sampled ${N1} tokens from P = [0.5, 0.3, 0.15, 0.05]:`);
tokenNames.forEach((name, i) =>
  console.log(
    `  "${name}"`.padEnd(12) +
    `expected ${(tokenProbs[i]! * 100).toFixed(1)}%   got ${(((counts[i] ?? 0) / N1) * 100).toFixed(2)}%`
  )
);
console.log("→ empirical frequencies converge to the belief numbers.");

// ── 2. Box-Muller: Gaussians from uniform randoms ────────────────
// Computers only hand out uniforms. Box-Muller turns two uniforms
// into a perfect standard Gaussian: z = √(−2·ln u₁) · cos(2π u₂)
function gaussianSample(mu = 0, sigma = 1): number {
  let u1 = 0;
  while (u1 === 0) u1 = Math.random(); // avoid ln(0)
  const u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mu + sigma * z;
}

console.log("\n=== 2. Box-Muller Gaussian sampler — then VERIFY it ===");
const target = { mu: 5, sigma: 2 };
const gs: number[] = [];
for (let i = 0; i < 100_000; i++) gs.push(gaussianSample(target.mu, target.sigma));
const gm = mean(gs), gsd = Math.sqrt(variance(gs));
console.log(`target:    μ = ${target.mu}, σ = ${target.sigma}`);
console.log(`estimated: μ = ${gm.toFixed(4)}, σ = ${gsd.toFixed(4)}   (from 100k samples)`);
const within1 = gs.filter(x => Math.abs(x - target.mu) <= target.sigma).length / gs.length;
const within2 = gs.filter(x => Math.abs(x - target.mu) <= 2 * target.sigma).length / gs.length;
console.log(`within μ±σ:  ${(within1 * 100).toFixed(1)}%  (theory: 68.3%)`);
console.log(`within μ±2σ: ${(within2 * 100).toFixed(1)}%  (theory: 95.4%)`);
console.log("shape of the samples (should be a bell):");
histogram(gs, -1, 11, 12);

// ── 3. Central Limit Theorem with dice ───────────────────────────
// One die is FLAT. Averages of dice become a BELL. Watch it happen.
const rollDie = () => 1 + Math.floor(Math.random() * 6);
const avgOfDice = (n: number) => {
  let s = 0;
  for (let i = 0; i < n; i++) s += rollDie();
  return s / n;
};

console.log("\n=== 3. Central Limit Theorem: dice averages become Gaussian ===");
for (const n of [1, 2, 20]) {
  const samples: number[] = [];
  for (let i = 0; i < 50_000; i++) samples.push(avgOfDice(n));
  const sd = Math.sqrt(variance(samples));
  console.log(`\naverage of ${n} ${n === 1 ? "die" : "dice"}  (mean ${mean(samples).toFixed(3)}, σ ${sd.toFixed(3)}):`);
  // bin edges aligned to the possible outcomes so bars sit on real values
  if (n === 1) histogram(samples, 0.5, 6.5, 6, 34);
  else histogram(samples, 0.75, 6.25, 11, 34);
}
// One die has σ ≈ 1.708. CLT predicts the average of n has σ/√n.
console.log(`\nCLT prediction for σ of the average: σ/√n = 1.708/√20 = ${(1.708 / Math.sqrt(20)).toFixed(3)}`);
console.log("→ matches the measured σ above. Noise shrinks as √n —");
console.log("  the same law behind minibatch gradients and error bars.");

// ── 4. Bayes' rule: the medical test, by brute-force counting ────
console.log("\n=== 4. Bayes' rule: positive test ≠ probably sick ===");
const P_sick = 0.01;        // prior: 1% of people have the disease
const P_pos_sick = 0.95;    // sensitivity
const P_pos_healthy = 0.05; // false positive rate

// Formula:
const P_pos = P_pos_sick * P_sick + P_pos_healthy * (1 - P_sick);
const posterior = (P_pos_sick * P_sick) / P_pos;
console.log(`prior P(sick) = ${P_sick}, P(+|sick) = ${P_pos_sick}, P(+|healthy) = ${P_pos_healthy}`);
console.log(`Bayes: P(sick|+) = ${posterior.toFixed(4)}  ← only ~16%, not 95%!`);

// Simulation: generate a million people, count what Bayes counts.
let sickAndPos = 0, totalPos = 0;
for (let i = 0; i < 1_000_000; i++) {
  const sick = Math.random() < P_sick;
  const pos = Math.random() < (sick ? P_pos_sick : P_pos_healthy);
  if (pos) {
    totalPos++;
    if (sick) sickAndPos++;
  }
}
console.log(`simulation of 1M people: ${sickAndPos} sick among ${totalPos} positives → ${(sickAndPos / totalPos).toFixed(4)}`);
console.log("→ false positives from the huge healthy group swamp the true ones.");
console.log("  (Research moral: surprising results from unlikely methods are usually bugs.)");

// ── 5. Maximum likelihood: fitting a coin's bias ─────────────────
// Flip a coin with SECRET bias, then recover the bias from data alone
// by scanning which p makes the observed flips most probable.
console.log("\n=== 5. MLE: fit a coin bias (cross-entropy in miniature) ===");
const secretP = 0.7;
const flips: number[] = [];
for (let i = 0; i < 200; i++) flips.push(Math.random() < secretP ? 1 : 0);
const heads = flips.reduce((a, b) => a + b, 0);
console.log(`secret bias p = ${secretP}; observed ${heads} heads in ${flips.length} flips`);

// log-likelihood(p) = heads·log(p) + tails·log(1−p)
// (log turns the product of 200 tiny probabilities into a safe sum)
const logLik = (p: number) => heads * Math.log(p) + (flips.length - heads) * Math.log(1 - p);

console.log("\n  p      log-likelihood");
let bestP = 0, bestLL = -Infinity;
for (let p = 0.05; p < 0.999; p += 0.05) {
  const ll = logLik(p);
  if (ll > bestLL) { bestLL = ll; bestP = p; }
  const bar = "█".repeat(Math.max(0, Math.round(40 + ll / 5)));
  console.log(`  ${p.toFixed(2)}  ${ll.toFixed(2).padStart(9)} ${bar}`);
}
console.log(`\nMLE estimate: p ≈ ${bestP.toFixed(2)}   closed form (heads/flips): ${(heads / flips.length).toFixed(3)}`);
console.log("→ maximizing log-likelihood = minimizing −log P(data) = cross-entropy.");
console.log("  Training an LLM is EXACTLY this, for a 50,000-sided coin at every token.");

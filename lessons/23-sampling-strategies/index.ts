// Lesson 23 — Sampling Strategies: Temperature, Top-K, Top-P
// No external imports — pure TypeScript/JavaScript

// ---------------------------------------------------------------------------
// Math utilities
// ---------------------------------------------------------------------------

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - max)); // subtract max for numerical stability
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function applyTemperature(logits: number[], temperature: number): number[] {
  if (temperature <= 0) throw new Error("Temperature must be > 0");
  return logits.map((x) => x / temperature);
}

// Seeded PRNG (mulberry32) so we can reproduce runs
function makePrng(seed: number): () => number {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Sample one index from a probability distribution using a PRNG
function sampleFromProbs(probs: number[], rand: () => number): number {
  const r = rand();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (r < cumulative) return i;
  }
  return probs.length - 1; // fallback for floating point rounding
}

// ---------------------------------------------------------------------------
// 1. Greedy sampling — always returns argmax
// ---------------------------------------------------------------------------

function greedySample(logits: number[]): number {
  let best = 0;
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > logits[best]) best = i;
  }
  return best;
}

// ---------------------------------------------------------------------------
// 2. Temperature sampling — scale logits, softmax, then sample
// ---------------------------------------------------------------------------

function temperatureSample(
  logits: number[],
  temperature: number,
  rand: () => number
): number {
  const scaled = applyTemperature(logits, temperature);
  const probs = softmax(scaled);
  return sampleFromProbs(probs, rand);
}

// ---------------------------------------------------------------------------
// 3. Top-K sampling — keep only top K tokens, then temperature + softmax
// ---------------------------------------------------------------------------

function topKSample(
  logits: number[],
  k: number,
  temperature: number,
  rand: () => number
): number {
  const clampedK = Math.min(k, logits.length);

  // Get sorted indices (descending by logit value)
  const indices = logits
    .map((v, i) => ({ v, i }))
    .sort((a, b) => b.v - a.v)
    .map((x) => x.i);

  // Build masked logits: -Infinity for tokens outside top-K
  const masked = new Array(logits.length).fill(-Infinity);
  for (let j = 0; j < clampedK; j++) {
    masked[indices[j]] = logits[indices[j]];
  }

  const scaled = applyTemperature(masked, temperature);
  const probs = softmax(scaled);
  return sampleFromProbs(probs, rand);
}

// ---------------------------------------------------------------------------
// 4. Top-P (Nucleus) sampling
// ---------------------------------------------------------------------------

function topPSample(
  logits: number[],
  p: number,
  temperature: number,
  rand: () => number
): number {
  // Step 1: apply temperature
  const scaled = applyTemperature(logits, temperature);

  // Step 2: compute full softmax probabilities
  const probs = softmax(scaled);

  // Step 3: sort indices by probability descending
  const sortedIndices = probs
    .map((prob, idx) => ({ prob, idx }))
    .sort((a, b) => b.prob - a.prob);

  // Step 4: find the nucleus — smallest set with cumulative prob >= p
  let cumulative = 0;
  const nucleusIndices: number[] = [];
  for (const { prob, idx } of sortedIndices) {
    nucleusIndices.push(idx);
    cumulative += prob;
    if (cumulative >= p) break;
  }

  // Step 5: build masked logits with only nucleus tokens
  const masked = new Array(logits.length).fill(-Infinity);
  for (const idx of nucleusIndices) {
    masked[idx] = scaled[idx];
  }

  // Step 6: renormalize and sample
  const nucleusProbs = softmax(masked);
  return sampleFromProbs(nucleusProbs, rand);
}

// ---------------------------------------------------------------------------
// Tiny vocabulary for demonstration
// ---------------------------------------------------------------------------

const VOCAB = [
  // Common words (high baseline probability in many contexts)
  "the", "a", "is", "in", "of", "and", "to", "it", "was", "that",
  // Nouns
  "cat", "dog", "tree", "sky", "river", "mountain", "city", "house", "book", "light",
  // Verbs
  "ran", "saw", "found", "built", "grew", "fell", "rose", "moved", "spoke", "walked",
  // Adjectives / modifiers
  "old", "new", "small", "bright", "dark", "quiet", "wild", "cold", "warm", "deep",
  // Rare / unusual tokens that should rarely appear
  "zephyr", "quasar", "fjord", "oblong", "vex", "quirk", "glitch", "xenon", "warp", "prism",
];

// Simulate realistic logit distributions for a GPT
// We model a scenario where the model has moderate confidence:
// a few tokens are clearly good, most are mediocre, rare tokens are bad.
function makeLogits(seed: number): number[] {
  const rng = makePrng(seed);
  return VOCAB.map((_, i) => {
    const base = i < 10 ? 2.5 - i * 0.3 : // common words: high logits
                 i < 20 ? 0.5 + rng() * 0.5 : // nouns: medium
                 i < 30 ? 0.0 + rng() * 0.5 : // verbs: medium-low
                 i < 40 ? -0.5 + rng() * 0.5 : // adjectives: low
                         -2.0 - rng() * 1.0;  // rare tokens: very low
    return base + (rng() - 0.5) * 0.3; // small noise
  });
}

// Generate N tokens using a given sampling function
function generateTokens(
  n: number,
  sampler: (logits: number[], rand: () => number) => number,
  seed: number
): string[] {
  const tokens: string[] = [];
  const rand = makePrng(seed);
  for (let step = 0; step < n; step++) {
    // In a real GPT, logits would depend on previous tokens.
    // Here we use step-varying logits to simulate context changes.
    const logits = makeLogits(seed * 1000 + step);
    const idx = sampler(logits, rand);
    tokens.push(VOCAB[idx]);
  }
  return tokens;
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

function computeStats(tokens: string[]): {
  uniqueCount: number;
  totalCount: number;
  repetitionRate: number;
  topTokens: Array<[string, number]>;
} {
  const counts = new Map<string, number>();
  for (const t of tokens) counts.set(t, (counts.get(t) ?? 0) + 1);

  const uniqueCount = counts.size;
  const totalCount = tokens.length;

  // Repetition rate: proportion of consecutive pairs that are identical
  let repeated = 0;
  for (let i = 1; i < tokens.length; i++) {
    if (tokens[i] === tokens[i - 1]) repeated++;
  }
  const repetitionRate = repeated / (tokens.length - 1);

  const topTokens = [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  return { uniqueCount, totalCount, repetitionRate, topTokens };
}

// ---------------------------------------------------------------------------
// ASCII bar chart: visualize which tokens are in/out of the nucleus/top-K
// ---------------------------------------------------------------------------

function visualizeDistribution(
  logits: number[],
  temperature: number,
  k: number,
  p: number
): void {
  console.log("\n--- Distribution Visualization (seed=42 logits, top 15 tokens) ---");
  console.log(`Temperature: ${temperature}  |  Top-K: ${k}  |  Top-P: ${p}`);
  console.log("");

  const scaled = applyTemperature(logits, temperature);
  const probs = softmax(scaled);

  // Sort by probability descending, show top 15
  const sorted = probs
    .map((prob, idx) => ({ prob, idx, token: VOCAB[idx] }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 15);

  // Determine which tokens are in top-K
  const topKIndices = new Set(sorted.slice(0, k).map((x) => x.idx));

  // Determine nucleus for top-P
  let cumulative = 0;
  const nucleusIndices = new Set<number>();
  for (const { prob, idx } of sorted) {
    nucleusIndices.add(idx);
    cumulative += prob;
    if (cumulative >= p) break;
  }

  // Cumulative for display
  let runningCumulative = 0;

  const barMax = 40;
  const headerToken = "Token".padEnd(10);
  const headerProb = "Prob  ";
  const headerBar = "Bar".padEnd(barMax + 2);
  const headerCumul = "Cumul  ";
  const headerK = "K";
  const headerP = "P";
  console.log(`${headerToken} ${headerProb} ${headerBar} ${headerCumul} ${headerK} ${headerP}`);
  console.log("-".repeat(75));

  for (const { prob, idx, token } of sorted) {
    runningCumulative += prob;
    const barLen = Math.round(prob * barMax * 10); // scale up for visibility
    const bar = "█".repeat(Math.min(barLen, barMax));
    const inK = topKIndices.has(idx) ? "Y" : " ";
    const inP = nucleusIndices.has(idx) ? "Y" : " ";
    const probStr = (prob * 100).toFixed(1).padStart(5) + "%";
    const cumulStr = (runningCumulative * 100).toFixed(1).padStart(5) + "%";
    console.log(
      `${token.padEnd(10)} ${probStr} |${bar.padEnd(barMax)}| ${cumulStr}  ${inK} ${inP}`
    );
  }

  console.log("-".repeat(75));
  console.log(`K=in Top-K, P=in Nucleus (top-p=${p})`);
}

// ---------------------------------------------------------------------------
// Benchmark: run each strategy multiple times, measure variance
// ---------------------------------------------------------------------------

function benchmark(nRuns: number = 10, nTokens: number = 20): void {
  console.log(`\n=== Benchmark: ${nRuns} runs x ${nTokens} tokens each ===\n`);

  interface BenchResult {
    name: string;
    uniqueCounts: number[];
    repRates: number[];
  }

  const strategies: Array<{
    name: string;
    sampler: (logits: number[], rand: () => number) => number;
  }> = [
    {
      name: "Greedy",
      sampler: (logits, _rand) => greedySample(logits),
    },
    {
      name: "Temp=0.7",
      sampler: (logits, rand) => temperatureSample(logits, 0.7, rand),
    },
    {
      name: "Top-K=10,T=0.7",
      sampler: (logits, rand) => topKSample(logits, 10, 0.7, rand),
    },
    {
      name: "Top-P=0.9,T=0.7",
      sampler: (logits, rand) => topPSample(logits, 0.9, 0.7, rand),
    },
  ];

  const results: BenchResult[] = strategies.map((s) => ({
    name: s.name,
    uniqueCounts: [],
    repRates: [],
  }));

  for (let run = 0; run < nRuns; run++) {
    const seed = run * 7 + 13; // different seed each run
    for (let si = 0; si < strategies.length; si++) {
      const tokens = generateTokens(nTokens, strategies[si].sampler, seed);
      const stats = computeStats(tokens);
      results[si].uniqueCounts.push(stats.uniqueCount);
      results[si].repRates.push(stats.repetitionRate);
    }
  }

  const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
  const stddev = (arr: number[]) => {
    const m = mean(arr);
    return Math.sqrt(arr.reduce((acc, x) => acc + (x - m) ** 2, 0) / arr.length);
  };

  const nameW = 18;
  const colW = 12;
  console.log(
    "Strategy".padEnd(nameW) +
    "Unique(mean)".padEnd(colW) +
    "Unique(std)".padEnd(colW) +
    "RepRate(mean)".padEnd(colW) +
    "RepRate(std)".padEnd(colW)
  );
  console.log("-".repeat(nameW + colW * 4));

  for (const r of results) {
    const uMean = mean(r.uniqueCounts).toFixed(1).padEnd(colW);
    const uStd = stddev(r.uniqueCounts).toFixed(2).padEnd(colW);
    const rMean = (mean(r.repRates) * 100).toFixed(1).padEnd(colW - 1) + "%";
    const rStd = (stddev(r.repRates) * 100).toFixed(2).padEnd(colW - 1) + "%";
    console.log(`${r.name.padEnd(nameW)}${uMean}${uStd}${rMean} ${rStd}`);
  }
  console.log("\nHigher unique count = more diverse output.");
  console.log("Lower repetition rate = fewer immediate repeats.");
}

// ---------------------------------------------------------------------------
// Main demo
// ---------------------------------------------------------------------------

function main(): void {
  const N_TOKENS = 50;
  const DEMO_SEED = 42;

  console.log("================================================================");
  console.log("Lesson 23 — Sampling Strategies Demo");
  console.log("================================================================");
  console.log(`Vocabulary size: ${VOCAB.length} tokens`);
  console.log(`Generating ${N_TOKENS} tokens with each strategy (seed=${DEMO_SEED})`);
  console.log("");

  // Define strategies
  const strategies: Array<{
    name: string;
    description: string;
    sampler: (logits: number[], rand: () => number) => number;
  }> = [
    {
      name: "Greedy",
      description: "Always pick argmax — no randomness",
      sampler: (logits, _rand) => greedySample(logits),
    },
    {
      name: "Temperature T=0.2",
      description: "Very focused — near-greedy but slightly random",
      sampler: (logits, rand) => temperatureSample(logits, 0.2, rand),
    },
    {
      name: "Temperature T=0.7",
      description: "Balanced — standard creative setting",
      sampler: (logits, rand) => temperatureSample(logits, 0.7, rand),
    },
    {
      name: "Temperature T=1.5",
      description: "Very flat distribution — high randomness",
      sampler: (logits, rand) => temperatureSample(logits, 1.5, rand),
    },
    {
      name: "Top-K=5, T=0.7",
      description: "Only top 5 candidates allowed",
      sampler: (logits, rand) => topKSample(logits, 5, 0.7, rand),
    },
    {
      name: "Top-K=20, T=0.7",
      description: "Top 20 candidates — more variety",
      sampler: (logits, rand) => topKSample(logits, 20, 0.7, rand),
    },
    {
      name: "Top-P=0.5, T=0.7",
      description: "Nucleus covering 50% — tight",
      sampler: (logits, rand) => topPSample(logits, 0.5, 0.7, rand),
    },
    {
      name: "Top-P=0.9, T=0.7",
      description: "Nucleus covering 90% — standard production setting",
      sampler: (logits, rand) => topPSample(logits, 0.9, 0.7, rand),
    },
    {
      name: "Top-P=0.95, T=0.7",
      description: "Nucleus covering 95% — slightly more expansive",
      sampler: (logits, rand) => topPSample(logits, 0.95, 0.7, rand),
    },
  ];

  // Generate and print output for each strategy
  for (const s of strategies) {
    const tokens = generateTokens(N_TOKENS, s.sampler, DEMO_SEED);
    const stats = computeStats(tokens);

    console.log(`--- ${s.name} ---`);
    console.log(`    ${s.description}`);
    console.log(`    Output: ${tokens.join(" ")}`);
    console.log(`    Unique tokens: ${stats.uniqueCount}/${VOCAB.length}`);
    console.log(`    Repetition rate: ${(stats.repetitionRate * 100).toFixed(1)}%`);
    const topStr = stats.topTokens.map(([t, c]) => `"${t}"x${c}`).join(", ");
    console.log(`    Most common: ${topStr}`);
    console.log("");
  }

  // ---------------------------------------------------------------------------
  // Summary comparison table
  // ---------------------------------------------------------------------------

  console.log("================================================================");
  console.log("Summary Table");
  console.log("================================================================");

  const nameW = 22;
  const colW = 10;
  console.log(
    "Strategy".padEnd(nameW) +
    "Unique".padEnd(colW) +
    "RepRate".padEnd(colW) +
    "Character"
  );
  console.log("-".repeat(nameW + colW * 2 + 25));

  const characters = [
    "Deterministic loops",
    "Near-greedy, stable",
    "Balanced quality",
    "Chaotic, noisy",
    "Precise, constrained",
    "Good variety",
    "Tight, conservative",
    "Standard production",
    "Slightly expansive",
  ];

  for (let si = 0; si < strategies.length; si++) {
    const tokens = generateTokens(N_TOKENS, strategies[si].sampler, DEMO_SEED);
    const stats = computeStats(tokens);
    console.log(
      strategies[si].name.padEnd(nameW) +
      String(stats.uniqueCount).padEnd(colW) +
      ((stats.repetitionRate * 100).toFixed(1) + "%").padEnd(colW) +
      characters[si]
    );
  }

  // ---------------------------------------------------------------------------
  // Visualize logit distribution under different strategies
  // ---------------------------------------------------------------------------

  console.log("\n================================================================");
  console.log("Distribution Visualization");
  console.log("================================================================");

  const exampleLogits = makeLogits(DEMO_SEED);

  visualizeDistribution(exampleLogits, 1.0, 10, 0.9);
  visualizeDistribution(exampleLogits, 0.5, 10, 0.9);
  visualizeDistribution(exampleLogits, 2.0, 10, 0.9);

  // ---------------------------------------------------------------------------
  // Nucleus size analysis — how many tokens end up in nucleus across contexts
  // ---------------------------------------------------------------------------

  console.log("\n================================================================");
  console.log("Nucleus Size Analysis (Top-P=0.9, T=0.7) across 20 contexts");
  console.log("================================================================");
  console.log("Demonstrates that top-p adapts to the model's confidence level.\n");

  const nucleusSizes: number[] = [];
  for (let step = 0; step < 20; step++) {
    const logits = makeLogits(step * 17 + 3);
    const scaled = applyTemperature(logits, 0.7);
    const probs = softmax(scaled);
    const sorted = [...probs].map((p2, i) => ({ p: p2, i })).sort((a, b) => b.p - a.p);
    let cum = 0;
    let size = 0;
    for (const { p: prob } of sorted) {
      size++;
      cum += prob;
      if (cum >= 0.9) break;
    }
    nucleusSizes.push(size);
  }

  for (let i = 0; i < nucleusSizes.length; i++) {
    const size = nucleusSizes[i];
    const bar = "█".repeat(size);
    console.log(`Context ${String(i + 1).padStart(2)}: nucleus size = ${String(size).padStart(2)} ${bar}`);
  }

  const avgNucleus = nucleusSizes.reduce((a, b) => a + b, 0) / nucleusSizes.length;
  console.log(`\nAverage nucleus size: ${avgNucleus.toFixed(1)} tokens`);
  console.log("Min:", Math.min(...nucleusSizes), " Max:", Math.max(...nucleusSizes));

  // ---------------------------------------------------------------------------
  // Benchmark
  // ---------------------------------------------------------------------------

  benchmark(10, 30);

  // ---------------------------------------------------------------------------
  // Key takeaways
  // ---------------------------------------------------------------------------

  console.log("\n================================================================");
  console.log("Key Takeaways");
  console.log("================================================================");
  console.log("1. Greedy decoding produces loops and very low diversity.");
  console.log("2. Temperature T<1 sharpens the distribution (fewer unique tokens).");
  console.log("3. Temperature T>1 flattens it (more unique but possibly incoherent).");
  console.log("4. Top-K=5 is too restrictive; K=20 gives more natural variety.");
  console.log("5. Top-P adapts the nucleus size to the model's confidence.");
  console.log("6. Top-P=0.9 + T=0.7 is the standard production combination.");
  console.log("7. No universal best setting — tune for your specific task.");
  console.log("================================================================\n");
}

main();

// Baselines & ablations: build a 3-component text classifier, then find out
// which components actually earn their keep.
//
// Task: sentiment (1 = positive, 0 = negative) on a tiny embedded dataset.
// The model is logistic regression over three "components":
//   1. char-bigram features   — which two-letter chunks appear in the text
//   2. a length feature       — how long the text is
//   3. case normalization     — lowercase everything before featurizing
//
// We train the FULL model, then retrain with each component removed (the
// ablation), and print the whole table next to the majority-class baseline.
// Spoiler the table will confirm: one of the three components barely matters.

// ── Seeded RNG for reproducible init ──
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

// ── Embedded toy dataset ──
// Imbalanced on purpose (more negatives), and with realistic messy casing:
// reviewers on the internet SHOUT sometimes.
const trainData: Array<[string, number]> = [
  ["i love this movie", 1],
  ["what a great film", 1],
  ["really wonderful acting", 1],
  ["great story and great cast", 1],
  ["i enjoyed every minute", 1],
  ["a wonderful little gem", 1],
  ["lovely and moving", 1],
  ["great fun for everyone", 1],
  ["terrible plot and bad acting", 0],
  ["i hated this film", 0],
  ["what a boring mess", 0],
  ["awful from start to finish", 0],
  ["bad script bad pacing", 0],
  ["truly terrible movie", 0],
  ["boring and predictable", 0],
  ["i hated every minute", 0],
  ["an awful waste of time", 0],
  ["dull boring and bad", 0],
  ["the worst film this year", 0],
  ["bad acting ruined it", 0],
  ["a terrible boring slog", 0],
  ["worst script i have seen", 0],
];
const testData: Array<[string, number]> = [
  // Plain lowercase items — easy for every variant that has bigrams.
  ["a great and lovely film", 1],
  ["i love the wonderful story", 1],
  ["boring plot terrible cast", 0],
  ["i hated the script", 0],
  ["the worst mess this year", 0],
  ["bad pacing and a bad ending", 0],
  // SHOUTY items: the sentiment word is capitalized, and the lowercase
  // filler words lean the WRONG way (they co-occur with the other class
  // in training). Only a model that lowercases can read these correctly.
  ["truly a GREAT film", 1], //          "truly" seen in "truly terrible movie"
  ["a WONDERFUL script indeed", 1], //   "script" seen only in negatives
  ["a BORING story and cast", 0], //     "story and cast" seen in a positive
  ["the acting is TERRIBLE really", 0], // "really ... acting" seen in a positive
  ["DULL film for everyone", 0], //      "for everyone" seen in a positive
  ["every minute is AWFUL", 0], //       "every minute" appears in both classes
];

// ── The three components ──
// Component 3: case normalization. Ablated = featurize the raw string,
// so "AWFUL" and "awful" produce completely different bigrams.
function normalize(text: string, useNorm: boolean): string {
  return useNorm ? text.toLowerCase() : text;
}

// Component 1: char bigrams, hashed into a fixed number of buckets.
const BIGRAM_DIM = 256;
function bigramFeatures(text: string): number[] {
  const f = Array<number>(BIGRAM_DIM).fill(0);
  for (let i = 0; i < text.length - 1; i++) {
    const code = text.charCodeAt(i) * 31 + text.charCodeAt(i + 1);
    f[code % BIGRAM_DIM] = 1; // presence, not count
  }
  return f;
}

// Component 2: normalized length. (Rigged truth: how LONG a review is says
// nothing about whether it is positive — we'll let the ablation expose that.)
function lengthFeature(text: string): number {
  return text.length / 30;
}

interface Components {
  bigrams: boolean;
  length: boolean;
  norm: boolean;
}

function featurize(raw: string, use: Components): number[] {
  const text = normalize(raw, use.norm);
  const f: number[] = [];
  if (use.bigrams) f.push(...bigramFeatures(text));
  if (use.length) f.push(lengthFeature(text));
  return f;
}

// ── Logistic regression, retrained from scratch for every variant ──
function trainVariant(name: string, use: Components): { name: string; acc: number } {
  const rand = mulberry32(42);
  const dim = featurize(trainData[0]?.[0] ?? "", use).length;
  const w = Array.from({ length: dim }, () => (rand() - 0.5) * 0.01);
  let b = 0;

  const lr = 0.5;
  for (let epoch = 0; epoch < 150; epoch++) {
    for (const [text, y] of trainData) {
      const x = featurize(text, use);
      let z = b;
      for (let j = 0; j < dim; j++) z += (w[j] ?? 0) * (x[j] ?? 0);
      const p = 1 / (1 + Math.exp(-z));
      const g = p - y; // gradient of cross-entropy wrt z
      for (let j = 0; j < dim; j++) w[j] = (w[j] ?? 0) - lr * g * (x[j] ?? 0);
      b -= lr * g;
    }
  }

  let correct = 0;
  for (const [text, y] of testData) {
    const x = featurize(text, use);
    let z = b;
    for (let j = 0; j < dim; j++) z += (w[j] ?? 0) * (x[j] ?? 0);
    if ((z > 0 ? 1 : 0) === y) correct++;
  }
  return { name, acc: correct / testData.length };
}

// ── Baseline: majority class (no learning at all) ──
const negTrain = trainData.filter(([, y]) => y === 0).length;
const majorityLabel = negTrain > trainData.length / 2 ? 0 : 1;
const majorityAcc =
  testData.filter(([, y]) => y === majorityLabel).length / testData.length;

// ── Run the ablation: full model, then remove ONE component per row ──
const rows = [
  trainVariant("full model (bigrams + length + norm)", { bigrams: true, length: true, norm: true }),
  trainVariant("  − bigram features", { bigrams: false, length: true, norm: true }),
  trainVariant("  − length feature", { bigrams: true, length: false, norm: true }),
  trainVariant("  − case normalization", { bigrams: true, length: true, norm: false }),
];
const full = rows[0]?.acc ?? 0;

console.log(`=== Ablation table (test accuracy, ${testData.length} held-out examples) ===\n`);
const pct = (v: number) => (v * 100).toFixed(1).padStart(5) + "%";
console.log("variant                                    acc      Δ vs full");
console.log("-----------------------------------------  -------  ---------");
for (const r of rows) {
  const delta = r.acc - full;
  const d = r.acc === full ? "   —" : (delta > 0 ? "+" : "") + (100 * delta).toFixed(1);
  console.log(`${r.name.padEnd(43)}${pct(r.acc)}   ${d}`);
}
console.log(
  `${"majority class (always predict 'negative')".padEnd(43)}${pct(majorityAcc)}   ${(100 * (majorityAcc - full)).toFixed(1)}`
);

console.log("\n=== How to read this table ===");
console.log("• − bigrams: accuracy collapses to the majority baseline. The words");
console.log("  ARE the model; this component does all the heavy lifting.");
console.log("• − length: nothing changes. This component is decoration. If it");
console.log("  cost compute we would delete it — and a paper whose headline");
console.log("  contribution was the length feature would be selling noise.");
console.log("• − normalization: a real but smaller drop. Without lowercasing,");
console.log("  'AWFUL' shares no bigrams with the 'awful' seen in training,");
console.log("  so shouty test reviews go unrecognized.");
console.log("• majority class: the number that gives every other number meaning.");
console.log(`  'We got ${(100 * full).toFixed(0)}%' impresses only because a rock gets ${(100 * majorityAcc).toFixed(0)}%.`);

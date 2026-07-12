// Baselines & ablations: build a 3-component text classifier, then find out
// which components actually earn their keep.
//
// Task: sentiment (1 = positive, 0 = negative) on a tiny embedded dataset.
// The model is logistic regression over three "components":
//   1. char-bigram features   — which two-letter chunks of training text appear
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
// Imbalanced on purpose (more negatives), all-lowercase training text.
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
// Test set: real reviewers SHOUT. In the shouty items the sentiment word is
// capitalized and the lowercase filler words lean the WRONG way (they co-occur
// with the other class in training). Only a variant that lowercases first can
// read those items correctly.
const testData: Array<[string, number]> = [
  // plain lowercase items — easy for every variant that has bigram features
  ["a great and lovely film", 1],
  ["i love the wonderful story", 1],
  ["boring plot terrible cast", 0],
  ["i hated the script", 0],
  ["the worst mess this year", 0],
  ["bad pacing and a bad ending", 0],
  // shouty items
  ["truly a GREAT film", 1], //            "truly" seen in "truly terrible movie"
  ["a WONDERFUL story indeed", 1], //      "indeed" unseen; bare "story" is weak
  ["a BORING story and cast", 0], //       "story and ... cast" seen in a positive
  ["the acting is TERRIBLE really", 0], // "really ... acting" seen in a positive
  ["a DULL waste for everyone", 0], //     "for everyone" seen in a positive
  ["every minute is AWFUL", 0], //         "every minute" appears in both classes
];

interface Components {
  bigrams: boolean; // component 1
  length: boolean; // component 2
  norm: boolean; // component 3
}

function bigrams(text: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length - 1; i++) out.push(text.slice(i, i + 2));
  return out;
}

// ── One full train+eval per variant: exactly one component differs per row ──
function trainVariant(name: string, use: Components): { name: string; acc: number } {
  // Component 3: case normalization (ablated = featurize the raw string,
  // so "AWFUL" shares no bigrams with the "awful" seen in training).
  const nz = (s: string) => (use.norm ? s.toLowerCase() : s);

  // Component 1: a bigram vocabulary built from the (maybe-normalized) train
  // text. A bigram never seen in training has no feature — it contributes 0.
  const vocab = new Map<string, number>();
  if (use.bigrams)
    for (const [text] of trainData)
      for (const bg of bigrams(nz(text)))
        if (!vocab.has(bg)) vocab.set(bg, vocab.size);

  const dim = vocab.size + (use.length ? 1 : 0);
  const featurize = (raw: string): number[] => {
    const f = Array<number>(dim).fill(0);
    for (const bg of bigrams(nz(raw))) {
      const j = vocab.get(bg);
      if (j !== undefined) f[j] = 1; // presence, not count
    }
    // Component 2: normalized length. (Rigged truth: how LONG a review is
    // says nothing about sentiment — the ablation will expose that.)
    if (use.length) f[dim - 1] = raw.length / 30;
    return f;
  };

  // Logistic regression trained from scratch with plain SGD.
  const rand = mulberry32(42);
  const w = Array.from({ length: dim }, () => (rand() - 0.5) * 0.01);
  let b = 0;
  const lr = 0.5;
  for (let epoch = 0; epoch < 150; epoch++) {
    for (const [text, y] of trainData) {
      const x = featurize(text);
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
    const x = featurize(text);
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
console.log("  'AWFUL' shares no bigrams with the 'awful' seen in training, so");
console.log("  shouty reviews are read only through their misleading filler words.");
console.log("• majority class: the number that gives every other number meaning.");
console.log(`  'We got ${(100 * full).toFixed(0)}%' impresses only because a rock gets ${(100 * majorityAcc).toFixed(0)}%.`);

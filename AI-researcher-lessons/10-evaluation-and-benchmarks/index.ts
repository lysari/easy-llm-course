// Evaluation, hands-on:
//   PART 1 — precision / recall / F1 from a confusion matrix
//   PART 2 — accuracy lying on a 95/5 imbalanced dataset
//   PART 3 — perplexity: score two toy language models on held-out text
//
// Run: npx ts-node index.ts

// ════════════════════════════════════════════════════════════════════
// PART 1 — The confusion matrix and its metrics
// ════════════════════════════════════════════════════════════════════

interface Confusion { tp: number; fp: number; fn: number; tn: number }

// Count the four outcomes by comparing predictions to truth (1 = positive)
function confusion(truth: number[], pred: number[]): Confusion {
  const c: Confusion = { tp: 0, fp: 0, fn: 0, tn: 0 };
  for (let i = 0; i < truth.length; i++) {
    const t = truth[i] ?? 0, p = pred[i] ?? 0;
    if (t === 1 && p === 1) c.tp++;
    else if (t === 0 && p === 1) c.fp++;
    else if (t === 1 && p === 0) c.fn++;
    else c.tn++;
  }
  return c;
}

function metrics(c: Confusion) {
  const total = c.tp + c.fp + c.fn + c.tn;
  const accuracy = total === 0 ? 0 : (c.tp + c.tn) / total;
  const precision = c.tp + c.fp === 0 ? 0 : c.tp / (c.tp + c.fp);
  const recall = c.tp + c.fn === 0 ? 0 : c.tp / (c.tp + c.fn);
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return { accuracy, precision, recall, f1 };
}

const pct = (v: number) => (v * 100).toFixed(1).padStart(5) + "%";

console.log("=== PART 1: confusion matrix → precision, recall, F1 ===\n");
// A small hand-checkable example: 10 samples
const truth1 = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0];
const pred1 = [1, 1, 0, 1, 0, 1, 0, 0, 0, 0];
const c1 = confusion(truth1, pred1);
console.log(`  truth: [${truth1.join(", ")}]`);
console.log(`  pred : [${pred1.join(", ")}]\n`);
console.log(`                    actually 1   actually 0`);
console.log(`  predicted 1        TP = ${c1.tp}       FP = ${c1.fp}`);
console.log(`  predicted 0        FN = ${c1.fn}       TN = ${c1.tn}\n`);
const m1 = metrics(c1);
console.log(`  accuracy  = (TP+TN)/all      = (${c1.tp}+${c1.tn})/10 = ${pct(m1.accuracy)}`);
console.log(`  precision = TP/(TP+FP)       = ${c1.tp}/${c1.tp + c1.fp}      = ${pct(m1.precision)}  (when it says 1, is it right?)`);
console.log(`  recall    = TP/(TP+FN)       = ${c1.tp}/${c1.tp + c1.fn}      = ${pct(m1.recall)}  (of the real 1s, how many caught?)`);
console.log(`  F1        = harmonic mean    = ${pct(m1.f1)}`);

// ════════════════════════════════════════════════════════════════════
// PART 2 — Accuracy lies on imbalanced data (95% ham / 5% spam)
// ════════════════════════════════════════════════════════════════════
console.log("\n=== PART 2: 1000 emails — 950 ham, 50 spam (95/5 imbalance) ===\n");

// Build the dataset: first 50 are spam (1), the rest ham (0)
const N = 1000, SPAM = 50;
const truth2 = Array.from({ length: N }, (_, i) => (i < SPAM ? 1 : 0));

// Classifier A: "the do-nothing" — predicts ham for EVERYTHING
const predA = Array<number>(N).fill(0);

// Classifier B: a genuinely useful spam filter —
// catches 45 of the 50 spams, but raises 80 false alarms among the 950 hams
const predB = truth2.map((t, i) => {
  if (t === 1) return i < 45 ? 1 : 0;          // catches 45/50 spam
  return i < SPAM + 80 ? 1 : 0;                 // 80 hams wrongly flagged
});

const mA = metrics(confusion(truth2, predA));
const mB = metrics(confusion(truth2, predB));

console.log("  classifier            | accuracy | precision | recall | F1");
console.log("  ----------------------|----------|-----------|--------|-------");
console.log(`  A: always says "ham"  |  ${pct(mA.accuracy)}  |  ${pct(mA.precision)}   | ${pct(mA.recall)} | ${pct(mA.f1)}`);
console.log(`  B: real spam filter   |  ${pct(mB.accuracy)}  |  ${pct(mB.precision)}   | ${pct(mB.recall)} | ${pct(mB.f1)}`);
console.log(`\n  Accuracy says the DO-NOTHING classifier wins (${pct(mA.accuracy)} vs ${pct(mB.accuracy)}).`);
console.log("  It catches ZERO spam. Recall and F1 expose it immediately.");
console.log("  On imbalanced data, accuracy mostly measures the imbalance.");

// ════════════════════════════════════════════════════════════════════
// PART 3 — Perplexity: which toy language model is better?
//
// Model: character bigrams. P(next char | current char) estimated from a
// training corpus by counting, with add-1 (Laplace) smoothing so no
// probability is ever exactly 0.
// ════════════════════════════════════════════════════════════════════
console.log("\n=== PART 3: perplexity of two toy models on held-out text ===\n");

const VOCAB = "abcdefghijklmnopqrstuvwxyz ".split(""); // 26 letters + space
const V = VOCAB.length;

type Bigram = Map<string, Map<string, number>>;

function trainBigram(corpus: string): Bigram {
  const counts: Bigram = new Map();
  const text = corpus.toLowerCase().replace(/[^a-z ]/g, "");
  for (let i = 0; i + 1 < text.length; i++) {
    const a = text[i] ?? " ", b = text[i + 1] ?? " ";
    if (!counts.has(a)) counts.set(a, new Map());
    const row = counts.get(a)!;
    row.set(b, (row.get(b) ?? 0) + 1);
  }
  return counts;
}

// P(b | a) with add-1 smoothing: (count(a,b) + 1) / (count(a,·) + V)
function prob(model: Bigram, a: string, b: string): number {
  const row = model.get(a);
  const rowTotal = row ? Array.from(row.values()).reduce((s, v) => s + v, 0) : 0;
  const c = row?.get(b) ?? 0;
  return (c + 1) / (rowTotal + V);
}

// Perplexity = e^(average negative log-likelihood per character)
function perplexity(model: Bigram, text: string): { ppl: number; nll: number } {
  let nll = 0, n = 0;
  for (let i = 0; i + 1 < text.length; i++) {
    nll += -Math.log(prob(model, text[i] ?? " ", text[i + 1] ?? " "));
    n++;
  }
  const avg = nll / n;
  return { ppl: Math.exp(avg), nll: avg };
}

// Two models trained on different corpora
const corpusA = // English-like: same style as the held-out text
  "the cat sat on the mat and the dog sat on the log " +
  "a cat and a rat sat in the hat the fat cat ate the rat " +
  "the dog and the cat ran to the mat and sat";
const corpusB = // valid text, but a completely different distribution
  "zyx qzv wqj kxz vzq jxw qkz zvx wjq xzk zqw vkx " +
  "qqz xxv wwk zzj vvq kkx jjw qzx zvw xkq";

const modelA = trainBigram(corpusA);
const modelB = trainBigram(corpusB);

const heldOut = "the rat sat on the hat"; // NOT in either training corpus

const a = perplexity(modelA, heldOut);
const b = perplexity(modelB, heldOut);

console.log(`  held-out text: "${heldOut}"  (neither model has seen this string)\n`);
console.log("  model                    | avg surprise (nats/char) | perplexity");
console.log("  -------------------------|--------------------------|-----------");
console.log(`  A (English-like corpus)  |          ${a.nll.toFixed(4)}          |   ${a.ppl.toFixed(2)}`);
console.log(`  B (gibberish corpus)     |          ${b.nll.toFixed(4)}          |   ${b.ppl.toFixed(2)}`);
console.log(`  uniform guessing         |          ${Math.log(V).toFixed(4)}          |   ${V.toFixed(2)}\n`);
console.log(`  Model A is "as confused as a ${a.ppl.toFixed(1)}-way choice" per character;`);
console.log(`  model B is worse than uniform guessing (its habits actively mislead it).`);
console.log(`  → pick model A. Lower perplexity = better next-token prediction —`);
console.log("    the same number an LLM's pretraining loss curve is tracking.");

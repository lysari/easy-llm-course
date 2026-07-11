// Information theory from scratch: surprise, entropy, cross-entropy
// (the LLM loss), KL divergence (the RLHF leash), perplexity, and
// the compression = prediction identity.

// All logs base 2 → everything is measured in BITS.
const log2 = (x: number) => Math.log(x) / Math.LN2;

// Surprise (self-information) of one event with probability p.
const surprise = (p: number) => -log2(p);

// Entropy: average surprise under the true distribution P.
// H(P) = −Σ pᵢ log₂ pᵢ    (terms with p = 0 contribute 0)
function entropy(P: number[]): number {
  return P.reduce((h, p) => (p > 0 ? h - p * log2(p) : h), 0);
}

// Cross-entropy: average surprise of a Q-believer in a P-world.
// H(P,Q) = −Σ pᵢ log₂ qᵢ
function crossEntropy(P: number[], Q: number[]): number {
  return P.reduce((h, p, i) => {
    if (p === 0) return h;
    return h - p * log2(Q[i] ?? 0); // q=0 while p>0 → Infinity (deservedly)
  }, 0);
}

// KL divergence: the extra bits caused by believing Q instead of P.
// KL(P‖Q) = Σ pᵢ log₂(pᵢ/qᵢ) = H(P,Q) − H(P)
function klDivergence(P: number[], Q: number[]): number {
  return P.reduce((d, p, i) => (p > 0 ? d + p * log2(p / (Q[i] ?? 0)) : d), 0);
}

// ── 1. Surprise: rare = informative ──────────────────────────────
console.log("=== 1. Surprise = −log₂(p) ===");
for (const [event, p] of [
  ["the sun rose (p=1.0)      ", 1.0],
  ["fair coin heads (p=0.5)   ", 0.5],
  ["roll a 6 (p=1/6)          ", 1 / 6],
  ["your house on fire (p=1e-4)", 1e-4],
] as [string, number][]) {
  console.log(`  ${event} → ${surprise(p).toFixed(3)} bits`);
}
console.log("→ independent surprises ADD: two coin flips = " +
  `${surprise(0.5 * 0.5).toFixed(1)} bits = 1 + 1. That's why it's a log.`);

// ── 2. Entropy of real text's character distribution ─────────────
console.log("\n=== 2. Entropy of text (average surprise per character) ===");
const text =
  "the quick brown fox jumps over the lazy dog and then the dog " +
  "chases the fox back over the hill because that is what dogs do";

// Count character frequencies → empirical distribution
function charDistribution(s: string): { chars: string[]; probs: number[] } {
  const counts = new Map<string, number>();
  for (const ch of s) counts.set(ch, (counts.get(ch) ?? 0) + 1);
  const chars = [...counts.keys()].sort();
  const probs = chars.map(c => (counts.get(c) ?? 0) / s.length);
  return { chars, probs };
}

const { chars, probs } = charDistribution(text);
const Htext = entropy(probs);
console.log(`text (${text.length} chars, ${chars.length} distinct): "${text.slice(0, 40)}..."`);
const topChars = chars
  .map((c, i) => ({ c, p: probs[i] ?? 0 }))
  .sort((a, b) => b.p - a.p)
  .slice(0, 5);
for (const { c, p } of topChars) {
  console.log(`  '${c === " " ? "␣" : c}'  p = ${p.toFixed(3)}  surprise = ${surprise(p).toFixed(2)} bits`);
}
console.log(`entropy of character distribution: ${Htext.toFixed(3)} bits/char`);
console.log(`compare: uniform over ${chars.length} chars would be log₂(${chars.length}) = ${log2(chars.length).toFixed(3)} bits/char`);
console.log("→ English characters are far from uniform → far fewer bits needed.");

// ── 3. Cross-entropy and KL between concrete distributions ───────
console.log("\n=== 3. Cross-entropy & KL: the cost of wrong beliefs ===");
const P = [0.5, 0.25, 0.125, 0.125]; // reality
const Q1 = [0.5, 0.25, 0.125, 0.125]; // a perfect model
const Q2 = [0.25, 0.25, 0.25, 0.25]; // an ignorant (uniform) model
const Q3 = [0.9, 0.05, 0.025, 0.025]; // an overconfident model

console.log(`reality P = [${P.join(", ")}],  H(P) = ${entropy(P).toFixed(3)} bits`);
for (const [name, Q] of [["perfect Q=P     ", Q1], ["uniform         ", Q2], ["overconfident   ", Q3]] as [string, number[]][]) {
  const hpq = crossEntropy(P, Q);
  const kl = klDivergence(P, Q);
  console.log(`  ${name} H(P,Q) = ${hpq.toFixed(3)}   KL(P‖Q) = ${kl.toFixed(3)}   check H+KL = ${(entropy(P) + kl).toFixed(3)}`);
}
console.log("→ H(P,Q) = H(P) + KL(P‖Q): unavoidable surprise + your model's wrongness.");
console.log(`→ asymmetry: KL(P‖Q3) = ${klDivergence(P, Q3).toFixed(3)} but KL(Q3‖P) = ${klDivergence(Q3, P).toFixed(3)} — not a distance!`);

// ── 4. Perplexity of a toy character model on a string ───────────
console.log("\n=== 4. Perplexity: cross-entropy, exponentiated ===");
// Toy "language model": predicts each character from fixed probabilities
// (a 0-gram model). Model A learned from similar text; model B is uniform.
const testString = "the dog jumps over the fox";

function evaluateModel(name: string, modelProb: (ch: string) => number): void {
  let totalBits = 0;
  for (const ch of testString) totalBits += surprise(modelProb(ch));
  const bitsPerChar = totalBits / testString.length;
  const ppl = 2 ** bitsPerChar;
  console.log(`  ${name}: ${totalBits.toFixed(1)} total bits, ` +
    `${bitsPerChar.toFixed(3)} bits/char, perplexity = ${ppl.toFixed(2)}`);
}

// Model A: the character frequencies measured in part 2 (smoothed a little)
const probOf = new Map<string, number>();
chars.forEach((c, i) => probOf.set(c, probs[i] ?? 0));
const modelA = (ch: string) => 0.99 * (probOf.get(ch) ?? 0) + 0.01 / 27;
// Model B: uniform over 27 symbols (a-z + space) — knows nothing
const modelB = (_ch: string) => 1 / 27;

console.log(`test string: "${testString}"`);
evaluateModel("model A (learned freqs)", modelA);
evaluateModel("model B (uniform)      ", modelB);
console.log("→ perplexity K = 'as surprised as choosing among K equally likely options.'");
console.log("  Model B's perplexity is exactly 27 — it really is guessing among 27.");

// ── 5. Compression = prediction ──────────────────────────────────
console.log("\n=== 5. Compression = prediction ===");
// Shannon: the optimal code gives each symbol −log₂(p) bits.
// So a model's cross-entropy IS its size as a compressor. Price the text:
const naiveBits = text.length * 8; // ASCII: 8 bits/char, no model at all
const uniformBits = text.length * log2(chars.length); // "some model": knows the alphabet
const freqBits = text.length * Htext; // frequency model: optimal 0-gram code
console.log(`"${text.slice(0, 30)}..." (${text.length} chars) costs:`);
console.log(`  ASCII (no model):            ${naiveBits} bits`);
console.log(`  uniform over its alphabet:   ${Math.ceil(uniformBits)} bits`);
console.log(`  frequency model (H(P)):      ${Math.ceil(freqBits)} bits`);
console.log("→ every bit of predictive knowledge shaves bits off the encoding.");
console.log("  An LLM predicting ~1 bit/char could store this in ~" + text.length + " bits.");
console.log("  Better predictor ⇔ better compressor — same number, always.");

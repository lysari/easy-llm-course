// Lesson 27 — Reward Modeling: Teaching AI What "Good" Means
// Implements a reward model that learns human preferences from comparison data.
//
// Architecture: same transformer as Lesson 14 (Tiny GPT) + a value head
//   that maps the last token's hidden state to a scalar reward score.
// Training objective: Bradley-Terry preference loss
//   L = -log(sigmoid(r_good - r_bad))

// ── Math helpers ──────────────────────────────────────────────────────────────

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function matmul(A: number[][], B: number[][]): number[][] {
  const rows = A.length;
  const cols = B[0]?.length ?? 0;
  const inner = B.length;
  const C = Array.from({ length: rows }, () => Array<number>(cols).fill(0));
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++)
      for (let k = 0; k < inner; k++)
        C[i]![j]! += (A[i]?.[k] ?? 0) * (B[k]?.[j] ?? 0);
  return C;
}

function transpose(A: number[][]): number[][] {
  return (A[0] ?? []).map((_, j) => A.map(row => row[j] ?? 0));
}

function layerNorm(x: number[], eps = 1e-5): number[] {
  const mean = x.reduce((s, v) => s + v, 0) / x.length;
  const variance = x.reduce((s, v) => s + (v - mean) ** 2, 0) / x.length;
  return x.map(v => (v - mean) / Math.sqrt(variance + eps));
}

// ── Tokenizer ─────────────────────────────────────────────────────────────────
// Simple char-level tokenizer for the toy dataset.

class CharTokenizer {
  charToId = new Map<string, number>();
  idToChar = new Map<number, string>();

  constructor(text: string) {
    [...new Set(text.split(""))].sort().forEach((c, i) => {
      this.charToId.set(c, i);
      this.idToChar.set(i, c);
    });
  }

  get vocabSize(): number { return this.charToId.size; }

  encode(text: string): number[] {
    return text.split("").map(c => this.charToId.get(c) ?? 0);
  }

  decode(ids: number[]): string {
    return ids.map(i => this.idToChar.get(i) ?? "?").join("");
  }
}

// ── Dataset ───────────────────────────────────────────────────────────────────
// We teach the reward model to prefer:
//   - Correct pattern completions over wrong ones
//   - Concise answers over repetitive ones
//   - On-topic responses over off-topic ones
//
// The "task": complete simple patterns like "1 2 3 _" or "a b c _"

interface ComparisonPair {
  prompt: string;
  good: string;      // preferred response
  bad: string;       // dispreferred response
  reason: string;    // human-readable explanation (not used in training)
}

const COMPARISONS: ComparisonPair[] = [
  // Correct vs wrong completions
  { prompt: "1 2 3 ", good: "4",         bad: "7",         reason: "correct next number" },
  { prompt: "2 4 6 ", good: "8",         bad: "5",         reason: "correct even pattern" },
  { prompt: "a b c ", good: "d",         bad: "z",         reason: "correct alphabet" },
  { prompt: "1 3 5 ", good: "7",         bad: "6",         reason: "correct odd pattern" },
  { prompt: "10 20 30 ", good: "40",     bad: "25",        reason: "correct tens pattern" },

  // Concise vs repetitive
  { prompt: "what is 2+2? ", good: "4",  bad: "4 4 4 4 4", reason: "concise not repetitive" },
  { prompt: "spell cat: ",   good: "cat", bad: "cat cat cat", reason: "not repetitive" },
  { prompt: "color of sky: ", good: "blue", bad: "blue blue blue blue", reason: "single answer" },
  { prompt: "count to 3: ",  good: "1 2 3", bad: "1 1 1 1 1 1", reason: "correct not repeated" },
  { prompt: "hello world: ", good: "hi",   bad: "hi hi hi hi hi", reason: "concise greeting" },

  // On-topic vs off-topic
  { prompt: "1 2 3 ", good: "4",         bad: "apple",     reason: "stays on topic" },
  { prompt: "a b c ", good: "d",         bad: "123",       reason: "stays on topic" },
  { prompt: "2 4 6 ", good: "8",         bad: "xyz",       reason: "stays on topic" },
  { prompt: "what is 3+3? ", good: "6",  bad: "cat",       reason: "answers question" },
  { prompt: "count up: ",   good: "1 2 3", bad: "a b c",   reason: "correct domain" },

  // Accurate vs inaccurate
  { prompt: "5 minus 2 = ", good: "3",   bad: "4",         reason: "arithmetic correct" },
  { prompt: "2 times 4 = ", good: "8",   bad: "6",         reason: "multiplication correct" },
  { prompt: "10 minus 3 = ", good: "7",  bad: "8",         reason: "subtraction correct" },
  { prompt: "3 times 3 = ",  good: "9",  bad: "6",         reason: "multiplication correct" },
  { prompt: "4 plus 4 = ",   good: "8",  bad: "9",         reason: "addition correct" },
];

// Build vocabulary from all text in the dataset
const allText = COMPARISONS.flatMap(p => [p.prompt, p.good, p.bad]).join("");
const tokenizer = new CharTokenizer(allText);
const VOCAB_SIZE = tokenizer.vocabSize;

// ── Model config ──────────────────────────────────────────────────────────────

const EMBED_DIM = 24;
const BLOCK_SIZE = 16;   // max sequence length
const LR = 0.01;
const EPOCHS = 300;

// ── Random initializer ────────────────────────────────────────────────────────

function rand(r: number, c: number, scale = 0.1): number[][] {
  return Array.from({ length: r }, () =>
    Array.from({ length: c }, () => (Math.random() - 0.5) * scale)
  );
}

// ── Reward Model ──────────────────────────────────────────────────────────────
//
// Architecture:
//   1. Token embedding table     [vocabSize × embedDim]
//   2. Positional embedding table [blockSize × embedDim]
//   3. Transformer block (attention + residual + layernorm)
//   4. Value head: W_v [embedDim × 1] — maps last hidden state to scalar
//
// This is identical to Lesson 14's Tiny GPT EXCEPT we replace the vocab
// projection head with a single-neuron value head.

class RewardModel {
  // Transformer weights
  embTable: number[][];    // [vocabSize × embedDim]
  posTable: number[][];    // [blockSize × embedDim]
  Wq: number[][];          // [embedDim × embedDim]
  Wk: number[][];          // [embedDim × embedDim]
  Wv: number[][];          // [embedDim × embedDim]
  Wproj: number[][];       // [embedDim × embedDim] — attention output projection

  // Value head: maps last token hidden state → scalar reward
  // Shape: [embedDim × 1] — effectively a vector of embedDim weights
  valueHead: number[];     // [embedDim]
  valueBias: number;       // scalar bias

  constructor() {
    this.embTable = rand(VOCAB_SIZE, EMBED_DIM);
    this.posTable = rand(BLOCK_SIZE, EMBED_DIM, 0.01);
    this.Wq       = rand(EMBED_DIM, EMBED_DIM);
    this.Wk       = rand(EMBED_DIM, EMBED_DIM);
    this.Wv       = rand(EMBED_DIM, EMBED_DIM);
    this.Wproj    = rand(EMBED_DIM, EMBED_DIM);
    // Value head: small init so scores start near 0
    this.valueHead = Array.from({ length: EMBED_DIM }, () => (Math.random() - 0.5) * 0.01);
    this.valueBias = 0;
  }

  // Returns the last-token hidden state after the transformer layers.
  // We separate this so we can backprop through the value head explicitly.
  private transformerForward(tokens: number[]): {
    hiddenStates: number[][];  // [T × embedDim]
    lastHidden: number[];      // [embedDim] — the T-th hidden state
  } {
    const T = Math.min(tokens.length, BLOCK_SIZE);
    const trimmed = tokens.slice(-T);  // use most recent T tokens

    // Embed + positional encoding
    const X: number[][] = trimmed.map((tok, pos) =>
      (this.embTable[tok] ?? Array(EMBED_DIM).fill(0)).map(
        (v, j) => v + (this.posTable[pos]?.[j] ?? 0)
      )
    );

    // Self-attention
    const Q = matmul(X, this.Wq);
    const K = matmul(X, this.Wk);
    const V = matmul(X, this.Wv);
    const scale = Math.sqrt(EMBED_DIM);

    const scores = matmul(Q, transpose(K)).map(row => row.map(s => s / scale));

    // Causal mask
    for (let i = 0; i < T; i++)
      for (let j = i + 1; j < T; j++)
        scores[i]![j] = -Infinity;

    const attnWeights = scores.map(row => softmax(row));
    const attnOut = matmul(attnWeights, V);

    // Residual + LayerNorm + output projection
    const hiddenStates: number[][] = X.map((row, i) => {
      const residual = row.map((v, j) => v + (attnOut[i]?.[j] ?? 0));
      const normed = layerNorm(residual);
      // Project through Wproj (attention output projection)
      return normed.map((_, j) =>
        this.Wproj[j]!.reduce((s, w, k) => s + w * (normed[k] ?? 0), 0)
      );
    });

    return {
      hiddenStates,
      lastHidden: hiddenStates[hiddenStates.length - 1]!,
    };
  }

  // score(tokens) → scalar reward
  // This is the key method: it turns any token sequence into a single number.
  score(tokens: number[]): number {
    const { lastHidden } = this.transformerForward(tokens);
    // Dot product of lastHidden with valueHead + bias
    const rawScore = lastHidden.reduce((s, v, j) => s + v * (this.valueHead[j] ?? 0), 0);
    return rawScore + this.valueBias;
  }

  // forward() returns both the last hidden state (for backprop) and the scalar score.
  forward(tokens: number[]): { lastHidden: number[]; score: number } {
    const { lastHidden } = this.transformerForward(tokens);
    const score = lastHidden.reduce((s, v, j) => s + v * (this.valueHead[j] ?? 0), 0)
      + this.valueBias;
    return { lastHidden, score };
  }
}

// ── Preference Loss ───────────────────────────────────────────────────────────
//
// Bradley-Terry model:
//   P(good preferred over bad) = sigmoid(r_good - r_bad)
//   Loss = -log(sigmoid(r_good - r_bad))
//
// Gradient w.r.t. r_good: -(1 - sigmoid(r_good - r_bad))
// Gradient w.r.t. r_bad:  +(1 - sigmoid(r_good - r_bad))
//
// When the model is correct (r_good >> r_bad):
//   sigmoid → 1, gradient → 0 (no update needed)
// When the model is wrong (r_good << r_bad):
//   sigmoid → 0, gradient → -1 (large update)

function preferenceLoss(goodScore: number, badScore: number): number {
  return -Math.log(sigmoid(goodScore - badScore) + 1e-9);
}

// ── Training ──────────────────────────────────────────────────────────────────
//
// Backprop through the value head only (the transformer body is frozen for
// simplicity, same pattern as Lesson 14). The value head is a single linear
// layer so its gradient derivation is straightforward:
//
//   score = lastHidden · valueHead + valueBias
//   dL/d(score_good) = -(1 - sigmoid(score_good - score_bad))
//   dL/d(score_bad)  = +(1 - sigmoid(score_good - score_bad))
//   dL/d(valueHead[j]) = dL/d(score) * lastHidden[j]
//   dL/d(valueBias)    = dL/d(score)
//
// We also back-propagate into the embedding table for the last token,
// so the embeddings adjust to produce hidden states that the value head
// can distinguish. (Attention weights are frozen, as in Lesson 14.)

const rm = new RewardModel();

console.log("=== Lesson 27: Reward Modeling ===\n");
console.log(`Dataset: ${COMPARISONS.length} comparison pairs`);
console.log(`Vocabulary: ${VOCAB_SIZE} characters`);
console.log(`Embedding dim: ${EMBED_DIM}`);
console.log(`Architecture: Transformer (${EMBED_DIM}d) + value head (${EMBED_DIM}→1)\n`);

// Pre-training accuracy
function evalAccuracy(model: RewardModel): number {
  let correct = 0;
  for (const pair of COMPARISONS) {
    const goodTokens = tokenizer.encode(pair.prompt + pair.good);
    const badTokens  = tokenizer.encode(pair.prompt + pair.bad);
    if (model.score(goodTokens) > model.score(badTokens)) correct++;
  }
  return correct / COMPARISONS.length;
}

console.log(`Pre-training accuracy: ${(evalAccuracy(rm) * 100).toFixed(1)}%  (random ≈ 50%)\n`);

// Training loop
console.log(`--- Training for ${EPOCHS} epochs ---\n`);

for (let epoch = 0; epoch < EPOCHS; epoch++) {
  let totalLoss = 0;

  for (const pair of COMPARISONS) {
    const goodTokens = tokenizer.encode(pair.prompt + pair.good);
    const badTokens  = tokenizer.encode(pair.prompt + pair.bad);

    const goodFwd = rm.forward(goodTokens);
    const badFwd  = rm.forward(badTokens);

    const loss = preferenceLoss(goodFwd.score, badFwd.score);
    totalLoss += loss;

    // Gradient of preference loss w.r.t. each score
    // d(-log(sigmoid(r_g - r_b)))/d(r_g) = -(1 - sigmoid(r_g - r_b))
    const prob = sigmoid(goodFwd.score - badFwd.score);
    const dGood = -(1 - prob);   // want to increase r_good
    const dBad  =  (1 - prob);   // want to decrease r_bad

    // Backprop through value head for the "good" response
    for (let j = 0; j < EMBED_DIM; j++) {
      rm.valueHead[j]! -= LR * dGood * (goodFwd.lastHidden[j] ?? 0);
    }
    rm.valueBias -= LR * dGood;

    // Backprop through value head for the "bad" response
    // Note: the value head weights are shared — we accumulate both gradients.
    // We update from the bad pass separately to keep clarity.
    for (let j = 0; j < EMBED_DIM; j++) {
      rm.valueHead[j]! -= LR * dBad * (badFwd.lastHidden[j] ?? 0);
    }
    rm.valueBias -= LR * dBad;

    // Backprop into embedding table for the last token of each sequence
    // dL/d(lastHidden[j]) = dL/d(score) * valueHead[j]
    // dL/d(embTable[tok][j]) ≈ dL/d(lastHidden[j])   (frozen attention approx)
    const goodLastTok = goodTokens[goodTokens.length - 1] ?? 0;
    const badLastTok  = badTokens[badTokens.length - 1] ?? 0;

    for (let j = 0; j < EMBED_DIM; j++) {
      const vhj = rm.valueHead[j] ?? 0;
      rm.embTable[goodLastTok]![j]! -= LR * dGood * vhj * 0.1;
      rm.embTable[badLastTok]![j]!  -= LR * dBad  * vhj * 0.1;
    }
  }

  const avgLoss = totalLoss / COMPARISONS.length;

  if (epoch === 0 || (epoch + 1) % 50 === 0 || epoch === EPOCHS - 1) {
    const acc = evalAccuracy(rm);
    console.log(
      `Epoch ${String(epoch + 1).padStart(3)} — ` +
      `Loss: ${avgLoss.toFixed(4)}  ` +
      `Accuracy: ${(acc * 100).toFixed(1)}%`
    );
  }
}

console.log(`\nPost-training accuracy: ${(evalAccuracy(rm) * 100).toFixed(1)}%\n`);

// ── Score Distribution ────────────────────────────────────────────────────────

console.log("=== Score Distribution: Good vs Bad Responses ===\n");

const goodScores: number[] = [];
const badScores: number[] = [];

for (const pair of COMPARISONS) {
  goodScores.push(rm.score(tokenizer.encode(pair.prompt + pair.good)));
  badScores.push(rm.score(tokenizer.encode(pair.prompt + pair.bad)));
}

const allScores = [...goodScores, ...badScores];
const minScore = Math.min(...allScores);
const maxScore = Math.max(...allScores);
const range = maxScore - minScore || 1;

// ASCII histogram with 10 buckets
const NUM_BUCKETS = 10;
const goodCounts = Array(NUM_BUCKETS).fill(0);
const badCounts  = Array(NUM_BUCKETS).fill(0);

for (const s of goodScores) {
  const idx = Math.min(Math.floor(((s - minScore) / range) * NUM_BUCKETS), NUM_BUCKETS - 1);
  goodCounts[idx]++;
}
for (const s of badScores) {
  const idx = Math.min(Math.floor(((s - minScore) / range) * NUM_BUCKETS), NUM_BUCKETS - 1);
  badCounts[idx]++;
}

const bucketWidth = range / NUM_BUCKETS;
console.log("Score histogram (G = good responses, B = bad responses):");
console.log(`Range: [${minScore.toFixed(2)}, ${maxScore.toFixed(2)}]\n`);

for (let i = 0; i < NUM_BUCKETS; i++) {
  const bucketStart = (minScore + i * bucketWidth).toFixed(2);
  const bucketEnd   = (minScore + (i + 1) * bucketWidth).toFixed(2);
  const gBar = "G".repeat(goodCounts[i]);
  const bBar = "B".repeat(badCounts[i]);
  console.log(`[${bucketStart.padStart(6)}, ${bucketEnd.padStart(6)}] ${gBar}${bBar}`);
}

console.log("\nIdeal: G's cluster on the right (high scores), B's on the left (low scores).\n");

// Per-pair breakdown
console.log("=== Per-Pair Scores (sample of 8) ===\n");
console.log("Prompt".padEnd(20) + "Good Response".padEnd(15) + "Good Score".padEnd(14) +
            "Bad Response".padEnd(15) + "Bad Score".padEnd(12) + "RM Correct?");
console.log("-".repeat(90));

for (let i = 0; i < Math.min(8, COMPARISONS.length); i++) {
  const pair = COMPARISONS[i]!;
  const gs = goodScores[i]!;
  const bs = badScores[i]!;
  const correct = gs > bs ? "YES" : "NO ";
  console.log(
    pair.prompt.padEnd(20) +
    pair.good.padEnd(15) +
    gs.toFixed(4).padEnd(14) +
    pair.bad.padEnd(15) +
    bs.toFixed(4).padEnd(12) +
    correct
  );
}

// ── Reward Hacking Demo ───────────────────────────────────────────────────────
//
// Reward hacking: a response that scores high on the RM but is semantically
// wrong or useless. This happens because the RM is an imperfect proxy for
// actual human preferences.
//
// Example: we craft a "cheating" response that looks like a "good" response
// token-by-token (sharing the same last character) but gives the wrong answer.

console.log("\n=== Reward Hacking Demo ===\n");
console.log(
  "Reward hacking occurs when the LLM learns to game the RM instead of\n" +
  "actually being helpful. Here we simulate it with crafted 'cheat' responses.\n"
);

const hackingExamples = [
  {
    prompt: "1 2 3 ",
    correct: "4",
    cheating: "4 4 4 4 4",    // repetitive but ends with "4" — same last char
    description: "repetitive answer (same last token as correct)",
  },
  {
    prompt: "2 4 6 ",
    correct: "8",
    cheating: "wrong answer ending in 8",    // nonsense but ends on same character
    description: "off-topic string that ends on the 'right' character",
  },
  {
    prompt: "1 2 3 ",
    correct: "4",
    cheating: "not a number 4",   // technically contains "4" but is wrong
    description: "contains the right character but is not a direct answer",
  },
];

for (const ex of hackingExamples) {
  const correctTokens  = tokenizer.encode(ex.prompt + ex.correct);
  const cheatingTokens = tokenizer.encode(ex.prompt + ex.cheating);

  // Use safe fallback for tokens not in vocabulary
  const safeCheatTokens = ex.cheating.split("").map(c => {
    const id = tokenizer.charToId.get(c);
    return id !== undefined ? id : 0;
  });
  const safeCheatFull = tokenizer.encode(ex.prompt).concat(safeCheatTokens);

  const correctScore  = rm.score(correctTokens);
  const cheatingScore = rm.score(safeCheatFull);
  const hacked = cheatingScore > correctScore;

  console.log(`Prompt:    "${ex.prompt}"`);
  console.log(`Correct:   "${ex.correct}"  → RM score: ${correctScore.toFixed(4)}`);
  console.log(`Cheating:  "${ex.cheating}"  → RM score: ${cheatingScore.toFixed(4)}`);
  console.log(`RM hacked? ${hacked ? "YES — cheating scored higher!" : "No — RM held up."}`);
  console.log(`Reason:    ${ex.description}`);
  console.log();
}

console.log("─".repeat(65));
console.log("Why this happens:");
console.log();
console.log("  The RM scores the LAST TOKEN's hidden state. A response that");
console.log("  happens to end on the same token as a 'good' response can fool");
console.log("  the RM, even if the rest of the response is garbage.");
console.log();
console.log("  In real LLMs this manifests as:");
console.log("  - Responses padded with affirmations (\"Certainly! I'm happy to help!\"");
console.log("    \"Great question! ...\" \"I hope that was helpful!\")");
console.log("  - Overly verbose answers that 'look' thorough but are repetitive");
console.log("  - Confident-sounding wrong answers");
console.log();

// ── KL Penalty Explanation ────────────────────────────────────────────────────

console.log("=== The KL Divergence Fix ===\n");
console.log("During RL training (Lesson 28), the full objective is:");
console.log();
console.log("  maximize: r(response) - β * KL(LLM_rl || LLM_sft)");
console.log();
console.log("Where:");
console.log("  r(response)         = reward model score (what we trained above)");
console.log("  KL(LLM_rl||LLM_sft) = how far the RL model has drifted from SFT");
console.log("  β                   = penalty coefficient (typically 0.02–0.2)");
console.log();
console.log("The KL term penalizes generating text that the SFT model would");
console.log("never produce. This prevents reward hacking because:");
console.log();
console.log("  1. The reward model was trained on SFT-model outputs");
console.log("  2. If the RL model generates out-of-distribution text,");
console.log("     the RM's scores are unreliable");
console.log("  3. The KL penalty forces the model to stay 'in distribution'");
console.log("     where the RM has signal");
console.log();
console.log("Intuition: the RM is a map. Reward hacking is walking off the edge");
console.log("of the map. The KL penalty keeps you on the map.");
console.log();

// ── Summary ───────────────────────────────────────────────────────────────────

console.log("=== Summary ===\n");
console.log("What we built:");
console.log("  1. Preference dataset: 20 (prompt, good, bad) comparison triples");
console.log("  2. RewardModel: Transformer + value head (embedDim → 1 scalar)");
console.log("  3. Preference loss: -log(sigmoid(r_good - r_bad))  [Bradley-Terry]");
console.log("  4. Training: backprop through value head + embedding table");
console.log("  5. Evaluation: % of pairs where RM correctly ranks good > bad");
console.log("  6. Reward hacking demo: crafted responses that fool the RM");
console.log();
console.log("Key numbers from real RLHF (Stiennon et al. 2020):");
console.log("  - 60,000 human comparisons (vs our 20)");
console.log("  - RM accuracy ~75% (human inter-annotator agreement ~73%)");
console.log("  - KL coefficient β ≈ 0.02–0.2 in practice");
console.log();
console.log("Next: Lesson 28 — PPO: using this reward model to fine-tune the LLM.");

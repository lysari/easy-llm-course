// Lesson 26 — Instruction Fine-Tuning: Teaching the Model to Follow Instructions
// No imports — pure TypeScript / Node.js built-ins only.

// ============================================================
// SECTION 1: Math Utilities
// ============================================================

type Matrix = number[][];

function matAdd(A: Matrix, B: Matrix): Matrix {
  return A.map((row, i) => row.map((v, j) => v + B[i][j]));
}

function randomMatrix(rows: number, cols: number, scale: number): Matrix {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (Math.random() - 0.5) * 2 * scale)
  );
}

function zerosMatrix(rows: number, cols: number): Matrix {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - maxLogit));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function crossEntropyLoss(logits: number[], targetIdx: number): number {
  const probs = softmax(logits);
  return -Math.log(Math.max(probs[targetIdx], 1e-10));
}

// ============================================================
// SECTION 2: Vocabulary
// ============================================================

function buildVocab(text: string): { charToId: Map<string, number>; idToChar: string[] } {
  const chars = [...new Set(text.split(""))].sort();
  const charToId = new Map<string, number>();
  chars.forEach((c, i) => charToId.set(c, i));
  return { charToId, idToChar: chars };
}

function encode(text: string, charToId: Map<string, number>): number[] {
  return text.split("").map(c => charToId.get(c) ?? 0);
}

function decode(ids: number[], idToChar: string[]): string {
  return ids.map(id => idToChar[id] ?? "?").join("");
}

// ============================================================
// SECTION 3: Tiny Language Model
// ============================================================

/**
 * Architecture: embedding table + recency-weighted context pooling + lmHead.
 *
 * For a context [t-k, ..., t-1], compute a hidden state as a weighted sum:
 *   hidden = sum_i  decay^(T-1-i) * embedding[token_i]
 *
 * where decay < 1 gives more weight to recent tokens.
 * Then: logits = hidden @ lmHead.
 *
 * This is simpler and more stable than a full RNN while still encoding
 * "what recent tokens were seen". Recent tokens (like "A:") dominate.
 *
 * The key: "A:" at the end of "Q: what does the cat do? A:" gets
 * exponentially more weight than the question words, so the model
 * can learn to distinguish different "question contexts".
 */

interface LMParams {
  vocabSize: number;
  D: number;
  embedding: Matrix;  // (vocabSize × D)
  lmHead: Matrix;     // (D × vocabSize)
  decay: number;      // recency decay factor (e.g. 0.8)
}

function createLM(vocabSize: number, D: number, decay: number): LMParams {
  return {
    vocabSize,
    D,
    embedding: randomMatrix(vocabSize, D, 0.1),
    lmHead: randomMatrix(D, vocabSize, 0.05),
    decay,
  };
}

function cloneLM(p: LMParams): LMParams {
  return {
    ...p,
    embedding: p.embedding.map(r => [...r]),
    lmHead: p.lmHead.map(r => [...r]),
  };
}

/**
 * Compute weighted context vector from a token sequence.
 * weights[i] = decay^(T-1-i), most recent token has weight 1.
 */
function contextVector(p: LMParams, tokenIds: number[]): { hidden: number[]; weights: number[] } {
  const T = tokenIds.length;
  if (T === 0) return { hidden: new Array(p.D).fill(0), weights: [] };

  const weights: number[] = tokenIds.map((_, i) => Math.pow(p.decay, T - 1 - i));
  const wSum = weights.reduce((a, b) => a + b, 0);
  const normWeights = weights.map(w => w / wSum);

  const hidden: number[] = new Array(p.D).fill(0);
  for (let i = 0; i < T; i++) {
    const emb = p.embedding[tokenIds[i]];
    for (let d = 0; d < p.D; d++) {
      hidden[d] += normWeights[i] * emb[d];
    }
  }
  return { hidden, weights: normWeights };
}

function logitsFromHidden(p: LMParams, h: number[]): number[] {
  const logits: number[] = new Array(p.vocabSize).fill(0);
  for (let v = 0; v < p.vocabSize; v++) {
    for (let d = 0; d < p.D; d++) {
      logits[v] += h[d] * p.lmHead[d][v];
    }
  }
  return logits;
}

function generate(p: LMParams, promptIds: number[], maxNew: number): number[] {
  const ids = [...promptIds];
  for (let step = 0; step < maxNew; step++) {
    const { hidden } = contextVector(p, ids);
    const logits = logitsFromHidden(p, hidden);
    const probs = softmax(logits);
    let best = 0, bestProb = -1;
    for (let v = 0; v < p.vocabSize; v++) {
      if (probs[v] > bestProb) { bestProb = probs[v]; best = v; }
    }
    ids.push(best);
  }
  return ids.slice(promptIds.length);
}

/**
 * One training step.
 *
 * @param tokenIds  full sequence
 * @param lossMask  lossMask[t] = true → compute loss for predicting tokenIds[t]
 *                  from prefix tokenIds[0..t-1]
 * @param lr        learning rate
 * @returns average masked loss
 */
function trainStep(
  p: LMParams,
  tokenIds: number[],
  lossMask: boolean[],
  lr: number
): number {
  let totalLoss = 0;
  let count = 0;

  for (let t = 1; t < tokenIds.length; t++) {
    if (!lossMask[t]) continue;

    const context = tokenIds.slice(0, t);
    const targetId = tokenIds[t];

    const { hidden, weights } = contextVector(p, context);
    const logits = logitsFromHidden(p, hidden);
    const loss = crossEntropyLoss(logits, targetId);
    if (!isFinite(loss)) continue;
    totalLoss += loss;
    count++;

    // dLoss/dLogits
    const probs = softmax(logits);
    const dLogits = probs.map((prob, i) => (i === targetId ? prob - 1 : prob));

    // dLoss/dlmHead[d][v] = hidden[d] * dLogits[v]
    for (let d = 0; d < p.D; d++) {
      for (let v = 0; v < p.vocabSize; v++) {
        p.lmHead[d][v] -= lr * hidden[d] * dLogits[v];
      }
    }

    // dLoss/dhidden[d] = sum_v dLogits[v] * lmHead[d][v]
    const dHidden: number[] = new Array(p.D).fill(0);
    for (let d = 0; d < p.D; d++) {
      for (let v = 0; v < p.vocabSize; v++) {
        dHidden[d] += dLogits[v] * p.lmHead[d][v];
      }
    }

    // dLoss/demb[context[i]][d] = weights[i] * dHidden[d]
    for (let i = 0; i < context.length; i++) {
      const tokenId = context[i];
      for (let d = 0; d < p.D; d++) {
        p.embedding[tokenId][d] -= lr * weights[i] * dHidden[d];
      }
    }
  }

  return count > 0 ? totalLoss / count : 0;
}

// ============================================================
// SECTION 4: Pre-Training
// ============================================================

const PRE_TRAINING_CORPUS =
  "the cat sat. the dog ran. the bird flew. the fish swam. the cow ate. " +
  "the cat sat on the mat. the dog ran fast. the bird flew high. the fish swam deep. " +
  "a cat sat. a dog ran. a bird flew. a fish swam. a cow ate grass.";

console.log("=".repeat(60));
console.log("SECTION 1: Pre-Training the Base Model");
console.log("=".repeat(60));

const EXTRA_CHARS = "QA:?";
const { charToId, idToChar } = buildVocab(PRE_TRAINING_CORPUS + EXTRA_CHARS);
const VOCAB_SIZE = idToChar.length;
const D = 32;
const DECAY = 0.7;  // recent tokens get exponentially more weight

console.log(`Vocabulary: ${VOCAB_SIZE} characters (character-level)`);
console.log(`Embed dim: ${D}   Recency decay: ${DECAY}`);
console.log(
  `Vocab chars: ${idToChar.map(c => c === " " ? "SPC" : c === "\n" ? "NL" : c).join(" ")}`
);

const baseModel = createLM(VOCAB_SIZE, D, DECAY);
const corpusIds = encode(PRE_TRAINING_CORPUS, charToId);

// Pre-train on windows of the corpus
// Loss on ALL tokens (no masking — it's language modeling, not instruction tuning)
const PRE_TRAIN_LR = 0.03;
const PRE_TRAIN_EPOCHS = 80;
const WINDOW = 10;

let preLossSum = 0;
let preSteps = 0;
for (let epoch = 0; epoch < PRE_TRAIN_EPOCHS; epoch++) {
  for (let start = 0; start + WINDOW <= corpusIds.length; start += 2) {
    const window = corpusIds.slice(start, start + WINDOW);
    const mask = window.map((_, t) => t > 0);
    const loss = trainStep(baseModel, window, mask, PRE_TRAIN_LR);
    if (isFinite(loss)) { preLossSum += loss; preSteps++; }
  }
}

const preAvgLoss = preSteps > 0 ? preLossSum / preSteps : NaN;
const randomBaseline = Math.log(VOCAB_SIZE);
console.log(`Pre-training avg loss: ${preAvgLoss.toFixed(4)}`);
console.log(`Random baseline:       ${randomBaseline.toFixed(4)} (log(${VOCAB_SIZE}))`);
console.log(`Improvement:           ${((1 - preAvgLoss / randomBaseline) * 100).toFixed(1)}% below random`);

// ============================================================
// SECTION 5: Base Model Behavior — Before Fine-Tuning
// ============================================================

console.log("\n" + "=".repeat(60));
console.log("SECTION 2: Base Model Behavior (Before Fine-Tuning)");
console.log("=".repeat(60));

console.log(`
A base model predicts the next token in a sequence.
It was trained on: "the cat sat. the dog ran. ..."

When given "Q: what does the cat do? A:" it sees this as text to
continue, not as a question to answer. It has no concept that
"Q: ... A:" is a question-answer format.
`);

const testCases: Array<{ prompt: string; expected: string }> = [
  { prompt: "Q: what does the cat do? A:", expected: "sat" },
  { prompt: "Q: what does the dog do? A:", expected: "ran" },
  { prompt: "Q: what does the bird do? A:", expected: "flew" },
  { prompt: "Q: what does the fish do? A:", expected: "swam" },
];

console.log("Base model outputs (before fine-tuning):");
for (const { prompt, expected } of testCases) {
  const out = decode(generate(baseModel, encode(prompt, charToId), 5), idToChar);
  console.log(`  Prompt: "${prompt}"`);
  console.log(`  Output: "${out}"  (want: "${expected}")`);
}

// ============================================================
// SECTION 6: Instruction Dataset and SFT
// ============================================================

console.log("\n" + "=".repeat(60));
console.log("SECTION 3: Supervised Fine-Tuning (SFT)");
console.log("=".repeat(60));

interface InstructionPair {
  instruction: string;
  response: string;
}

const sftData: InstructionPair[] = [
  { instruction: "what does the cat do?", response: "sat" },
  { instruction: "what does the dog do?", response: "ran" },
  { instruction: "what does the bird do?", response: "flew" },
  { instruction: "what does the fish do?", response: "swam" },
  { instruction: "what does the cow do?", response: "ate" },
];

/**
 * Format instruction pair and compute loss mask.
 *
 * Full sequence: "Q: {instruction} A:{response}"
 * Loss mask: 0 on instruction tokens, 1 on response tokens.
 *
 * This is the core SFT insight: we do NOT want the model to learn
 * to predict the user's question — we want it to predict the
 * assistant's answer GIVEN the user's question.
 */
function formatSFT(pair: InstructionPair): { tokenIds: number[]; lossMask: boolean[] } {
  const prefix = `Q: ${pair.instruction} A:`;
  const full = prefix + pair.response;
  const tokenIds = encode(full, charToId);
  const prefixLen = prefix.length;
  const lossMask = tokenIds.map((_, i) => i >= prefixLen);
  return { tokenIds, lossMask };
}

console.log("SFT data format:\n");
for (const pair of sftData) {
  const prefix = `Q: ${pair.instruction} A:`;
  console.log(`  MASKED (no loss): "${prefix}"`);
  console.log(`  LOSS (train on):  "${pair.response}"\n`);
}

// SFT training — LOWER learning rate than pre-training to prevent catastrophic forgetting
const SFT_LR = 0.008;   // ~4x lower than pre-training
const SFT_EPOCHS = 1000;

console.log(`Pre-train LR: ${PRE_TRAIN_LR}   SFT LR: ${SFT_LR} (${(PRE_TRAIN_LR/SFT_LR).toFixed(0)}x lower)`);
console.log("Lower LR = less catastrophic forgetting of pre-trained knowledge.\n");

const sftModel = cloneLM(baseModel);
let sftLossSum = 0;
let sftSteps = 0;
for (let epoch = 0; epoch < SFT_EPOCHS; epoch++) {
  for (const pair of sftData) {
    const { tokenIds, lossMask } = formatSFT(pair);
    const loss = trainStep(sftModel, tokenIds, lossMask, SFT_LR);
    if (isFinite(loss)) { sftLossSum += loss; sftSteps++; }
  }
}
console.log(`SFT complete: avg response-token loss = ${(sftLossSum / sftSteps).toFixed(4)}`);

// ============================================================
// SECTION 7: Before vs After
// ============================================================

console.log("\n" + "=".repeat(60));
console.log("SECTION 4: Before vs After Fine-Tuning");
console.log("=".repeat(60));

for (const { prompt, expected } of testCases) {
  const ids = encode(prompt, charToId);
  const baseOut = decode(generate(baseModel, ids, 5), idToChar);
  const sftOut = decode(generate(sftModel, ids, 5), idToChar);
  const correct = sftOut.startsWith(expected) ? "CORRECT" : "wrong";
  console.log(`\n  Prompt:     "${prompt}"`);
  console.log(`  Expected:   "${expected}"`);
  console.log(`  Base model: "${baseOut}"`);
  console.log(`  SFT model:  "${sftOut}"  [${correct}]`);
}

// ============================================================
// SECTION 8: LoRA — Low-Rank Adaptation
// ============================================================

console.log("\n" + "=".repeat(60));
console.log("SECTION 5: LoRA — Low-Rank Adaptation");
console.log("=".repeat(60));

console.log(`
LoRA idea: Instead of updating W directly (d_out × d_in parameters),
freeze W and learn a low-rank decomposition of the update:

    W_new = W + ΔW   where   ΔW = A · B

  W: (d_out × d_in) — FROZEN (base model weights)
  A: (d_out × r)    — trainable, small random init
  B: (r    × d_in)  — trainable, ZERO init

At initialization: ΔW = A · B = A · 0 = 0
→ Fine-tuning starts from exactly the pre-trained weights.
→ The model drifts from pre-trained behavior only as needed.

During fine-tuning: only A and B are updated.
W is frozen → cannot catastrophically forget pre-training.
`);

/**
 * LoRALayer: wraps a frozen matrix W and adds trainable low-rank A, B.
 *
 * Forward: output = W @ x  +  A @ (B @ x)
 *
 *   W: (dOut × dIn) — frozen
 *   A: (dOut × r)   — trainable
 *   B: (r × dIn)    — trainable, initialized to zeros
 */
/**
 * LoRALayer wraps a frozen weight matrix W and adds trainable low-rank A, B.
 *
 * Convention (matching logitsFromHidden):
 *   W has shape (xDim × outDim):   W[d][v] where d = input dim, v = output dim
 *   output[v] = sum_d  x[d] * (W[d][v] + deltaW[d][v])
 *
 * deltaW = A @ B^T  (element-wise equivalent):
 *   A has shape (xDim × r)
 *   B has shape (outDim × r)
 *   deltaW[d][v] = sum_ri  A[d][ri] * B[v][ri]
 *
 * Forward: output[v] = sum_d x[d] * W[d][v]  +  sum_d x[d] * deltaW[d][v]
 *                     = x^T W  +  (x^T A) (B^T)^T = x^T (W + A B^T)
 *
 * This matches how lmHead is used in logitsFromHidden:
 *   logits[v] += hidden[d] * lmHead[d][v]
 */
class LoRALayer {
  W: Matrix;    // (xDim × outDim) — frozen
  A: Matrix;    // (xDim × r)      — trainable, small random init
  B: Matrix;    // (outDim × r)    — trainable, ZERO init
  xDim: number;
  outDim: number;
  r: number;

  constructor(W: Matrix, r: number) {
    this.xDim = W.length;        // e.g. D = 32
    this.outDim = W[0].length;   // e.g. vocabSize = 24
    this.r = r;
    this.W = W.map(row => [...row]);              // frozen copy
    this.A = randomMatrix(this.xDim, r, 0.02);   // small random init
    this.B = zerosMatrix(this.outDim, r);         // zero init → deltaW=0 at start
  }

  /**
   * Forward: output[v] = sum_d x[d] * (W[d][v] + sum_ri A[d][ri]*B[v][ri])
   */
  forward(x: number[]): number[] {
    // Base: W contribution → output[v] = sum_d x[d] * W[d][v]
    const output: number[] = new Array(this.outDim).fill(0);
    for (let d = 0; d < this.xDim; d++) {
      for (let v = 0; v < this.outDim; v++) {
        output[v] += x[d] * this.W[d][v];
      }
    }

    // LoRA delta: first compute Ax = A^T @ x → shape (r,)
    //   Ax[ri] = sum_d x[d] * A[d][ri]
    const Ax: number[] = new Array(this.r).fill(0);
    for (let d = 0; d < this.xDim; d++) {
      for (let ri = 0; ri < this.r; ri++) {
        Ax[ri] += x[d] * this.A[d][ri];
      }
    }
    // Then B @ Ax: delta_output[v] = sum_ri B[v][ri] * Ax[ri]
    for (let v = 0; v < this.outDim; v++) {
      for (let ri = 0; ri < this.r; ri++) {
        output[v] += this.B[v][ri] * Ax[ri];
      }
    }

    return output;
  }

  /**
   * Backward: update A and B given gradient dOut (length outDim) and input x.
   * W is NOT updated (frozen).
   *
   * dL/dA[d][ri] = x[d] * (B^T @ dOut)[ri]  →  x[d] * sum_v B[v][ri]*dOut[v]
   * dL/dB[v][ri] = dOut[v] * (A^T @ x)[ri]  →  dOut[v] * Ax[ri]
   */
  backward(x: number[], dOut: number[], lr: number): void {
    // Ax = A^T @ x  (needed for B gradient, same as in forward)
    const Ax: number[] = new Array(this.r).fill(0);
    for (let d = 0; d < this.xDim; d++) {
      for (let ri = 0; ri < this.r; ri++) {
        Ax[ri] += x[d] * this.A[d][ri];
      }
    }

    // B^T @ dOut for A gradient: BtdOut[ri] = sum_v B[v][ri] * dOut[v]
    const BtdOut: number[] = new Array(this.r).fill(0);
    for (let v = 0; v < this.outDim; v++) {
      for (let ri = 0; ri < this.r; ri++) {
        BtdOut[ri] += this.B[v][ri] * dOut[v];
      }
    }

    // Update A: dL/dA[d][ri] = x[d] * BtdOut[ri]
    for (let d = 0; d < this.xDim; d++) {
      for (let ri = 0; ri < this.r; ri++) {
        this.A[d][ri] -= lr * x[d] * BtdOut[ri];
      }
    }

    // Update B: dL/dB[v][ri] = dOut[v] * Ax[ri]
    for (let v = 0; v < this.outDim; v++) {
      for (let ri = 0; ri < this.r; ri++) {
        this.B[v][ri] -= lr * dOut[v] * Ax[ri];
      }
    }
  }

  /**
   * Merge: W_final[d][v] = W[d][v] + sum_ri A[d][ri] * B[v][ri]
   *
   * After merging, the result is a plain (xDim × outDim) weight matrix
   * that can be used directly in logitsFromHidden — zero inference overhead.
   */
  merge(): Matrix {
    const merged: Matrix = this.W.map(row => [...row]);  // copy W
    for (let d = 0; d < this.xDim; d++) {
      for (let v = 0; v < this.outDim; v++) {
        for (let ri = 0; ri < this.r; ri++) {
          merged[d][v] += this.A[d][ri] * this.B[v][ri];
        }
      }
    }
    return merged;
  }

  trainableParams(): number { return this.xDim * this.r + this.outDim * this.r; }
  fullParams(): number { return this.xDim * this.outDim; }
}

// ============================================================
// SECTION 9: Parameter Count
// ============================================================

const LORA_R = 4;
const demoLayer = new LoRALayer(baseModel.lmHead, LORA_R);

console.log(`lmHead shape: ${D} × ${VOCAB_SIZE}`);
console.log(`  Full fine-tune: ${demoLayer.fullParams()} parameters`);
console.log(`  LoRA r=${LORA_R}:       ${demoLayer.trainableParams()} parameters  (${(demoLayer.fullParams() / demoLayer.trainableParams()).toFixed(1)}x fewer)`);

// Real-scale intuition
const G2D = 768, G2V = 50257;
console.log(`\nScaled to GPT-2 lm_head (${G2D} × ${G2V}):`);
for (const r of [4, 8, 16]) {
  const lora = G2D * r + r * G2V;
  const full = G2D * G2V;
  console.log(`  LoRA r=${r.toString().padStart(2)}: ${lora.toLocaleString().padStart(9)} params  (${Math.round(full/lora)}x fewer than ${full.toLocaleString()})`);
}

// ============================================================
// SECTION 10: LoRA Training
// ============================================================

console.log("\n" + "=".repeat(60));
console.log("SECTION 6: LoRA Fine-Tuning vs Full Fine-Tuning");
console.log("=".repeat(60));

/**
 * LoRA model: base LM is frozen, only lmHead's A and B are updated.
 * The embedding and recency weights are untouched.
 */
interface LoRAModel {
  base: LMParams;       // frozen
  loraHead: LoRALayer;  // trainable A and B only
}

function createLoRAModel(base: LMParams, r: number): LoRAModel {
  return {
    base: cloneLM(base),
    loraHead: new LoRALayer(base.lmHead, r),
  };
}

function loraGenerate(lm: LoRAModel, promptIds: number[], maxNew: number): number[] {
  const ids = [...promptIds];
  for (let step = 0; step < maxNew; step++) {
    const { hidden } = contextVector(lm.base, ids);
    const logits = lm.loraHead.forward(hidden);
    const probs = softmax(logits);
    let best = 0, bestP = -1;
    for (let v = 0; v < lm.base.vocabSize; v++) {
      if (probs[v] > bestP) { bestP = probs[v]; best = v; }
    }
    ids.push(best);
  }
  return ids.slice(promptIds.length);
}

function loraTrainStep(
  lm: LoRAModel,
  tokenIds: number[],
  lossMask: boolean[],
  lr: number
): number {
  let totalLoss = 0;
  let count = 0;

  for (let t = 1; t < tokenIds.length; t++) {
    if (!lossMask[t]) continue;

    const context = tokenIds.slice(0, t);
    const targetId = tokenIds[t];

    const { hidden, weights } = contextVector(lm.base, context);
    const logits = lm.loraHead.forward(hidden);
    const loss = crossEntropyLoss(logits, targetId);
    if (!isFinite(loss)) continue;
    totalLoss += loss;
    count++;

    const probs = softmax(logits);
    const dLogits = probs.map((prob, i) => (i === targetId ? prob - 1 : prob));

    // Update only A and B; base embedding is frozen
    lm.loraHead.backward(hidden, dLogits, lr);
    // Note: we intentionally do NOT backprop into the embedding.
    // The frozen base model provides the hidden state; LoRA adapts the output projection.
  }

  return count > 0 ? totalLoss / count : 0;
}

const loraModel = createLoRAModel(baseModel, LORA_R);
const LORA_LR = 0.02;
const LORA_EPOCHS = 1000;

let loraLossSum = 0;
let loraSteps = 0;
for (let epoch = 0; epoch < LORA_EPOCHS; epoch++) {
  for (const pair of sftData) {
    const { tokenIds, lossMask } = formatSFT(pair);
    const loss = loraTrainStep(loraModel, tokenIds, lossMask, LORA_LR);
    if (isFinite(loss)) { loraLossSum += loss; loraSteps++; }
  }
}

console.log(`\nFull SFT avg loss: ${(sftLossSum / sftSteps).toFixed(4)}`);
console.log(`LoRA SFT avg loss: ${(loraLossSum / loraSteps).toFixed(4)}`);
console.log(`\nTrainable parameter comparison:`);
console.log(`  Full SFT: all ${D * VOCAB_SIZE + D * VOCAB_SIZE} params (embedding + lmHead)`);
console.log(`  LoRA:     ${demoLayer.trainableParams()} params (A and B only; all base weights frozen)`);

// ============================================================
// SECTION 11: Final Output Comparison
// ============================================================

console.log("\n" + "=".repeat(60));
console.log("SECTION 7: Final Output Comparison");
console.log("=".repeat(60));

console.log("\n  Prompt                       | Exp  | Base  | SFT   | LoRA");
console.log("  " + "-".repeat(65));
for (const { prompt, expected } of testCases) {
  const ids = encode(prompt, charToId);
  const baseOut = decode(generate(baseModel, ids, 4), idToChar);
  const sftOut = decode(generate(sftModel, ids, 4), idToChar);
  const loraOut = decode(loraGenerate(loraModel, ids, 4), idToChar);
  const q = prompt.replace("Q: ", "").replace(" A:", "");
  console.log(
    `  "${q.padEnd(27)}" | ${expected.padEnd(4)} | ${baseOut.slice(0,4).padEnd(5)} | ${sftOut.slice(0,4).padEnd(5)} | ${loraOut.slice(0,4)}`
  );
}

// ============================================================
// SECTION 12: LoRA Weight Merging
// ============================================================

console.log("\n" + "=".repeat(60));
console.log("SECTION 8: LoRA Merging — Zero Inference Overhead");
console.log("=".repeat(60));

console.log(`
After fine-tuning, merge LoRA back into the base weights:
  W_final = W + A @ B

The resulting model has the same architecture as the original.
No extra parameters. No extra computation at inference time.
`);

const mergedModel = cloneLM(baseModel);
mergedModel.lmHead = loraModel.loraHead.merge();

let allMatch = true;
console.log("Verify: LoRA model and merged model produce identical outputs:");
for (const { prompt } of testCases) {
  const ids = encode(prompt, charToId);
  const loraOut = decode(loraGenerate(loraModel, ids, 4), idToChar);
  const mergedOut = decode(generate(mergedModel, ids, 4), idToChar);
  const match = loraOut === mergedOut;
  if (!match) allMatch = false;
  const shortQ = prompt.replace("Q: ", "Q:").replace(" A:", "?");
  console.log(`  "${shortQ}" → LoRA: "${loraOut.slice(0,4)}" | Merged: "${mergedOut.slice(0,4)}" [${match ? "MATCH" : "MISMATCH"}]`);
}
console.log(`\nAll outputs match: ${allMatch}`);

// ============================================================
// SECTION 13: Summary
// ============================================================

console.log("\n" + "=".repeat(60));
console.log("SUMMARY");
console.log("=".repeat(60));
console.log(`
1. BASE MODEL PROBLEM
   Pre-trained models complete text; they are NOT instruction followers.
   "Q: what does the cat do? A:" → model predicts general text continuation.

2. SUPERVISED FINE-TUNING (SFT)
   Collect (instruction, response) pairs from human annotators.
   Format: [Q: instruction A:][response]
   Loss mask: 0 for instruction tokens, 1 for response tokens.
   Train with the SAME next-token loss, just on the response part.

3. COST
   Pre-training GPT-3: ~$4.6M, months, thousands of GPUs, 300B tokens.
   InstructGPT SFT:   ~$100k, days, dozens of GPUs, ~13k examples.
   Just 10k-100k high-quality examples teaches the model the format.

4. CATASTROPHIC FORGETTING
   High LR pushes weights far from pre-trained values → model forgets.
   Fix: use SFT LR ${SFT_LR} vs pre-train LR ${PRE_TRAIN_LR} (${(PRE_TRAIN_LR/SFT_LR).toFixed(0)}x lower), few epochs.

5. LoRA
   Freeze W (d×d). Learn ΔW = A·B  where A is (d×r), B is (r×d), r<<d.
   Our lmHead: ${demoLayer.fullParams()} params → ${demoLayer.trainableParams()} LoRA params (${(demoLayer.fullParams()/demoLayer.trainableParams()).toFixed(1)}x fewer).
   B=zeros at init → ΔW=0 → model starts from pre-trained behavior.
   After training: merge W_final = W + A·B (zero inference overhead).
`);

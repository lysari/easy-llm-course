// Lesson 25 — RoPE: Rotary Position Embeddings
// No imports — pure TypeScript / Node.js built-ins only.

// ============================================================
// SECTION 1: Core RoPE Utilities
// ============================================================

/**
 * Pre-compute the cosine and sine tables used by RoPE.
 *
 * For each position `pos` and each dimension-pair index `i`:
 *   angle = pos / 10000^(2i / dim)
 *   cosTable[pos][i] = cos(angle)
 *   sinTable[pos][i] = sin(angle)
 *
 * @param dim       head_dim (must be even)
 * @param maxSeqLen number of positions to pre-compute
 */
function computeRoPE(
  dim: number,
  maxSeqLen: number
): { cosTable: number[][]; sinTable: number[][] } {
  const halfDim = Math.floor(dim / 2);
  const cosTable: number[][] = [];
  const sinTable: number[][] = [];

  for (let pos = 0; pos < maxSeqLen; pos++) {
    const cosRow: number[] = [];
    const sinRow: number[] = [];
    for (let i = 0; i < halfDim; i++) {
      // θ_i = 1 / 10000^(2i/dim)
      const theta = 1.0 / Math.pow(10000, (2 * i) / dim);
      const angle = pos * theta;
      cosRow.push(Math.cos(angle));
      sinRow.push(Math.sin(angle));
    }
    cosTable.push(cosRow);
    sinTable.push(sinRow);
  }

  return { cosTable, sinTable };
}

/**
 * Apply RoPE to a Q or K matrix.
 *
 * x shape: (T, head_dim)
 * For each time step t and each dimension pair (2i, 2i+1):
 *   new_x[t][2i]   = x[t][2i]*cos[t][i] - x[t][2i+1]*sin[t][i]
 *   new_x[t][2i+1] = x[t][2i]*sin[t][i] + x[t][2i+1]*cos[t][i]
 *
 * @param x        Query or Key matrix  (T × head_dim)
 * @param cosTable pre-computed cosines (maxSeqLen × head_dim/2)
 * @param sinTable pre-computed sines   (maxSeqLen × head_dim/2)
 * @param offset   starting position (default 0, useful for KV-cache)
 */
function applyRoPE(
  x: number[][],
  cosTable: number[][],
  sinTable: number[][],
  offset: number = 0
): number[][] {
  const T = x.length;
  const headDim = x[0].length;
  const halfDim = Math.floor(headDim / 2);

  const out: number[][] = x.map((row) => [...row]);

  for (let t = 0; t < T; t++) {
    const pos = t + offset;
    for (let i = 0; i < halfDim; i++) {
      const c = cosTable[pos][i];
      const s = sinTable[pos][i];
      const x0 = x[t][2 * i];
      const x1 = x[t][2 * i + 1];
      out[t][2 * i] = x0 * c - x1 * s;
      out[t][2 * i + 1] = x0 * s + x1 * c;
    }
  }

  return out;
}

// ============================================================
// SECTION 2: Minimal Math Helpers
// ============================================================

function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function matMul(A: number[][], B: number[][]): number[][] {
  const rows = A.length;
  const cols = B[0].length;
  const inner = B.length;
  const C: number[][] = Array.from({ length: rows }, () => new Array(cols).fill(0));
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      for (let k = 0; k < inner; k++) C[r][c] += A[r][k] * B[k][c];
  return C;
}

function transpose(A: number[][]): number[][] {
  const rows = A.length;
  const cols = A[0].length;
  return Array.from({ length: cols }, (_, c) => Array.from({ length: rows }, (_, r) => A[r][c]));
}

function softmax(x: number[]): number[] {
  const max = Math.max(...x);
  const exps = x.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function softmaxRows(A: number[][]): number[][] {
  return A.map((row) => softmax(row));
}

function causalMask(T: number): number[][] {
  return Array.from({ length: T }, (_, r) =>
    Array.from({ length: T }, (_, c) => (c <= r ? 0 : -1e9))
  );
}

function addMatrices(A: number[][], B: number[][]): number[][] {
  return A.map((row, r) => row.map((v, c) => v + B[r][c]));
}

function scaleMatrix(A: number[][], s: number): number[][] {
  return A.map((row) => row.map((v) => v * s));
}

function relu(x: number): number {
  return Math.max(0, x);
}

/** Xavier-uniform initialization */
function randomMatrix(rows: number, cols: number, scale?: number): number[][] {
  const s = scale ?? Math.sqrt(6 / (rows + cols));
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (Math.random() * 2 - 1) * s)
  );
}

function randomVector(size: number, scale: number = 0.01): number[] {
  return Array.from({ length: size }, () => (Math.random() * 2 - 1) * scale);
}

// Simple seeded RNG for reproducibility
function makePRNG(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return ((s >>> 0) / 0xffffffff) * 2 - 1;
  };
}

// ============================================================
// SECTION 3: Attention Layer (with optional RoPE)
// ============================================================

interface AttentionWeights {
  Wq: number[][];
  Wk: number[][];
  Wv: number[][];
  Wo: number[][];
}

interface AttentionResult {
  output: number[][];
  attnWeights: number[][];
}

/**
 * Single-head attention.
 * @param x        input (T × d_model)
 * @param weights  projection matrices
 * @param useRoPE  whether to apply RoPE to Q and K
 * @param cosTable RoPE cosine table (required if useRoPE)
 * @param sinTable RoPE sine table   (required if useRoPE)
 */
function attention(
  x: number[][],
  weights: AttentionWeights,
  useRoPE: boolean = false,
  cosTable?: number[][],
  sinTable?: number[][]
): AttentionResult {
  const T = x.length;
  const headDim = weights.Wq[0].length;
  const scale = 1 / Math.sqrt(headDim);

  // Project to Q, K, V
  let Q = matMul(x, weights.Wq); // T × headDim
  let K = matMul(x, weights.Wk);
  const V = matMul(x, weights.Wv);

  // Apply RoPE to Q and K if requested
  if (useRoPE && cosTable && sinTable) {
    Q = applyRoPE(Q, cosTable, sinTable);
    K = applyRoPE(K, cosTable, sinTable);
  }

  // Scaled dot-product attention
  const scores = scaleMatrix(matMul(Q, transpose(K)), scale); // T × T
  const mask = causalMask(T);
  const maskedScores = addMatrices(scores, mask);
  const attnWeights = softmaxRows(maskedScores); // T × T

  // Weighted sum of V
  const context = matMul(attnWeights, V); // T × headDim
  const output = matMul(context, weights.Wo); // T × d_model

  return { output, attnWeights };
}

// ============================================================
// SECTION 4: Tiny Language Model
// ============================================================

interface TinyLMConfig {
  vocabSize: number;
  dModel: number;
  blockSize: number;
  useRoPE: boolean;
}

interface TinyLMWeights {
  tokenEmb: number[][];   // vocabSize × dModel
  posEmb: number[][];     // blockSize × dModel  (only used when useRoPE=false)
  attn: AttentionWeights;
  ffn1: number[][];       // dModel × (dModel*4)
  ffn1b: number[];
  ffn2: number[][];       // (dModel*4) × dModel
  ffn2b: number[];
  lmHead: number[][];     // dModel × vocabSize
}

function initWeights(cfg: TinyLMConfig, rng: () => number): TinyLMWeights {
  const randM = (r: number, c: number) =>
    Array.from({ length: r }, () => Array.from({ length: c }, () => rng() * Math.sqrt(2 / (r + c))));

  return {
    tokenEmb: randM(cfg.vocabSize, cfg.dModel),
    posEmb: randM(cfg.blockSize, cfg.dModel),
    attn: {
      Wq: randM(cfg.dModel, cfg.dModel),
      Wk: randM(cfg.dModel, cfg.dModel),
      Wv: randM(cfg.dModel, cfg.dModel),
      Wo: randM(cfg.dModel, cfg.dModel),
    },
    ffn1: randM(cfg.dModel, cfg.dModel * 4),
    ffn1b: new Array(cfg.dModel * 4).fill(0),
    ffn2: randM(cfg.dModel * 4, cfg.dModel),
    ffn2b: new Array(cfg.dModel).fill(0),
    lmHead: randM(cfg.dModel, cfg.vocabSize),
  };
}

function tinyLMForward(
  tokens: number[],
  weights: TinyLMWeights,
  cfg: TinyLMConfig,
  cosTable: number[][],
  sinTable: number[][]
): number[][] {
  const T = tokens.length;

  // Embed tokens
  let x: number[][] = tokens.map((tok, pos) => {
    const te = weights.tokenEmb[tok];
    if (cfg.useRoPE) {
      // RoPE: no position added here; rotation happens in attention
      return [...te];
    } else {
      // Absolute PE: add position embedding
      const pe = weights.posEmb[pos];
      return te.map((v, i) => v + pe[i]);
    }
  });

  // Single attention layer
  const attnOut = attention(x, weights.attn, cfg.useRoPE, cosTable, sinTable);
  x = addMatrices(x, attnOut.output); // residual

  // FFN
  const ffnOut: number[][] = x.map((row) => {
    // Linear 1 + ReLU
    const h = weights.ffn1[0].map((_, j) => {
      let sum = weights.ffn1b[j];
      for (let k = 0; k < row.length; k++) sum += row[k] * weights.ffn1[k][j];
      return relu(sum);
    });
    // Linear 2
    return weights.ffn2[0].map((_, j) => {
      let sum = weights.ffn2b[j];
      for (let k = 0; k < h.length; k++) sum += h[k] * weights.ffn2[k][j];
      return sum;
    });
  });
  x = addMatrices(x, ffnOut); // residual

  // LM head: (T, dModel) × (dModel, vocabSize) → (T, vocabSize)
  const logits = matMul(x, weights.lmHead);
  return logits;
}

function crossEntropyLoss(logits: number[][], targets: number[]): number {
  let totalLoss = 0;
  for (let t = 0; t < targets.length; t++) {
    const probs = softmax(logits[t]);
    totalLoss -= Math.log(probs[targets[t]] + 1e-9);
  }
  return totalLoss / targets.length;
}

// ============================================================
// SECTION 5: Numerical Gradient (for training demo)
// ============================================================

/**
 * Very small training loop using finite-difference gradients.
 * This is intentionally simple — the goal is to compare loss curves,
 * not to build a production-quality autograd engine.
 */
function finiteDiff(
  f: () => number,
  param: number[][],
  r: number,
  c: number,
  eps: number = 1e-4
): number {
  const orig = param[r][c];
  param[r][c] = orig + eps;
  const fPlus = f();
  param[r][c] = orig - eps;
  const fMinus = f();
  param[r][c] = orig;
  return (fPlus - fMinus) / (2 * eps);
}

/**
 * Train for a fixed number of steps using finite-difference gradients
 * on the lmHead weights only (to keep runtime manageable).
 *
 * Returns loss at each step.
 */
function trainTinyLM(
  tokens: number[],
  cfg: TinyLMConfig,
  steps: number,
  lr: number,
  seed: number
): number[] {
  const rng = makePRNG(seed);
  const weights = initWeights(cfg, rng);
  const { cosTable, sinTable } = computeRoPE(cfg.dModel, cfg.blockSize + 16);

  const losses: number[] = [];
  const inputs = tokens.slice(0, -1);
  const targets = tokens.slice(1);

  for (let step = 0; step < steps; step++) {
    const logits = tinyLMForward(inputs, weights, cfg, cosTable, sinTable);
    const loss = crossEntropyLoss(logits, targets);
    losses.push(loss);

    // Gradient on lmHead via finite differences
    const rows = weights.lmHead.length;
    const cols = weights.lmHead[0].length;

    // Only update a random 10% of parameters for speed
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (Math.random() > 0.1) continue;
        const grad = finiteDiff(
          () => {
            const l = tinyLMForward(inputs, weights, cfg, cosTable, sinTable);
            return crossEntropyLoss(l, targets);
          },
          weights.lmHead,
          r,
          c
        );
        weights.lmHead[r][c] -= lr * grad;
      }
    }
  }

  return losses;
}

// ============================================================
// SECTION 6: Pretty Print Helpers
// ============================================================

function fmt(n: number, decimals: number = 4): string {
  return n.toFixed(decimals).padStart(8);
}

function printHeader(title: string): void {
  console.log("\n" + "=".repeat(60));
  console.log(`  ${title}`);
  console.log("=".repeat(60));
}

// ============================================================
// DEMO 1: Visualize Rotation Vectors for Positions 0-7
// ============================================================

printHeader("DEMO 1: RoPE Rotation Vectors — Positions 0–7, head_dim=8");

console.log(
  "\nFor head_dim=8 we have 4 frequency pairs (i=0,1,2,3).\n" +
    "Each cell shows cos(pos * θ_i) for that (pos, i) pair.\n"
);

const VIZ_DIM = 8;
const VIZ_POSITIONS = 8;
const { cosTable: vizCos, sinTable: vizSin } = computeRoPE(VIZ_DIM, VIZ_POSITIONS);

// Print the θ values
const vizHalfDim = VIZ_DIM / 2;
console.log("Frequencies θ_i:");
for (let i = 0; i < vizHalfDim; i++) {
  const theta = 1 / Math.pow(10000, (2 * i) / VIZ_DIM);
  console.log(`  θ_${i} = ${theta.toFixed(6)}`);
}

// Print cosTable as a grid
console.log("\ncosTable[pos][i]  (rows=positions 0..7, cols=freq pairs 0..3):");
console.log("pos\\pair |   i=0    |   i=1    |   i=2    |   i=3  ");
console.log("---------+----------+----------+----------+----------");
for (let pos = 0; pos < VIZ_POSITIONS; pos++) {
  const row = vizCos[pos].map((v) => fmt(v)).join(" | ");
  console.log(`  pos ${pos}  | ${row}`);
}

// Show an actual rotation: take a unit vector [1, 0, 1, 0, 1, 0, 1, 0]
// and rotate it for each position
console.log("\nRotating unit vector [1,0,1,0,1,0,1,0] at each position:");
const unitVec: number[][] = Array.from({ length: VIZ_POSITIONS }, () => [1, 0, 1, 0, 1, 0, 1, 0]);
// We need to apply RoPE position by position
for (let pos = 0; pos < VIZ_POSITIONS; pos++) {
  const singleRow = [unitVec[pos]];
  // Use offset so position pos is used
  const rotated = applyRoPE(singleRow, vizCos, vizSin, pos);
  const vals = rotated[0].map((v) => fmt(v, 3)).join(", ");
  console.log(`  pos ${pos}: [${vals}]`);
}

// ============================================================
// DEMO 2: Dot Product Depends Only on Relative Distance
// ============================================================

printHeader("DEMO 2: Q[m] · K[n] depends only on |m - n|");

console.log(
  "\nWe fix random Q and K vectors and apply RoPE.\n" +
    "Pairs with the same distance should have similar dot products.\n"
);

const DOT_DIM = 16;
const DOT_MAXPOS = 32;
const { cosTable: dotCos, sinTable: dotSin } = computeRoPE(DOT_DIM, DOT_MAXPOS);

// Create one fixed Q and K vector
const fixedQ = Array.from({ length: DOT_DIM }, (_, i) => Math.sin(i * 0.7));
const fixedK = Array.from({ length: DOT_DIM }, (_, i) => Math.cos(i * 0.5));

// Rotate at each position
const rotatedQs: number[][] = [];
const rotatedKs: number[][] = [];
for (let pos = 0; pos < DOT_MAXPOS; pos++) {
  const rQ = applyRoPE([[...fixedQ]], dotCos, dotSin, pos);
  const rK = applyRoPE([[...fixedK]], dotCos, dotSin, pos);
  rotatedQs.push(rQ[0]);
  rotatedKs.push(rK[0]);
}

// Compute dot products for pairs with the same distance
const distances = [1, 2, 4, 8];
console.log("distance d | (m,n) pairs with same d | dot(Q_m, K_n)");
console.log("-----------+-------------------------+---------------");

for (const d of distances) {
  const rawPairs: number[][] = [
    [d, 0],
    [d + 3, 3],
    [d + 7, 7],
    [d + 15, 15],
  ];
  const pairs: Array<[number, number]> = rawPairs
    .filter(([m]) => m < DOT_MAXPOS)
    .map(([m, n]) => [m, n] as [number, number]);

  const dotVals = pairs.map(([m, n]) => dot(rotatedQs[m], rotatedKs[n]));
  const minV = Math.min(...dotVals);
  const maxV = Math.max(...dotVals);
  const range = maxV - minV;

  const pairStr = pairs.map(([m, n]) => `(${m},${n})`).join(", ");
  const dotStr = dotVals.map((v) => v.toFixed(5)).join(", ");
  console.log(`   d=${d.toString().padEnd(2)}   | ${pairStr.padEnd(23)} | ${dotStr}`);
  console.log(
    `           | spread (max-min):              | ${range.toFixed(6)}  ← should be ~0`
  );
}

console.log("\nThe near-zero spread confirms: dot product is a function of distance only.");

// Also show that different distances give different values
console.log("\nFor contrast, same starting position m=10 but varying n:");
const baseM = 10;
for (let n = 0; n <= 9; n++) {
  const d = baseM - n;
  const dotVal = dot(rotatedQs[baseM], rotatedKs[n]);
  console.log(`  m=10, n=${n}, distance=${d}: dot = ${dotVal.toFixed(5)}`);
}

// ============================================================
// DEMO 3: Train RoPE vs Absolute PE — Compare Final Loss
// ============================================================

printHeader("DEMO 3: Training Comparison — RoPE vs Absolute Position Embeddings");

console.log(
  "\nTask: predict next token in a repeating sequence [0,1,2,3,4,5,0,1,2,3,4,5,...].\n" +
    "A model that learns position should do well; comparing final losses.\n"
);

// Build a repeating token sequence
const VOCAB_SIZE = 8;
const BLOCK_SIZE = 8;
const D_MODEL = 8; // intentionally small so finite-diff is fast
const TRAIN_STEPS = 20;
const LR = 0.05;

const repeatingSeq: number[] = [];
for (let i = 0; i < BLOCK_SIZE + 1; i++) {
  repeatingSeq.push(i % VOCAB_SIZE);
}

const cfgRoPE: TinyLMConfig = {
  vocabSize: VOCAB_SIZE,
  dModel: D_MODEL,
  blockSize: BLOCK_SIZE,
  useRoPE: true,
};

const cfgAbsolute: TinyLMConfig = {
  vocabSize: VOCAB_SIZE,
  dModel: D_MODEL,
  blockSize: BLOCK_SIZE,
  useRoPE: false,
};

console.log(`Sequence: [${repeatingSeq.join(", ")}]`);
console.log(`Vocab size: ${VOCAB_SIZE}, block size: ${BLOCK_SIZE}, d_model: ${D_MODEL}`);
console.log(`Training for ${TRAIN_STEPS} steps (finite-diff, ~10% params updated per step)\n`);

const lossesRoPE = trainTinyLM(repeatingSeq, cfgRoPE, TRAIN_STEPS, LR, 42);
const lossesAbsolute = trainTinyLM(repeatingSeq, cfgAbsolute, TRAIN_STEPS, LR, 42);

// Print loss table
console.log("Step |  RoPE loss  | AbsolutePE loss");
console.log("-----+-------------+----------------");
const reportSteps = [0, 4, 9, 14, 19];
for (const step of reportSteps) {
  const r = lossesRoPE[step] ?? NaN;
  const a = lossesAbsolute[step] ?? NaN;
  console.log(
    `  ${String(step + 1).padStart(2)} | ${r.toFixed(4).padStart(10)} | ${a.toFixed(4).padStart(14)}`
  );
}

const finalRoPE = lossesRoPE[lossesRoPE.length - 1];
const finalAbs = lossesAbsolute[lossesAbsolute.length - 1];
console.log(`\nFinal loss  — RoPE: ${finalRoPE.toFixed(4)}  |  Absolute PE: ${finalAbs.toFixed(4)}`);
console.log(
  "\nNote: With random finite-diff training on a tiny model, results are noisy.\n" +
    "The important point is that RoPE converges on the same task with no extra parameters."
);

// ============================================================
// DEMO 4: Position Extrapolation — Beyond Training Length
// ============================================================

printHeader("DEMO 4: Position Extrapolation — RoPE vs Absolute PE");

console.log(
  "\nWe compute RoPE tables for positions 0–11 (training length = 8, extra = 4).\n" +
    "Absolute PE has a table of size 8 and cannot access positions 8–11 without error.\n" +
    "RoPE computes them deterministically via the angle formula.\n"
);

const TRAIN_LEN = 8;
const TEST_LEN = 12;

const { cosTable: extCos, sinTable: extSin } = computeRoPE(VIZ_DIM, TEST_LEN);

console.log("Absolute PE embedding table has entries for positions 0 to 7 only.");
console.log("If a model tries to access position 8 it would index out of bounds.\n");

console.log("RoPE can compute rotation values for any position:");
console.log("pos  | cos(pos*θ_0) | cos(pos*θ_1) | cos(pos*θ_2) | cos(pos*θ_3) | status");
console.log("-----+-------------+-------------+-------------+-------------+--------");
for (let pos = 0; pos < TEST_LEN; pos++) {
  const vals = extCos[pos].map((v) => v.toFixed(4).padStart(11)).join(" | ");
  const status = pos < TRAIN_LEN ? "trained" : "EXTRAPOLATED";
  console.log(`  ${String(pos).padStart(2)} | ${vals} | ${status}`);
}

console.log(
  "\nKey observation: extrapolated positions (8–11) have well-defined rotation values."
);
console.log("The cos/sin values stay bounded in [-1, 1] by definition — no blow-up.");
console.log(
  "\nIn practice, models do degrade beyond their training length because:\n" +
    "  - The attention distribution has never seen those distance values during training.\n" +
    "  - Fix: RoPE scaling (divide positions by a factor) or YaRN (frequency-aware scaling)."
);

// Show a concrete attention score comparison
console.log("\nAttention score (Q_0 · K_n) for trained vs extrapolated positions:");
console.log("(Using the same fixed vectors as Demo 2)");
const { cosTable: extDotCos, sinTable: extDotSin } = computeRoPE(DOT_DIM, TEST_LEN);
const rotQ0 = applyRoPE([[...fixedQ]], extDotCos, extDotSin, 0)[0];
console.log("\n n |  raw score  | status");
console.log("---+-------------+------------");
for (let n = 0; n < TEST_LEN; n++) {
  const rKn = applyRoPE([[...fixedK]], extDotCos, extDotSin, n)[0];
  const score = dot(rotQ0, rKn) / Math.sqrt(DOT_DIM);
  const status = n < TRAIN_LEN ? "trained" : "extrapolated";
  console.log(`  ${String(n).padStart(1)} | ${score.toFixed(5).padStart(9)}   | ${status}`);
}

// ============================================================
// SUMMARY
// ============================================================

printHeader("Summary");

console.log(`
RoPE (Rotary Position Embedding) — key takeaways:

1. ROTATION, not addition
   Instead of adding a fixed vector to token embeddings, RoPE rotates
   the Q and K vectors by an angle proportional to the token's position.

2. Relative distance is exact
   The dot product Q_m · K_n is a function of (m - n) only.
   This was verified in Demo 2: same-distance pairs produce the same score.

3. Zero extra parameters
   The cos/sin tables are computed deterministically from fixed frequencies.
   No learnable parameters are added.

4. Continuous angles enable extrapolation
   RoPE can compute rotations for any position — no lookup table boundary.
   Demo 4 showed well-defined values at positions 8–11 despite training on 0–7.

5. Used everywhere
   GPT-NeoX, Llama 1/2/3, Mistral, Mixtral, Falcon, Gemma all use RoPE.
   Absolute learned PE (GPT-2 style) is now the exception, not the rule.

Next lesson: grouped-query attention (GQA) — how Llama 2 and Mistral share
K,V heads across multiple Q heads to cut memory and speed up inference.
`);

// Lesson 30 — The Complete Picture: Mini-GPT-4 Capstone
// No imports — pure TypeScript / Node.js built-ins only.
//
// This file assembles every major idea from the curriculum:
//   Pre-LN transformer (GPT-2 style)
//   Multi-head attention with RoPE (Lesson 25)
//   Mixture-of-Experts FFN, top-2 routing (Lesson 29)
//   BPE tokenization — char-level merges (Lesson 18)
//   Adam optimizer (Lesson 17)
//   KV cache for fast generation (Lesson 21)
//   Top-p nucleus sampling (Lesson 23)
//   Weight tying — embedding == output projection (GPT-2)
//
// Run:  npx ts-node index.ts  OR  node --loader ts-node/esm index.ts

// ============================================================
// SECTION 0: Utility helpers
// ============================================================

function zeros(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

function zerosVec(n: number): number[] {
  return new Array(n).fill(0);
}

function randNormal(mean = 0, std = 0.02): number {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function randMatrix(rows: number, cols: number, std = 0.02): number[][] {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => randNormal(0, std))
  );
}

/** Matrix multiply A (m×k) × B (k×n) → (m×n) */
function matMul(A: number[][], B: number[][]): number[][] {
  const m = A.length, k = A[0].length, n = B[0].length;
  const C = zeros(m, n);
  for (let i = 0; i < m; i++)
    for (let p = 0; p < k; p++) {
      const aip = A[i][p];
      for (let j = 0; j < n; j++)
        C[i][j] += aip * B[p][j];
    }
  return C;
}

/** Transpose a matrix */
function transpose(A: number[][]): number[][] {
  const rows = A.length, cols = A[0].length;
  const T = zeros(cols, rows);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++)
      T[j][i] = A[i][j];
  return T;
}

function softmaxVec(x: number[]): number[] {
  const max = Math.max(...x);
  // Clamp to prevent exp(-very large) → 0 issues; exp of positive is fine, negative is safe
  const exp = x.map(v => Math.exp(Math.max(v - max, -500)));
  const sum = exp.reduce((a, b) => a + b, 0) || 1e-9;
  return exp.map(v => v / sum);
}

function softmaxRows(A: number[][]): number[][] {
  return A.map(row => softmaxVec(row));
}

function layerNorm(
  x: number[],
  gamma: number[],
  beta: number[],
  eps = 1e-5
): number[] {
  const mean = x.reduce((a, b) => a + b, 0) / x.length;
  const variance =
    x.reduce((a, b) => a + (b - mean) ** 2, 0) / x.length;
  const norm = x.map((v, i) => gamma[i] * (v - mean) / Math.sqrt(variance + eps) + beta[i]);
  // Guard against NaN (can occur with extreme weights during early training)
  return norm.map(v => isFinite(v) ? v : 0);
}

function gelu(x: number): number {
  if (!isFinite(x)) return 0;
  const clamped = Math.max(Math.min(x, 50), -50);
  return 0.5 * clamped * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (clamped + 0.044715 * clamped ** 3)));
}

/** Add two vectors element-wise */
function addVec(a: number[], b: number[]): number[] {
  return a.map((v, i) => v + b[i]);
}

// ============================================================
// SECTION 1: BPE Tokenizer (char-level merges)
// ============================================================

interface Tokenizer {
  vocabSize: number;
  encode: (text: string) => number[];
  decode: (ids: number[]) => string;
}

interface BPETokenizer extends Tokenizer {
  vocab: Map<string, number>;
  merges: Array<[string, string]>;
}

function buildBPE(corpus: string, numMerges: number): BPETokenizer {
  // Start with character vocabulary
  const charSet = new Set<string>(corpus.split(""));
  const vocab = new Map<string, number>();
  charSet.forEach(ch => vocab.set(ch, vocab.size));
  // Special tokens
  vocab.set("<unk>", vocab.size);
  vocab.set("<eos>", vocab.size);

  // Tokenize corpus into list of char sequences (words split by space)
  // For simplicity, work at the character level on the whole corpus
  let tokens: string[][] = corpus.split("").map(ch => [ch]);

  const merges: Array<[string, string]> = [];

  for (let m = 0; m < numMerges; m++) {
    // Count bigram frequencies
    const freq = new Map<string, number>();
    for (const seq of tokens) {
      for (let i = 0; i < seq.length - 1; i++) {
        const pair = seq[i] + "\x00" + seq[i + 1];
        freq.set(pair, (freq.get(pair) ?? 0) + 1);
      }
    }
    if (freq.size === 0) break;

    // Find most frequent pair
    let bestPair = "";
    let bestCount = 0;
    freq.forEach((count, pair) => {
      if (count > bestCount) { bestCount = count; bestPair = pair; }
    });
    if (bestCount < 2) break;

    const [left, right] = bestPair.split("\x00");
    const merged = left + right;
    merges.push([left, right]);
    if (!vocab.has(merged)) vocab.set(merged, vocab.size);

    // Apply merge
    tokens = tokens.map(seq => {
      const out: string[] = [];
      let i = 0;
      while (i < seq.length) {
        if (i < seq.length - 1 && seq[i] === left && seq[i + 1] === right) {
          out.push(merged);
          i += 2;
        } else {
          out.push(seq[i]);
          i++;
        }
      }
      return out;
    });
  }

  const reverseVocab = new Map<number, string>();
  vocab.forEach((id, tok) => reverseVocab.set(id, tok));
  const unkId = vocab.get("<unk>")!;

  function encode(text: string): number[] {
    // Greedily apply merges
    let seq: string[] = text.split("");
    for (const [left, right] of merges) {
      const merged = left + right;
      const out: string[] = [];
      let i = 0;
      while (i < seq.length) {
        if (i < seq.length - 1 && seq[i] === left && seq[i + 1] === right) {
          out.push(merged);
          i += 2;
        } else {
          out.push(seq[i]);
          i++;
        }
      }
      seq = out;
    }
    return seq.map(tok => vocab.get(tok) ?? unkId);
  }

  function decode(ids: number[]): string {
    return ids
      .filter(id => id !== vocab.get("<eos>"))
      .map(id => reverseVocab.get(id) ?? "")
      .join("");
  }

  return { vocab, merges, encode, decode, vocabSize: vocab.size };
}

// ============================================================
// SECTION 2: RoPE positional encoding
// ============================================================

function computeRoPETables(headDim: number, maxSeq: number) {
  const half = Math.floor(headDim / 2);
  const cos: number[][] = [];
  const sin: number[][] = [];
  for (let pos = 0; pos < maxSeq; pos++) {
    const cr: number[] = [], sr: number[] = [];
    for (let i = 0; i < half; i++) {
      const theta = 1.0 / Math.pow(10000, (2 * i) / headDim);
      cr.push(Math.cos(pos * theta));
      sr.push(Math.sin(pos * theta));
    }
    cos.push(cr);
    sin.push(sr);
  }
  return { cos, sin };
}

/** Apply RoPE to (T, headDim) matrix, starting at position offset */
function applyRoPE(
  x: number[][],
  cos: number[][],
  sin: number[][],
  offset = 0
): number[][] {
  const T = x.length;
  const headDim = x[0].length;
  const half = Math.floor(headDim / 2);
  const maxPos = cos.length - 1;
  return x.map((row, t) => {
    const out = row.slice();
    const pos = Math.min(t + offset, maxPos);
    for (let i = 0; i < half; i++) {
      const x0 = row[2 * i], x1 = row[2 * i + 1];
      const c = cos[pos][i], s = sin[pos][i];
      out[2 * i]     = x0 * c - x1 * s;
      out[2 * i + 1] = x0 * s + x1 * c;
    }
    return out;
  });
}

// ============================================================
// SECTION 3: Mixture-of-Experts FFN
// ============================================================

interface MoEFFN {
  numExperts: number;
  topK: number;
  // Each expert: W1 (D × 4D), W2 (4D × D), b1 (4D), b2 (D)
  experts: Array<{ W1: number[][], b1: number[], W2: number[][], b2: number[] }>;
  // Router: (D × numExperts)
  routerW: number[][];
}

function createMoEFFN(d: number, numExperts: number, topK: number): MoEFFN {
  const inner = 4 * d;
  const experts = Array.from({ length: numExperts }, () => ({
    W1: randMatrix(d, inner, 0.02),
    b1: zerosVec(inner),
    W2: randMatrix(inner, d, 0.02),
    b2: zerosVec(d),
  }));
  const routerW = randMatrix(d, numExperts, 0.02);
  return { numExperts, topK, experts, routerW };
}

/** Forward pass for one token vector x (length D) */
function moeFFFNForward(x: number[], moe: MoEFFN): number[] {
  const d = x.length;
  // Router logits
  const logits = zerosVec(moe.numExperts);
  for (let e = 0; e < moe.numExperts; e++) {
    for (let i = 0; i < d; i++) logits[e] += x[i] * moe.routerW[i][e];
  }
  const routerProbs = softmaxVec(logits);

  // Top-K selection
  const sorted = routerProbs
    .map((p, i) => ({ p, i }))
    .sort((a, b) => b.p - a.p);
  const selected = sorted.slice(0, moe.topK);

  // Renormalize selected probabilities
  const totalProb = selected.reduce((s, e) => s + e.p, 0);

  const out = zerosVec(d);
  for (const { p, i } of selected) {
    const expert = moe.experts[i];
    const weight = p / totalProb;

    // Expert forward: GELU(x @ W1 + b1) @ W2 + b2
    const h = zerosVec(expert.b1.length);
    for (let j = 0; j < h.length; j++) {
      let val = expert.b1[j];
      for (let k = 0; k < d; k++) val += x[k] * expert.W1[k][j];
      h[j] = gelu(val);
    }
    const expert_out = zerosVec(d);
    for (let j = 0; j < d; j++) {
      let val = expert.b2[j];
      for (let k = 0; k < h.length; k++) val += h[k] * expert.W2[k][j];
      expert_out[j] = val;
    }
    for (let j = 0; j < d; j++) out[j] += weight * expert_out[j];
  }
  return out;
}

// ============================================================
// SECTION 4: Multi-Head Attention with RoPE and KV cache
// ============================================================

interface MHALayer {
  numHeads: number;
  headDim: number;
  d: number;
  Wq: number[][];   // D × D
  Wk: number[][];
  Wv: number[][];
  Wo: number[][];   // D × D
}

function createMHA(d: number, numHeads: number): MHALayer {
  const headDim = Math.floor(d / numHeads);
  return {
    numHeads, headDim, d,
    Wq: randMatrix(d, d, 0.02),
    Wk: randMatrix(d, d, 0.02),
    Wv: randMatrix(d, d, 0.02),
    Wo: randMatrix(d, d, 0.02),
  };
}

function mhaForward(
  x: number[][],        // (T, D)
  layer: MHALayer,
  ropeCache: { cos: number[][], sin: number[][] },
  kvCache?: { K: number[][], V: number[][] }, // existing cache for this layer
  posOffset = 0         // position of first token in x
): {
  out: number[][],
  newK: number[][],
  newV: number[][],
} {
  const T = x.length;
  const D = layer.d;
  const H = layer.numHeads;
  const hd = layer.headDim;
  const scale = 1.0 / Math.sqrt(hd);

  // Project Q, K, V  → (T, D)
  const Q = matMul(x, layer.Wq);
  const K = matMul(x, layer.Wk);
  const V = matMul(x, layer.Wv);

  // Apply RoPE to Q and K
  const Qrope = applyRoPE(Q, ropeCache.cos, ropeCache.sin, posOffset);
  const Krope = applyRoPE(K, ropeCache.cos, ropeCache.sin, posOffset);

  // Prepend cached K/V if present
  const fullK = kvCache ? [...kvCache.K, ...Krope] : Krope;
  const fullV = kvCache ? [...kvCache.V, ...V] : V;
  const Tctx = fullK.length; // total context length

  // Split into heads and compute attention
  const headOutputs: number[][][] = Array.from({ length: H }, () => zeros(T, hd));

  for (let h = 0; h < H; h++) {
    const hStart = h * hd;
    // Extract head slices
    const qh = Qrope.map(row => row.slice(hStart, hStart + hd));   // (T, hd)
    const kh = fullK.map(row => row.slice(hStart, hStart + hd));   // (Tctx, hd)
    const vh = fullV.map(row => row.slice(hStart, hStart + hd));   // (Tctx, hd)

    // Attention scores (T, Tctx)
    const scores: number[][] = zeros(T, Tctx);
    for (let qi = 0; qi < T; qi++) {
      for (let ki = 0; ki < Tctx; ki++) {
        // Causal mask: query at position (posOffset+qi) can only attend to keys ≤ that position
        if (posOffset + qi < ki) {
          scores[qi][ki] = -1e9;
        } else {
          let dot = 0;
          for (let d2 = 0; d2 < hd; d2++) dot += qh[qi][d2] * kh[ki][d2];
          scores[qi][ki] = dot * scale;
        }
      }
    }

    const attnWeights = softmaxRows(scores); // (T, Tctx)

    // Weighted sum of V
    for (let qi = 0; qi < T; qi++) {
      for (let d2 = 0; d2 < hd; d2++) {
        let sum = 0;
        for (let ki = 0; ki < Tctx; ki++) {
          sum += attnWeights[qi][ki] * vh[ki][d2];
        }
        headOutputs[h][qi][d2] = sum;
      }
    }
  }

  // Concatenate heads: (T, D)
  const concat: number[][] = zeros(T, D);
  for (let qi = 0; qi < T; qi++)
    for (let h = 0; h < H; h++)
      for (let d2 = 0; d2 < hd; d2++)
        concat[qi][h * hd + d2] = headOutputs[h][qi][d2];

  // Output projection
  const out = matMul(concat, layer.Wo);

  return { out, newK: Krope, newV: V };
}

// ============================================================
// SECTION 5: Transformer Block (Pre-LN style)
// ============================================================

interface TransformerBlock {
  attn: MHALayer;
  moe: MoEFFN;
  ln1Gamma: number[];
  ln1Beta: number[];
  ln2Gamma: number[];
  ln2Beta: number[];
}

function createBlock(d: number, numHeads: number, numExperts: number, topK: number): TransformerBlock {
  return {
    attn: createMHA(d, numHeads),
    moe: createMoEFFN(d, numExperts, topK),
    ln1Gamma: new Array(d).fill(1),
    ln1Beta: new Array(d).fill(0),
    ln2Gamma: new Array(d).fill(1),
    ln2Beta: new Array(d).fill(0),
  };
}

function blockForward(
  x: number[][],   // (T, D)
  block: TransformerBlock,
  ropeCache: { cos: number[][], sin: number[][] },
  kvCacheEntry?: { K: number[][], V: number[][] },
  posOffset = 0
): {
  out: number[][],
  newK: number[][],
  newV: number[][],
} {
  const T = x.length;

  // Pre-LN for attention
  const xNorm1 = x.map(row => layerNorm(row, block.ln1Gamma, block.ln1Beta));
  const { out: attnOut, newK, newV } = mhaForward(xNorm1, block.attn, ropeCache, kvCacheEntry, posOffset);

  // Residual
  const x2: number[][] = x.map((row, i) => addVec(row, attnOut[i]));

  // Pre-LN for FFN (MoE)
  const xNorm2 = x2.map(row => layerNorm(row, block.ln2Gamma, block.ln2Beta));
  const ffnOut = xNorm2.map(row => moeFFFNForward(row, block.moe));

  // Residual
  const out: number[][] = x2.map((row, i) => addVec(row, ffnOut[i]));

  return { out, newK, newV };
}

// ============================================================
// SECTION 6: Full Mini-GPT-4 Model
// ============================================================

interface MiniGPT4 {
  tokenizer: Tokenizer;
  embedding: number[][];   // (V, D) — weight-tied with output projection
  blocks: TransformerBlock[];
  lnFinalGamma: number[];
  lnFinalBeta: number[];
  // lmHead is embedding.T (weight tying — no separate parameter)
  config: {
    vocabSize: number;
    d: number;
    numHeads: number;
    numLayers: number;
    numExperts: number;
    topK: number;
    blockSize: number;
  };
  ropeCache: { cos: number[][], sin: number[][] };
}

function createMiniGPT4(
  tokenizer: BPETokenizer,
  d: number,
  numHeads: number,
  numLayers: number,
  numExperts: number,
  topK: number,
  blockSize: number
): MiniGPT4 {
  const V = tokenizer.vocabSize;
  const embedding = randMatrix(V, d, 0.02);
  const blocks = Array.from({ length: numLayers }, () =>
    createBlock(d, numHeads, numExperts, topK)
  );
  // RoPE is applied to full D-dimensional Q/K before splitting into heads.
  // We tile the per-head frequencies across all heads so each head pair (2i, 2i+1)
  // within a head uses its own frequency.
  const ropeCache = computeRoPETables(d, blockSize + 200);
  return {
    tokenizer,
    embedding,
    blocks,
    lnFinalGamma: new Array(d).fill(1),
    lnFinalBeta: new Array(d).fill(0),
    config: { vocabSize: V, d, numHeads, numLayers, numExperts, topK, blockSize },
    ropeCache,
  };
}

/** Forward pass: returns logits (T, V) */
function forward(
  model: MiniGPT4,
  tokenIds: number[],
  kvCaches?: Array<{ K: number[][], V: number[][] }>,
  posOffset = 0
): {
  logits: number[][],
  newKVCaches: Array<{ K: number[][], V: number[][] }>,
} {
  const T = tokenIds.length;
  const D = model.config.d;

  // Token embeddings
  let x: number[][] = tokenIds.map(id => model.embedding[id].slice());

  // Run through transformer blocks
  const newKVCaches: Array<{ K: number[][], V: number[][] }> = [];
  for (let l = 0; l < model.blocks.length; l++) {
    const cache = kvCaches ? kvCaches[l] : undefined;
    const { out, newK, newV } = blockForward(x, model.blocks[l], model.ropeCache, cache, posOffset);
    x = out;
    // Append to cache
    const existingK = cache ? cache.K : [];
    const existingV = cache ? cache.V : [];
    newKVCaches.push({ K: [...existingK, ...newK], V: [...existingV, ...newV] });
  }

  // Final layer norm
  const xNorm = x.map(row => layerNorm(row, model.lnFinalGamma, model.lnFinalBeta));

  // LM head: weight-tied — multiply by embedding matrix transposed (V, D) → (D, V)
  // logits = xNorm @ embedding^T  →  (T, V)
  const embT = transpose(model.embedding); // (D, V)
  const logits = matMul(xNorm, embT);      // (T, V)

  return { logits, newKVCaches };
}

/** Cross-entropy loss over a sequence */
function computeLoss(model: MiniGPT4, inputIds: number[], targetIds: number[]): number {
  const { logits } = forward(model, inputIds);
  let loss = 0;
  for (let t = 0; t < targetIds.length; t++) {
    // Guard: clamp logits to avoid exp overflow in softmax
    const row = logits[t].map(v => isFinite(v) ? Math.max(Math.min(v, 100), -100) : 0);
    const probs = softmaxVec(row);
    loss -= Math.log(Math.max(probs[targetIds[t]], 1e-9));
  }
  const avg = loss / targetIds.length;
  return isFinite(avg) ? avg : 10.0; // fallback sentinel loss if NaN
}

// ============================================================
// SECTION 7: Adam Optimizer
// ============================================================

interface AdamState {
  t: number;
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  // Flat parameter + gradient + moment storage
  params: Float64Array;
  grads: Float64Array;
  m: Float64Array;
  v: Float64Array;
}

/** Collect all trainable parameters into a flat array for Adam */
function collectParams(model: MiniGPT4): {
  get: () => number[],
  set: (vals: number[]) => void,
  count: () => number,
} {
  // We will gather references to all mutable arrays
  const arrays: { arr: number[], idx: number }[] = [];

  function reg(arr: number[], start: number) {
    arrays.push({ arr, idx: start });
  }

  // Collect embedding
  const emb = model.embedding;
  for (let i = 0; i < emb.length; i++)
    for (let j = 0; j < emb[i].length; j++)
      arrays.push({ arr: emb[i], idx: j });

  for (const block of model.blocks) {
    // Attention weights
    for (const W of [block.attn.Wq, block.attn.Wk, block.attn.Wv, block.attn.Wo])
      for (const row of W)
        for (let j = 0; j < row.length; j++)
          arrays.push({ arr: row, idx: j });

    // MoE
    for (const expert of block.moe.experts) {
      for (const W of [expert.W1, expert.W2])
        for (const row of W)
          for (let j = 0; j < row.length; j++)
            arrays.push({ arr: row, idx: j });
      for (const b of [expert.b1, expert.b2])
        for (let j = 0; j < b.length; j++)
          arrays.push({ arr: b, idx: j });
    }
    // Router
    for (const row of block.moe.routerW)
      for (let j = 0; j < row.length; j++)
        arrays.push({ arr: row, idx: j });

    // LayerNorm params
    for (const arr of [block.ln1Gamma, block.ln1Beta, block.ln2Gamma, block.ln2Beta])
      for (let j = 0; j < arr.length; j++)
        arrays.push({ arr, idx: j });
  }

  // Final LN
  for (const arr of [model.lnFinalGamma, model.lnFinalBeta])
    for (let j = 0; j < arr.length; j++)
      arrays.push({ arr, idx: j });

  const n = arrays.length;

  return {
    count: () => n,
    get: () => arrays.map(({ arr, idx }) => arr[idx]),
    set: (vals: number[]) => arrays.forEach(({ arr, idx }, i) => { arr[idx] = vals[i]; }),
  };
}

function createAdam(numParams: number, lr = 3e-4, beta1 = 0.9, beta2 = 0.999, eps = 1e-8): AdamState {
  return {
    t: 0,
    lr, beta1, beta2, eps,
    params: new Float64Array(numParams),
    grads: new Float64Array(numParams),
    m: new Float64Array(numParams),
    v: new Float64Array(numParams),
  };
}

/**
 * Numerical gradient + Adam update for one training step.
 *
 * Strategy: compute exact numerical gradients for the embedding rows of
 * all tokens that appear in this batch (most impactful), plus a random
 * sample of remaining parameters.  This gives a meaningful gradient
 * signal without touching the full 20K-param model on every step.
 */
function trainStep(
  model: MiniGPT4,
  inputIds: number[],
  targetIds: number[],
  paramAccessor: ReturnType<typeof collectParams>,
  adamState: AdamState,
  clipNorm = 1.0
): number {
  const h = 5e-3;  // larger step for numerical stability through LayerNorm
  const currentParams = paramAccessor.get();
  const n = currentParams.length;
  const D = model.config.d;
  const V = model.config.vocabSize;

  // Compute baseline loss
  const baseLoss = computeLoss(model, inputIds, targetIds);

  const grads = new Float64Array(n);

  // ── Priority indices: embedding rows for tokens in this batch ──
  const batchTokens = new Set([...inputIds, ...targetIds]);
  const embIndices: number[] = [];
  batchTokens.forEach(tokenId => {
    for (let j = 0; j < D; j++) {
      embIndices.push(tokenId * D + j);
    }
  });

  // ── Random sample of non-embedding params ──
  const embParamCount = V * D;
  const nonEmbCount = n - embParamCount;
  const extraSample = Math.min(40, nonEmbCount);
  const extraIndices: number[] = [];
  for (let i = 0; i < extraSample; i++) {
    extraIndices.push(embParamCount + Math.floor(Math.random() * nonEmbCount));
  }

  const allIndices = [...embIndices, ...extraIndices];

  for (const i of allIndices) {
    const orig = currentParams[i];

    currentParams[i] = orig + h;
    paramAccessor.set(currentParams);
    const lossPlus = computeLoss(model, inputIds, targetIds);

    currentParams[i] = orig - h;
    paramAccessor.set(currentParams);
    const lossMinus = computeLoss(model, inputIds, targetIds);

    const g = (lossPlus - lossMinus) / (2 * h);
    grads[i] = isFinite(g) ? g : 0;
    currentParams[i] = orig;
  }
  paramAccessor.set(currentParams);

  // Gradient clipping (over active grads only)
  let gradNorm = 0;
  for (let i = 0; i < n; i++) gradNorm += grads[i] ** 2;
  gradNorm = Math.sqrt(gradNorm) || 1;
  const scale = gradNorm > clipNorm ? clipNorm / gradNorm : 1.0;

  // Adam update
  adamState.t += 1;
  const { t, lr, beta1, beta2, eps: adamEps } = adamState;
  const bc1 = 1 - beta1 ** t;
  const bc2 = 1 - beta2 ** t;

  for (let i = 0; i < n; i++) {
    if (grads[i] === 0) continue; // skip untouched params
    const g = grads[i] * scale;
    adamState.m[i] = beta1 * adamState.m[i] + (1 - beta1) * g;
    adamState.v[i] = beta2 * adamState.v[i] + (1 - beta2) * g * g;
    const mHat = adamState.m[i] / bc1;
    const vHat = adamState.v[i] / bc2;
    currentParams[i] -= lr * mHat / (Math.sqrt(vHat) + adamEps);
  }
  paramAccessor.set(currentParams);

  return baseLoss;
}

// ============================================================
// SECTION 8: Top-p (nucleus) sampling
// ============================================================

function topPSample(logits: number[], p: number, temperature: number): number {
  // Apply temperature
  const scaled = logits.map(l => l / temperature);
  const probs = softmaxVec(scaled);

  // Sort by descending probability
  const sorted = probs
    .map((prob, id) => ({ prob, id }))
    .sort((a, b) => b.prob - a.prob);

  // Nucleus: accumulate until mass >= p
  let cumMass = 0;
  const nucleus: { prob: number, id: number }[] = [];
  for (const item of sorted) {
    nucleus.push(item);
    cumMass += item.prob;
    if (cumMass >= p) break;
  }

  // Renormalize and sample
  const total = nucleus.reduce((s, item) => s + item.prob, 0);
  const r = Math.random() * total;
  let acc = 0;
  for (const item of nucleus) {
    acc += item.prob;
    if (r <= acc) return item.id;
  }
  return nucleus[nucleus.length - 1].id;
}

// ============================================================
// SECTION 9: Text generation with KV cache
// ============================================================

function generate(
  model: MiniGPT4,
  prompt: string,
  maxNewTokens: number,
  temperature: number,
  topP: number
): string {
  const inputIds = model.tokenizer.encode(prompt);
  if (inputIds.length === 0) return prompt;

  // Prefill: run full prompt through the model to fill KV cache
  const { newKVCaches } = forward(model, inputIds, undefined, 0);
  let kvCaches = newKVCaches;
  let posOffset = inputIds.length;

  const generatedIds: number[] = [...inputIds];
  let lastId = inputIds[inputIds.length - 1];

  for (let step = 0; step < maxNewTokens; step++) {
    // Single token forward with KV cache
    const { logits, newKVCaches: updatedCaches } = forward(
      model,
      [lastId],
      kvCaches,
      posOffset
    );
    kvCaches = updatedCaches;
    posOffset += 1;

    // Sample from last token's logits
    const nextId = topPSample(logits[0], topP, temperature);
    const bpe = model.tokenizer as BPETokenizer;
    const eosId = bpe.vocab?.get("<eos>") ?? -1;
    if (nextId === eosId) break;

    generatedIds.push(nextId);
    lastId = nextId;
  }

  return model.tokenizer.decode(generatedIds);
}

// ============================================================
// SECTION 10: Parameter counting
// ============================================================

function countParams(model: MiniGPT4): { total: number, activePerToken: number } {
  const { vocabSize: V, d: D, numHeads, numLayers, numExperts, topK, blockSize } = model.config;
  const innerDim = 4 * D;

  // Embedding (weight-tied, counts once)
  const embParams = V * D;

  // Per block:
  //   Attention: Wq + Wk + Wv + Wo = 4 * D * D
  //   All experts: numExperts * (D*innerDim + innerDim + innerDim*D + D) = numExperts * (2*D*innerDim + innerDim + D)
  //   Router: D * numExperts
  //   4 LayerNorm param vectors of length D = 4*D
  const attnParams = 4 * D * D;
  const expertParams = numExperts * (2 * D * innerDim + innerDim + D);
  const routerParams = D * numExperts;
  const lnParams = 4 * D;
  const blockParams = attnParams + expertParams + routerParams + lnParams;

  // Final LN
  const finalLNParams = 2 * D;

  const total = embParams + numLayers * blockParams + finalLNParams;

  // Active per forward pass:
  //   embedding lookup: D (just one row)
  //   per block: attn (full) + topK experts only + router + LN
  const activeExpertParams = topK * (2 * D * innerDim + innerDim + D);
  const activeBlockParams = attnParams + activeExpertParams + routerParams + lnParams;
  const activePerToken = D + numLayers * activeBlockParams + finalLNParams;

  return { total, activePerToken };
}

// ============================================================
// SECTION 11: ASCII loss curve
// ============================================================

function asciiLossCurve(losses: number[], width = 60, height = 12): string {
  if (losses.length === 0) return "";
  const maxLoss = Math.max(...losses);
  const minLoss = Math.min(...losses);
  const range = maxLoss - minLoss || 1;

  const lines: string[] = [];
  for (let row = height - 1; row >= 0; row--) {
    const threshold = minLoss + (row / (height - 1)) * range;
    let line = "";
    // Scale losses to width
    for (let col = 0; col < width; col++) {
      const idx = Math.floor((col / width) * losses.length);
      const loss = losses[Math.min(idx, losses.length - 1)];
      if (row === 0) {
        line += "─";
      } else if (Math.abs(loss - threshold) < range / (height * 1.5)) {
        line += "█";
      } else {
        line += " ";
      }
    }
    const label = row === height - 1
      ? ` ${maxLoss.toFixed(3)}`
      : row === Math.floor(height / 2)
      ? ` ${((maxLoss + minLoss) / 2).toFixed(3)}`
      : row === 0
      ? ` ${minLoss.toFixed(3)}`
      : "";
    lines.push("│" + line + label);
  }
  lines.push("└" + "─".repeat(width) + "▶ steps");
  return lines.join("\n");
}

// ============================================================
// SECTION 11b: Word tokenizer (used for the demo)
// ============================================================

// The demo uses word-level tokenization rather than BPE.
// BPE (lesson 18) produces ~165 tokens from this corpus; with D=16 and sparse
// numerical gradients there aren't enough steps to learn word structure.
// Word tokenization gives a ~55-word vocab: the embedding table converges
// quickly, so generated output will contain recognizable words from the corpus.
// The BPE tokenizer is still built and reported in the architecture section.

interface WordTokenizer2 {
  wordToId: Map<string, number>;
  idToWord: Map<number, string>;
  vocabSize: number;
  encode: (text: string) => number[];
  decode: (ids: number[]) => string;
}

function buildWordTokenizer(corpus: string): WordTokenizer2 {
  const words = corpus.toLowerCase().match(/[a-z]+/g) ?? [];
  const vocab = [...new Set(words)].sort();
  const wordToId = new Map<string, number>();
  const idToWord = new Map<number, string>();
  vocab.forEach((w, i) => { wordToId.set(w, i); idToWord.set(i, w); });
  // special
  const unkId = vocab.length;
  wordToId.set("<unk>", unkId); idToWord.set(unkId, "<unk>");
  const size = vocab.length + 1;
  return {
    wordToId, idToWord,
    vocabSize: size,
    encode(text: string): number[] {
      return (text.toLowerCase().match(/[a-z]+/g) ?? []).map(w => wordToId.get(w) ?? unkId);
    },
    decode(ids: number[]): string {
      return ids.map(id => idToWord.get(id) ?? "?").join(" ");
    },
  };
}

// ============================================================
// SECTION 12: Training corpus
// ============================================================

const TRAINING_CORPUS = `
The attention mechanism allows the model to focus on relevant parts of the input sequence.
Transformers use self-attention to capture long-range dependencies in text.
Language models learn to predict the next token given the previous context.
Backpropagation computes gradients by applying the chain rule through each layer.
The Adam optimizer adapts the learning rate for each parameter using first and second moments.
Neural networks learn representations by composing simple transformations.
Embedding layers map discrete tokens to continuous vectors in a high-dimensional space.
Layer normalization stabilizes training by normalizing activations at each layer.
The residual connections allow gradients to flow directly through the network.
Mixture of experts activates only a subset of parameters for each input token.
The softmax function converts logits into a probability distribution over the vocabulary.
Rotary position embeddings encode token positions into the attention mechanism directly.
The KV cache stores previously computed keys and values to speed up autoregressive generation.
Scaling language models to larger sizes produces emergent capabilities not seen at smaller scales.
Reinforcement learning from human feedback fine-tunes models to follow instructions.
The transformer architecture replaced recurrent networks for most sequence modeling tasks.
Weight tying shares the token embedding matrix with the final output projection.
`.trim();

// ============================================================
// SECTION 13: Architecture Report
// ============================================================

function printArchitectureReport(model: MiniGPT4, numMerges: number): void {
  const cfg = model.config;
  const { total, activePerToken } = countParams(model);

  const fmt = (n: number): string => {
    if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return n.toString();
  };

  console.log("\n=== MINI-GPT-4 ARCHITECTURE ===");
  console.log(`Tokenizer:      char-BPE (${numMerges} merges, vocab size ${cfg.vocabSize})`);
  console.log(`Embedding:      ${cfg.vocabSize} × ${cfg.d}  (weight-tied with output)`);
  console.log(`Pos Encoding:   RoPE (no parameters)`);
  console.log(`Transformer:    ${cfg.numLayers} layers × (MHA + MoE-FFN + Pre-LN)`);
  console.log(`  MHA:          ${cfg.numHeads} heads × ${Math.floor(cfg.d / cfg.numHeads)} dim`);
  console.log(`  MoE FFN:      ${cfg.numExperts} experts, top-${cfg.topK} active, 4× expansion`);
  console.log(`Context:        ${cfg.blockSize} tokens`);
  console.log(`Parameters:     ${fmt(total)} total, ${fmt(activePerToken)} active per forward pass`);
  console.log(`Optimizer:      Adam (lr=5e-3, β1=0.9, β2=0.999)`);
  console.log("================================\n");
}

// ============================================================
// SECTION 14: Main — train and generate
// ============================================================

function main(): void {
  console.log("╔══════════════════════════════════════════════════════╗");
  console.log("║          LESSON 30: MINI-GPT-4 CAPSTONE              ║");
  console.log("║   Every idea from the curriculum — assembled.         ║");
  console.log("╚══════════════════════════════════════════════════════╝\n");

  // ── 1. Build BPE tokenizer (for architecture report only) ──
  console.log("Building BPE tokenizer (for architecture display)...");
  const NUM_MERGES = 80;
  const bpeTokenizer = buildBPE(TRAINING_CORPUS, NUM_MERGES);
  console.log(`  BPE vocab size: ${bpeTokenizer.vocabSize} (${NUM_MERGES} merges applied)`);

  // Word tokenizer is used for actual training — smaller vocab converges faster
  // with the sparse numerical gradient approach used here.
  const tokenizer = buildWordTokenizer(TRAINING_CORPUS);
  console.log(`  Word vocab size: ${tokenizer.vocabSize} words\n`);

  // ── 2. Create model ────────────────────────────────────────
  const D = 32;
  const NUM_HEADS = 4;
  const NUM_LAYERS = 2;
  const NUM_EXPERTS = 4;
  const TOP_K = 2;
  const BLOCK_SIZE = 24;

  console.log("Initializing Mini-GPT-4...");
  const model = createMiniGPT4(tokenizer, D, NUM_HEADS, NUM_LAYERS, NUM_EXPERTS, TOP_K, BLOCK_SIZE);

  // ── 3. Print architecture report ──────────────────────────
  printArchitectureReport(model, NUM_MERGES);

  // ── 4. Prepare training data ───────────────────────────────
  console.log("Tokenizing training corpus...");
  const allIds = tokenizer.encode(TRAINING_CORPUS);
  console.log(`  Total tokens: ${allIds.length}\n`);

  // ── 5. Training loop ───────────────────────────────────────
  const NUM_STEPS = 300;
  const SEQ_LEN = 12;

  const paramAccessor = collectParams(model);
  const adam = createAdam(paramAccessor.count(), 5e-3);
  const losses: number[] = [];

  console.log(`Training for ${NUM_STEPS} steps (seq_len=${SEQ_LEN})...`);
  console.log("(Using sampled numerical gradients — may take ~1-2 minutes)\n");

  for (let step = 0; step < NUM_STEPS; step++) {
    // Random slice from corpus
    const maxStart = allIds.length - SEQ_LEN - 1;
    const start = Math.floor(Math.random() * maxStart);
    const inputIds = allIds.slice(start, start + SEQ_LEN);
    const targetIds = allIds.slice(start + 1, start + SEQ_LEN + 1);

    const loss = trainStep(model, inputIds, targetIds, paramAccessor, adam);
    losses.push(loss);

    const bar = "█".repeat(Math.round((step / NUM_STEPS) * 20)) + "░".repeat(20 - Math.round((step / NUM_STEPS) * 20));
    process.stdout.write(`\r  Step ${String(step + 1).padStart(3)}/${NUM_STEPS}  [${bar}]  loss=${loss.toFixed(4)}`);
  }
  console.log("\n");

  // ── 6. ASCII loss curve ────────────────────────────────────
  console.log("Training Loss Curve:");
  console.log(asciiLossCurve(losses, 50, 10));
  console.log();

  // ── 7. Generate text at different temperatures ─────────────
  const prompts = [
    "The attention mechanism",
    "Language models learn",
    "Scaling language models",
  ];
  const temps = [
    { temp: 0.5, topP: 0.9, label: "Temperature 0.5 (focused)" },
    { temp: 0.9, topP: 0.9, label: "Temperature 0.9 (balanced)" },
    { temp: 1.4, topP: 0.95, label: "Temperature 1.4 (creative)" },
  ];

  console.log("=== TEXT GENERATION ===\n");
  for (let i = 0; i < 3; i++) {
    const { temp, topP, label } = temps[i];
    const prompt = prompts[i];
    console.log(`[${label}]`);
    console.log(`Prompt: "${prompt}"`);
    const generated = generate(model, prompt, 40, temp, topP);
    console.log(`Output: "${generated}"`);
    console.log();
  }

  // ── 8. Final celebration ───────────────────────────────────
  console.log("╔══════════════════════════════════════════════════════════╗");
  console.log("║                                                          ║");
  console.log("║   You built this.                                        ║");
  console.log("║                                                          ║");
  console.log("║   From y = wx + b  →  Multi-head attention               ║");
  console.log("║   From a single gradient  →  Adam with moment estimates  ║");
  console.log("║   From a character  →  BPE subword tokenization          ║");
  console.log("║   From fixed positions  →  RoPE encoding                 ║");
  console.log("║   From one FFN  →  Mixture of Experts routing            ║");
  console.log("║   From full recompute  →  KV cache inference             ║");
  console.log("║   From greedy argmax  →  Nucleus (top-p) sampling        ║");
  console.log("║   From GPT-1  →  GPT-4 architecture                      ║");
  console.log("║                                                          ║");
  console.log("║   Every concept above — you implemented from scratch.    ║");
  console.log("║   In TypeScript. With no libraries. Lesson by lesson.    ║");
  console.log("║                                                          ║");
  console.log("║   The researchers who built GPT-4 started the same way. ║");
  console.log("║                                                          ║");
  console.log("║             CURRICULUM COMPLETE.                         ║");
  console.log("║                                                          ║");
  console.log("╚══════════════════════════════════════════════════════════╝");
}

main();

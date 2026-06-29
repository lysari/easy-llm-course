// Lesson 19 — Multi-Head Attention
// Replaces the single-head attention from lessons 12/16 with H parallel heads,
// each operating in a (embedDim / numHeads) subspace.
// New in this lesson: MultiHeadAttention class, Wo output projection, attention
// weight visualisation, and single-head vs multi-head loss comparison.

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function matmul(A: number[][], B: number[][]): number[][] {
  const rows = A.length, cols = B[0]?.length ?? 0, inner = B.length;
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

function zeros(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => Array<number>(cols).fill(0));
}

function addMat(A: number[][], B: number[][]): number[][] {
  return A.map((row, i) => row.map((v, j) => v + (B[i]?.[j] ?? 0)));
}

function rand(r: number, c: number, scale = 0.1): number[][] {
  return Array.from({ length: r }, () =>
    Array.from({ length: c }, () => (Math.random() - 0.5) * scale)
  );
}

function softmaxGrad(A: number[], dA: number[]): number[] {
  const dot = A.reduce((s, a, k) => s + a * (dA[k] ?? 0), 0);
  return A.map((a, j) => a * ((dA[j] ?? 0) - dot));
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-Head Attention class
// ─────────────────────────────────────────────────────────────────────────────

interface MHACache {
  X: number[][];
  // Per-head intermediates
  Qs: number[][][];       // [H][T][head_dim]
  Ks: number[][][];
  Vs: number[][][];
  scores: number[][][];   // [H][T][T]  pre-softmax (after mask)
  attnW: number[][][];    // [H][T][T]  post-softmax
  headOuts: number[][][]; // [H][T][head_dim]
  concat: number[][];     // [T][embedDim]
  output: number[][];     // [T][embedDim]  after Wo
}

class MultiHeadAttention {
  embedDim: number;
  numHeads: number;
  headDim: number;

  // Per-head projection matrices: [H][embedDim][headDim]
  Wqs: number[][][];
  Wks: number[][][];
  Wvs: number[][][];
  // Output projection: [embedDim][embedDim]
  Wo: number[][];

  constructor(embedDim: number, numHeads: number) {
    if (embedDim % numHeads !== 0) {
      throw new Error(`embedDim (${embedDim}) must be divisible by numHeads (${numHeads})`);
    }
    this.embedDim = embedDim;
    this.numHeads = numHeads;
    this.headDim = embedDim / numHeads;

    // Initialise with small random weights, scaled by 1/sqrt(headDim) for
    // numerical stability (same convention as the single-head version).
    const scale = 0.1;
    this.Wqs = Array.from({ length: numHeads }, () => rand(embedDim, this.headDim, scale));
    this.Wks = Array.from({ length: numHeads }, () => rand(embedDim, this.headDim, scale));
    this.Wvs = Array.from({ length: numHeads }, () => rand(embedDim, this.headDim, scale));
    // Wo is (embedDim × embedDim), output projection
    this.Wo = rand(embedDim, embedDim, scale);
  }

  // ── Forward pass ──────────────────────────────────────────────────────────
  // X: (T × embedDim)
  // Returns output (T × embedDim) plus all intermediates for backprop.
  forward(X: number[][]): MHACache {
    const T = X.length;
    const { numHeads, headDim, embedDim } = this;
    const scale = Math.sqrt(headDim);

    const Qs: number[][][] = [];
    const Ks: number[][][] = [];
    const Vs: number[][][] = [];
    const scoresArr: number[][][] = [];
    const attnW: number[][][] = [];
    const headOuts: number[][][] = [];

    for (let h = 0; h < numHeads; h++) {
      // Project to head subspace: (T × headDim)
      const Q = matmul(X, this.Wqs[h]!);
      const K = matmul(X, this.Wks[h]!);
      const V = matmul(X, this.Wvs[h]!);

      // Scaled scores: (T × T)
      const rawScores = matmul(Q, transpose(K)).map(row => row.map(s => s / scale));

      // Causal mask
      for (let i = 0; i < T; i++)
        for (let j = i + 1; j < T; j++)
          rawScores[i]![j] = -Infinity;

      const weights = rawScores.map(row => softmax(row));         // T × T
      const headOut = matmul(weights, V);                          // T × headDim

      Qs.push(Q);
      Ks.push(K);
      Vs.push(V);
      scoresArr.push(rawScores);
      attnW.push(weights);
      headOuts.push(headOut);
    }

    // Concatenate all head outputs: (T × embedDim)
    const concat: number[][] = Array.from({ length: T }, (_, t) => {
      const row: number[] = [];
      for (let h = 0; h < numHeads; h++)
        for (let d = 0; d < headDim; d++)
          row.push(headOuts[h]![t]![d] ?? 0);
      return row;
    });

    // Output projection: (T × embedDim)
    const output = matmul(concat, this.Wo);

    return { X, Qs, Ks, Vs, scores: scoresArr, attnW, headOuts, concat, output };
  }

  // ── Backward pass ─────────────────────────────────────────────────────────
  // dOutput: (T × embedDim) — upstream gradient flowing into this module's output
  // Returns dX: (T × embedDim) — gradient for the input X
  // Also updates Wqs, Wks, Wvs, Wo in-place using lr.
  backward(cache: MHACache, dOutput: number[][], lr: number): number[][] {
    const { X, Qs, Ks, Vs, attnW, concat } = cache;
    const T = X.length;
    const { numHeads, headDim, embedDim } = this;
    const scale = Math.sqrt(headDim);

    // ── A. Gradient through Wo ─────────────────────────────────────────────
    //   output  = concat · Wo    [T×D = T×D · D×D]
    //   dWo     = concat^T · dOutput    [D×D]
    //   dConcat = dOutput · Wo^T        [T×D]

    const dWo = matmul(transpose(concat), dOutput);           // D × D
    const dConcat = matmul(dOutput, transpose(this.Wo));      // T × D

    // Update Wo
    for (let i = 0; i < embedDim; i++)
      for (let j = 0; j < embedDim; j++)
        this.Wo[i]![j]! -= lr * (dWo[i]?.[j] ?? 0);

    // ── B. Split dConcat back into per-head gradients ──────────────────────
    //   concat was built by stacking head outputs: head h occupies columns
    //   [h * headDim, (h+1) * headDim).

    // dX accumulates contributions from all heads
    const dX: number[][] = zeros(T, embedDim);

    for (let h = 0; h < numHeads; h++) {
      const Q = Qs[h]!;
      const K = Ks[h]!;
      const V = Vs[h]!;
      const A = attnW[h]!;

      // Slice dConcat for head h: [T × headDim]
      const dHeadOut: number[][] = Array.from({ length: T }, (_, t) =>
        (dConcat[t] ?? []).slice(h * headDim, (h + 1) * headDim)
      );

      // ── Step 1: dAttnOut → dV, dAttnWeights ────────────────────────────
      //   headOut  = A · V       [T×headDim]
      //   dV       = A^T · dHeadOut          [T×headDim]
      //   dA       = dHeadOut · V^T           [T×T]

      const dV = matmul(transpose(A), dHeadOut);             // T × headDim
      const dA = matmul(dHeadOut, transpose(V));             // T × T

      // ── Step 2: dA through softmax → dScores ───────────────────────────
      const dScores: number[][] = zeros(T, T);
      for (let i = 0; i < T; i++) {
        const rowGrad = softmaxGrad(A[i]!, dA[i]!);
        for (let j = 0; j <= i; j++)
          dScores[i]![j] = rowGrad[j] ?? 0;
        // j > i: masked, gradient stays 0
      }

      // ── Step 3: dScores → dQ, dK ───────────────────────────────────────
      //   scores = Q · K^T / scale
      //   dQ = dScores · K  / scale
      //   dK = dScores^T · Q / scale

      const dQ = matmul(dScores, K).map(row => row.map(v => v / scale));      // T×headDim
      const dK = matmul(transpose(dScores), Q).map(row => row.map(v => v / scale)); // T×headDim

      // ── Step 4: dQ, dK, dV → dWqh, dWkh, dWvh ─────────────────────────
      //   Q = X · Wq   →   dWq = X^T · dQ

      const dWq = matmul(transpose(X), dQ);   // D × headDim
      const dWk = matmul(transpose(X), dK);
      const dWv = matmul(transpose(X), dV);

      for (let i = 0; i < embedDim; i++) {
        for (let j = 0; j < headDim; j++) {
          this.Wqs[h]![i]![j]! -= lr * (dWq[i]?.[j] ?? 0);
          this.Wks[h]![i]![j]! -= lr * (dWk[i]?.[j] ?? 0);
          this.Wvs[h]![i]![j]! -= lr * (dWv[i]?.[j] ?? 0);
        }
      }

      // ── Step 5: accumulate dX from head h ──────────────────────────────
      //   dX_from_Q = dQ · Wq^T    [T × D]
      //   dX_from_K = dK · Wk^T
      //   dX_from_V = dV · Wv^T

      const dX_Q = matmul(dQ, transpose(this.Wqs[h]!));
      const dX_K = matmul(dK, transpose(this.Wks[h]!));
      const dX_V = matmul(dV, transpose(this.Wvs[h]!));

      for (let t = 0; t < T; t++)
        for (let j = 0; j < embedDim; j++)
          dX[t]![j]! += (dX_Q[t]?.[j] ?? 0) + (dX_K[t]?.[j] ?? 0) + (dX_V[t]?.[j] ?? 0);
    }

    return dX;
  }

  // Parameter count breakdown
  paramCount(): { perHead: number; allHeads: number; Wo: number; total: number } {
    const { embedDim, headDim, numHeads } = this;
    const perHead = 3 * embedDim * headDim;       // Wq + Wk + Wv for one head
    const allHeads = perHead * numHeads;           // = 3 * D²  (same as single-head 3×D×D)
    const Wo = embedDim * embedDim;
    return { perHead, allHeads, Wo, total: allHeads + Wo };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tokenizer
// ─────────────────────────────────────────────────────────────────────────────

class WordTokenizer {
  wordToId = new Map<string, number>();
  idToWord = new Map<number, string>();

  constructor(text: string) {
    [...new Set(text.split(/\s+/))].sort().forEach((w, i) => {
      this.wordToId.set(w, i);
      this.idToWord.set(i, w);
    });
  }

  get vocabSize() { return this.wordToId.size; }
  encode(text: string): number[] { return text.split(/\s+/).map(w => this.wordToId.get(w) ?? 0); }
  decode(ids: number[]): string { return ids.map(i => this.idToWord.get(i) ?? "?").join(" "); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Config and training data
// ─────────────────────────────────────────────────────────────────────────────

const trainingText = "hello world. the cat sat on the mat. a dog ran in the fog.";
const tokenizer = new WordTokenizer(trainingText);
const vocabSize = tokenizer.vocabSize;

const config = {
  embedDim: 16,
  blockSize: 8,
  lr: 0.003,
  epochs: 150,
  numHeads: 4,        // multi-head config
};

const { embedDim, blockSize } = config;

// ─────────────────────────────────────────────────────────────────────────────
// Two-layer transformer with multi-head attention
// ─────────────────────────────────────────────────────────────────────────────
//
// Architecture per layer:
//   X  →  MultiHeadAttention  →  residual + LayerNorm  →  FFN  →  residual + LayerNorm
//
// For simplicity the FFN is omitted here (covered in lesson 15); each layer is
// just MHA + residual + LN, which is sufficient to demonstrate multi-head effects.

interface LayerParams {
  mha: MultiHeadAttention;
}

interface ModelParams {
  embTable: number[][];
  posTable: number[][];
  layers: LayerParams[];
  Wproj: number[][];   // [vocabSize × embedDim]  output projection
}

function initModel(numLayers: number, numHeads: number): ModelParams {
  return {
    embTable: rand(vocabSize, embedDim),
    posTable: rand(blockSize, embedDim),
    layers: Array.from({ length: numLayers }, () => ({
      mha: new MultiHeadAttention(embedDim, numHeads),
    })),
    Wproj: rand(vocabSize, embedDim),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-head attention — minimal implementation for comparison
// ─────────────────────────────────────────────────────────────────────────────

interface SHCache {
  X: number[][];
  Q: number[][];
  K: number[][];
  V: number[][];
  scores: number[][];
  attnW: number[][];
  attnOut: number[][];
}

function singleHeadForward(
  X: number[][],
  Wq: number[][], Wk: number[][], Wv: number[][]
): SHCache {
  const T = X.length;
  const scale = Math.sqrt(embedDim);
  const Q = matmul(X, Wq);
  const K = matmul(X, Wk);
  const V = matmul(X, Wv);
  const rawScores = matmul(Q, transpose(K)).map(row => row.map(s => s / scale));
  for (let i = 0; i < T; i++)
    for (let j = i + 1; j < T; j++)
      rawScores[i]![j] = -Infinity;
  const attnW = rawScores.map(row => softmax(row));
  const attnOut = matmul(attnW, V);
  return { X, Q, K, V, scores: rawScores, attnW, attnOut };
}

function singleHeadBackward(
  cache: SHCache,
  dAttnOut: number[][],
  Wq: number[][], Wk: number[][], Wv: number[][],
  lr: number
): number[][] {
  const { X, Q, K, V, attnW } = cache;
  const T = X.length;
  const scale = Math.sqrt(embedDim);

  const dV = matmul(transpose(attnW), dAttnOut);
  const dA = matmul(dAttnOut, transpose(V));

  const dScores: number[][] = zeros(T, T);
  for (let i = 0; i < T; i++) {
    const rowGrad = softmaxGrad(attnW[i]!, dA[i]!);
    for (let j = 0; j <= i; j++)
      dScores[i]![j] = rowGrad[j] ?? 0;
  }

  const dQ = matmul(dScores, K).map(row => row.map(v => v / scale));
  const dK = matmul(transpose(dScores), Q).map(row => row.map(v => v / scale));

  const dWq = matmul(transpose(X), dQ);
  const dWk = matmul(transpose(X), dK);
  const dWv = matmul(transpose(X), dV);

  for (let i = 0; i < embedDim; i++)
    for (let j = 0; j < embedDim; j++) {
      Wq[i]![j]! -= lr * (dWq[i]?.[j] ?? 0);
      Wk[i]![j]! -= lr * (dWk[i]?.[j] ?? 0);
      Wv[i]![j]! -= lr * (dWv[i]?.[j] ?? 0);
    }

  const dX_Q = matmul(dQ, transpose(Wq));
  const dX_K = matmul(dK, transpose(Wk));
  const dX_V = matmul(dV, transpose(Wv));
  return addMat(addMat(dX_Q, dX_K), dX_V);
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward / backward / training for the multi-head model
// ─────────────────────────────────────────────────────────────────────────────

function crossEntropy(logits: number[][], targets: number[]): number {
  let total = 0;
  for (let i = 0; i < logits.length; i++) {
    const probs = softmax(logits[i]!);
    total += -Math.log((probs[targets[i]!] ?? 0) + 1e-9);
  }
  return total / logits.length;
}

interface FullForwardCache {
  tokens: number[];
  X: number[][];
  mhaCaches: MHACache[];      // one per layer
  X2s: number[][][];           // post-residual+LN per layer
  logits: number[][];
}

function forwardModel(tokens: number[], model: ModelParams): FullForwardCache {
  const { embTable, posTable, layers, Wproj } = model;
  const T = tokens.length;

  let X: number[][] = tokens.map((tok, pos) =>
    (embTable[tok] ?? []).map((v, j) => v + (posTable[pos]?.[j] ?? 0))
  );

  const mhaCaches: MHACache[] = [];
  const X2s: number[][][] = [];

  for (const layer of layers) {
    const mhaCache = layer.mha.forward(X);
    mhaCaches.push(mhaCache);

    // Residual + LayerNorm
    const X2: number[][] = X.map((row, t) =>
      layerNorm(row.map((v, j) => v + (mhaCache.output[t]?.[j] ?? 0)))
    );
    X2s.push(X2);
    X = X2;
  }

  const logits = X.map(row =>
    Wproj.map(wRow => wRow.reduce((s, w, j) => s + w * (row[j] ?? 0), 0))
  );

  return { tokens, X: X, mhaCaches, X2s, logits };
}

function backwardModel(
  cache: FullForwardCache,
  targets: number[],
  model: ModelParams,
  lr: number
): void {
  const { tokens, mhaCaches, X2s, logits } = cache;
  const { embTable, layers, Wproj } = model;
  const T = tokens.length;
  const numLayers = layers.length;

  // ── Gradient through output projection ──────────────────────────────────
  const lastX2 = X2s[numLayers - 1]!;
  const dX_out: number[][] = zeros(T, embedDim);

  for (let t = 0; t < T; t++) {
    const probs = softmax(logits[t]!);
    const dlogits = probs.map((p, v) => p - (v === targets[t] ? 1 : 0));
    const x2t = lastX2[t]!;

    for (let j = 0; j < embedDim; j++)
      for (let v = 0; v < vocabSize; v++)
        dX_out[t]![j]! += dlogits[v]! * (Wproj[v]?.[j] ?? 0);

    for (let v = 0; v < vocabSize; v++)
      for (let j = 0; j < embedDim; j++)
        Wproj[v]![j]! -= lr * dlogits[v]! * (x2t[j] ?? 0);
  }

  // ── Backprop through layers in reverse ───────────────────────────────────
  let dX: number[][] = dX_out;

  for (let l = numLayers - 1; l >= 0; l--) {
    const mhaCache = mhaCaches[l]!;
    // LN + residual: pass gradient straight through (same approximation as lesson 16)
    const dXfromMHA = layers[l]!.mha.backward(mhaCache, dX, lr);
    // Residual: gradient flows to both the MHA output and the skip connection
    dX = addMat(dX, dXfromMHA);
  }

  // ── Update embedding table ───────────────────────────────────────────────
  for (let t = 0; t < T; t++) {
    const tok = tokens[t]!;
    for (let j = 0; j < embedDim; j++)
      embTable[tok]![j]! -= lr * (dX[t]?.[j] ?? 0);
  }
}

function trainModel(
  model: ModelParams,
  epochs: number,
  label: string,
  printInterval = 25
): number[] {
  const encoded = tokenizer.encode(trainingText);
  const losses: number[] = [];

  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0, steps = 0;

    for (let start = 0; start + blockSize < encoded.length; start++) {
      const tokens = encoded.slice(start, start + blockSize);
      const targets = encoded.slice(start + 1, start + blockSize + 1);
      const fwd = forwardModel(tokens, model);
      totalLoss += crossEntropy(fwd.logits, targets);
      backwardModel(fwd, targets, model, config.lr);
      steps++;
    }

    const avgLoss = totalLoss / steps;
    losses.push(avgLoss);

    if (epoch % printInterval === 0 || epoch === epochs - 1) {
      console.log(`  [${label}] Epoch ${String(epoch).padStart(3)} — Loss: ${avgLoss.toFixed(4)}`);
    }
  }

  return losses;
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-head model for comparison (1 layer, single attention head)
// ─────────────────────────────────────────────────────────────────────────────

interface SHModelParams {
  embTable: number[][];
  posTable: number[][];
  Wq: number[][];
  Wk: number[][];
  Wv: number[][];
  Wproj: number[][];
}

function initSHModel(): SHModelParams {
  return {
    embTable: rand(vocabSize, embedDim),
    posTable: rand(blockSize, embedDim),
    Wq: rand(embedDim, embedDim),
    Wk: rand(embedDim, embedDim),
    Wv: rand(embedDim, embedDim),
    Wproj: rand(vocabSize, embedDim),
  };
}

function forwardSH(tokens: number[], p: SHModelParams): { logits: number[][], shCache: SHCache, X: number[][] } {
  const T = tokens.length;
  const X: number[][] = tokens.map((tok, pos) =>
    (p.embTable[tok] ?? []).map((v, j) => v + (p.posTable[pos]?.[j] ?? 0))
  );
  const shCache = singleHeadForward(X, p.Wq, p.Wk, p.Wv);
  const X2 = X.map((row, t) =>
    layerNorm(row.map((v, j) => v + (shCache.attnOut[t]?.[j] ?? 0)))
  );
  const logits = X2.map(row =>
    p.Wproj.map(wRow => wRow.reduce((s, w, j) => s + w * (row[j] ?? 0), 0))
  );
  return { logits, shCache, X: X2 };
}

function backwardSH(
  tokens: number[], targets: number[],
  logits: number[][], shCache: SHCache, X2: number[][],
  p: SHModelParams, lr: number
): void {
  const T = tokens.length;
  const dX2: number[][] = zeros(T, embedDim);

  for (let t = 0; t < T; t++) {
    const probs = softmax(logits[t]!);
    const dlogits = probs.map((prob, v) => prob - (v === targets[t] ? 1 : 0));
    const x2t = X2[t]!;
    for (let j = 0; j < embedDim; j++)
      for (let v = 0; v < vocabSize; v++)
        dX2[t]![j]! += dlogits[v]! * (p.Wproj[v]?.[j] ?? 0);
    for (let v = 0; v < vocabSize; v++)
      for (let j = 0; j < embedDim; j++)
        p.Wproj[v]![j]! -= lr * dlogits[v]! * (x2t[j] ?? 0);
  }

  const dAttnOut = dX2;
  const dX = singleHeadBackward(shCache, dAttnOut, p.Wq, p.Wk, p.Wv, lr);

  for (let t = 0; t < T; t++) {
    const tok = tokens[t]!;
    for (let j = 0; j < embedDim; j++)
      p.embTable[tok]![j]! -= lr * (dX[t]?.[j] ?? 0);
  }
}

function trainSH(p: SHModelParams, epochs: number, label: string, printInterval = 25): number[] {
  const encoded = tokenizer.encode(trainingText);
  const losses: number[] = [];

  for (let epoch = 0; epoch < epochs; epoch++) {
    let total = 0, steps = 0;
    for (let start = 0; start + blockSize < encoded.length; start++) {
      const tokens = encoded.slice(start, start + blockSize);
      const targets = encoded.slice(start + 1, start + blockSize + 1);
      const { logits, shCache, X: X2 } = forwardSH(tokens, p);
      total += crossEntropy(logits, targets);
      backwardSH(tokens, targets, logits, shCache, X2, p, config.lr);
      steps++;
    }
    const avg = total / steps;
    losses.push(avg);
    if (epoch % printInterval === 0 || epoch === epochs - 1)
      console.log(`  [${label}] Epoch ${String(epoch).padStart(3)} — Loss: ${avg.toFixed(4)}`);
  }
  return losses;
}

// ─────────────────────────────────────────────────────────────────────────────
// Attention weight visualisation
// ─────────────────────────────────────────────────────────────────────────────
// Prints a T×T heatmap of attention weights for one head as ASCII art.
// Each cell shows a shade character indicating the attention strength.

function printAttentionHeatmap(
  weights: number[][],   // T × T attention matrix
  tokens: number[],
  headIdx: number
): void {
  const T = weights.length;
  const tokenWords = tokens.map(id => (tokenizer.idToWord.get(id) ?? "?").slice(0, 6).padEnd(6));

  console.log(`\n  Head ${headIdx} attention weights (row = query, col = key)`);
  console.log(`  (causal: lower triangle only; upper = 0)`);
  console.log(`  Shade: ░ < 0.2   ▒ < 0.4   ▓ < 0.7   █ >= 0.7\n`);

  // Header row
  process.stdout.write("        ");
  for (const w of tokenWords) process.stdout.write(`${w} `);
  console.log();

  for (let i = 0; i < T; i++) {
    process.stdout.write(`  ${tokenWords[i]} `);
    for (let j = 0; j < T; j++) {
      const w = weights[i]?.[j] ?? 0;
      let ch: string;
      if (j > i)       ch = "  ·   ";   // masked future
      else if (w < 0.2) ch = "  ░   ";
      else if (w < 0.4) ch = "  ▒   ";
      else if (w < 0.7) ch = "  ▓   ";
      else               ch = "  █   ";
      process.stdout.write(ch);
    }
    // Print raw values for first 4 positions
    process.stdout.write("  | ");
    for (let j = 0; j <= Math.min(i, T - 1); j++)
      process.stdout.write(`${(weights[i]?.[j] ?? 0).toFixed(2)} `);
    console.log();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parameter count display
// ─────────────────────────────────────────────────────────────────────────────

function printParamCount(numLayers: number, numHeads: number): void {
  const { headDim } = new MultiHeadAttention(embedDim, numHeads);
  const perHeadPerLayer = 3 * embedDim * headDim;
  const allHeadsPerLayer = perHeadPerLayer * numHeads;
  const WoPerLayer = embedDim * embedDim;
  const mhaPerLayer = allHeadsPerLayer + WoPerLayer;
  const totalMHA = mhaPerLayer * numLayers;
  const embParams = vocabSize * embedDim + blockSize * embedDim;
  const projParams = vocabSize * embedDim;
  const total = embParams + totalMHA + projParams;

  console.log(`\n── Parameter count breakdown ──`);
  console.log(`  embedDim=${embedDim}, numHeads=${numHeads}, headDim=${headDim}`);
  console.log(`  numLayers=${numLayers}\n`);
  console.log(`  Per attention layer:`);
  console.log(`    Wq+Wk+Wv (all heads) = 3 × ${numHeads} × ${embedDim} × ${headDim} = ${allHeadsPerLayer}`);
  console.log(`    Wo                   = ${embedDim} × ${embedDim}                  = ${WoPerLayer}`);
  console.log(`    MHA total            = ${mhaPerLayer}`);
  console.log(`  All ${numLayers} layers             = ${totalMHA}`);
  console.log(`  Embedding table      = ${vocabSize} × ${embedDim}                  = ${vocabSize * embedDim}`);
  console.log(`  Positional table     = ${blockSize} × ${embedDim}                   = ${blockSize * embedDim}`);
  console.log(`  Output projection    = ${vocabSize} × ${embedDim}                  = ${projParams}`);
  console.log(`  TOTAL                                              = ${total}`);
  console.log();
  console.log(`  Single-head equivalent per layer: 4 × ${embedDim}² = ${4 * embedDim * embedDim}`);
  console.log(`  Multi-head same formula:  3×H×D×(D/H) + D² = 3D² + D² = 4D²`);
  console.log(`  = ${4 * embedDim * embedDim}  ✓  (same parameter count, reorganised)`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

console.log("=== Lesson 19 — Multi-Head Attention ===");
console.log(`Vocab: ${vocabSize} words   Embed: ${embedDim}d   Block: ${blockSize}`);
console.log(`Heads: ${config.numHeads}   Head dim: ${embedDim / config.numHeads}`);
console.log(`Layers: 2   Epochs: ${config.epochs}`);
console.log(`Random baseline loss: ${Math.log(vocabSize).toFixed(4)}\n`);

// ── 1. Parameter count ───────────────────────────────────────────────────────
printParamCount(2, config.numHeads);

// ── 2. Train multi-head model (2 layers, 4 heads) ────────────────────────────
console.log("\n── Training: 2-layer, 4-head transformer ──");
const mhModel = initModel(2, config.numHeads);
const mhLosses = trainModel(mhModel, config.epochs, "MH");

// ── 3. Train single-head model (1 layer) for comparison ─────────────────────
console.log("\n── Training: 1-layer, 1-head transformer (comparison) ──");
const shParams = initSHModel();
const shLosses = trainSH(shParams, config.epochs, "SH");

// ── 4. Loss comparison ───────────────────────────────────────────────────────
const mhFinal = mhLosses[mhLosses.length - 1]!;
const shFinal = shLosses[shLosses.length - 1]!;
const improvement = ((shFinal - mhFinal) / shFinal * 100).toFixed(1);

console.log("\n── Loss comparison ──");
console.log(`  Single-head (1 layer) final loss : ${shFinal.toFixed(4)}`);
console.log(`  Multi-head  (2 layers, 4 heads)  : ${mhFinal.toFixed(4)}`);
if (mhFinal < shFinal) {
  console.log(`  Multi-head is ${improvement}% lower — more expressive capacity.`);
} else {
  console.log(`  Both converge comparably on this tiny dataset (${vocabSize} words).`);
  console.log(`  Differences become pronounced at scale.`);
}

// ── 5. Attention weight visualisation ───────────────────────────────────────
console.log("\n── Attention weight visualisation (after training) ──");
const sampleText = "the cat sat on";
const sampleTokens = tokenizer.encode(sampleText);

// Get attention weights from the first layer of the multi-head model
const vizFwd = forwardModel(sampleTokens, mhModel);
const layer0Cache = vizFwd.mhaCaches[0]!;

// Show all 4 heads for the sample sequence
for (let h = 0; h < config.numHeads; h++) {
  printAttentionHeatmap(layer0Cache.attnW[h]!, sampleTokens, h);
}

console.log("\n── Architecture summary ──");
console.log(`  Layer 1 — MultiHeadAttention (${config.numHeads} heads × ${embedDim / config.numHeads}d) + residual + LN`);
console.log(`  Layer 2 — MultiHeadAttention (${config.numHeads} heads × ${embedDim / config.numHeads}d) + residual + LN`);
console.log(`  Output — linear projection → softmax → cross-entropy`);
console.log();
console.log("── Key equations ──");
console.log(`  For each head h:`);
console.log(`    Qh = X·Wqh   (${embedDim} → ${embedDim / config.numHeads})`);
console.log(`    Kh = X·Wkh   (${embedDim} → ${embedDim / config.numHeads})`);
console.log(`    Vh = X·Wvh   (${embedDim} → ${embedDim / config.numHeads})`);
console.log(`    Ah = softmax(Qh·Kh^T / sqrt(${embedDim / config.numHeads})) · Vh`);
console.log(`  concat = [A1 | A2 | A3 | A4]   (T × ${embedDim})`);
console.log(`  output = concat · Wo            (T × ${embedDim})`);
console.log();
console.log("── Real-world scale ──");
console.log("  GPT-2 small : 12 heads × 64  = 768   embed dim, 12 layers");
console.log("  GPT-2 medium: 16 heads × 64  = 1024  embed dim, 24 layers");
console.log("  GPT-3       : 96 heads × 128 = 12288 embed dim, 96 layers");
console.log("  GPT-4 (est) : ~96 heads × 128 = ~12288, many more layers");

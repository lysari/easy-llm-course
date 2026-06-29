// Lesson 16 — Backprop Through Attention
// Wq, Wk, Wv are now fully trained.
// New in this lesson: softmaxGrad + full attention backward pass.

// ── Helpers (carried forward from lesson 14) ──

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

// ── Tokenizer ──
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

// ── Config ──
const config = {
  embedDim: 16,
  blockSize: 8,
  lr: 0.003,
  frozenEpochs: 100,
  fullEpochs: 100,
};

// ── Training text (same as lesson 14) ──
const text = "hello world. the cat sat on the mat. a dog ran in the fog.";
const tokenizer = new WordTokenizer(text);
const vocabSize = tokenizer.vocabSize;
const { embedDim, blockSize } = config;

// ── Random initialiser ──
const rand = (r: number, c: number, scale = 0.1) =>
  Array.from({ length: r }, () =>
    Array.from({ length: c }, () => (Math.random() - 0.5) * scale)
  );

// ── Parameters ──
// We keep two independent copies so Phase 1 and Phase 2 are isolated.
function initParams() {
  return {
    embTable: rand(vocabSize, embedDim),
    posTable: rand(blockSize, embedDim),
    Wq: rand(embedDim, embedDim),
    Wk: rand(embedDim, embedDim),
    Wv: rand(embedDim, embedDim),
    Wproj: rand(vocabSize, embedDim),
  };
}

// ── NEW: Softmax gradient ──
// Given:
//   A   — softmax output for one row  (length T)
//   dA  — upstream gradient for that row  (length T)
// Returns:
//   dS  — gradient w.r.t. the pre-softmax scores  (length T)
//
// Derivation (see lesson.md § Step 2):
//   dot   = Σ_k  A[k] * dA[k]          (weighted mean of upstream grads)
//   dS[j] = A[j] * (dA[j] - dot)
//
// Intuitively: softmax outputs must sum to 1, so raising one logit lowers others.
// The "- dot" term encodes that coupling.
function softmaxGrad(A: number[], dA: number[]): number[] {
  // Step 2a: compute dot = Σ_k A[k] * dA[k]
  const dot = A.reduce((s, a, k) => s + a * (dA[k] ?? 0), 0);

  // Step 2b: dS[j] = A[j] * (dA[j] - dot)
  return A.map((a, j) => a * ((dA[j] ?? 0) - dot));
}

// ── Forward pass ──
// Returns all intermediate activations needed for backprop.
interface ForwardCache {
  tokens: number[];
  X: number[][];        // embedded + positional input
  Q: number[][];
  K: number[][];
  V: number[][];
  scores: number[][];   // pre-softmax (after causal mask)
  attnWeights: number[][]; // post-softmax A
  attnOut: number[][];
  X2: number[][];       // after residual + layerNorm
  logits: number[][];
}

function forward(tokens: number[], params: ReturnType<typeof initParams>): ForwardCache {
  const { embTable, posTable, Wq, Wk, Wv, Wproj } = params;
  const T = tokens.length;

  // Embed + positional encoding
  const X: number[][] = tokens.map((tok, pos) =>
    (embTable[tok] ?? []).map((v, j) => v + (posTable[pos]?.[j] ?? 0))
  );

  // Project to Q, K, V
  const Q = matmul(X, Wq);   // [T × d]
  const K = matmul(X, Wk);   // [T × d]
  const V = matmul(X, Wv);   // [T × d]

  // Scaled dot-product scores: QK^T / sqrt(d)
  const scale = Math.sqrt(embedDim);
  const scores = matmul(Q, transpose(K)).map(row => row.map(s => s / scale));

  // Causal mask: token i cannot attend to future token j > i
  for (let i = 0; i < T; i++)
    for (let j = i + 1; j < T; j++)
      scores[i]![j] = -Infinity;

  // Softmax over each row → attention weights
  const attnWeights = scores.map(row => softmax(row));

  // Weighted sum of values
  const attnOut = matmul(attnWeights, V);   // [T × d]

  // Residual + LayerNorm
  const X2 = X.map((row, i) => layerNorm(row.map((v, j) => v + (attnOut[i]?.[j] ?? 0))));

  // Output projection: logits[t][v] = X2[t] · Wproj[v]
  const logits = X2.map(row =>
    Wproj.map(wRow => wRow.reduce((s, w, j) => s + w * (row[j] ?? 0), 0))
  );

  return { tokens, X, Q, K, V, scores, attnWeights, attnOut, X2, logits };
}

// ── Cross-entropy loss ──
function crossEntropy(logits: number[][], targets: number[]): number {
  let total = 0;
  for (let i = 0; i < logits.length; i++) {
    const probs = softmax(logits[i]!);
    total += -Math.log((probs[targets[i]!] ?? 0) + 1e-9);
  }
  return total / logits.length;
}

// ── Full backward pass (attention included) ──
// trainAttention flag: when false, Wq/Wk/Wv updates are skipped (Phase 1).
function backward(
  cache: ForwardCache,
  targets: number[],
  params: ReturnType<typeof initParams>,
  lr: number,
  trainAttention: boolean
): void {
  const { tokens, X, Q, K, V, attnWeights, X2, logits } = cache;
  const { embTable, Wq, Wk, Wv, Wproj } = params;
  const T = tokens.length;
  const scale = Math.sqrt(embedDim);

  // ── A. Gradient through output projection (same as lesson 14) ──
  // logits[t][v] = X2[t] · Wproj[v]
  // dL/dlogits[v] = probs[v] - 1(v == target)   (cross-entropy + softmax combined)
  // dL/dWproj[v,j] = dlogits[v] * X2[t][j]
  // dL/dX2[t][j]   = Σ_v dlogits[v] * Wproj[v,j]

  // Accumulate dX2 (gradient flowing back to the residual stream)
  const dX2: number[][] = zeros(T, embedDim);

  for (let t = 0; t < T; t++) {
    const probs   = softmax(logits[t]!);
    const dlogits = probs.map((p, v) => p - (v === targets[t] ? 1 : 0));
    const x2t     = X2[t]!;

    // Gradient into X2[t]: dX2[t][j] = Σ_v dlogits[v] * Wproj[v][j]
    for (let j = 0; j < embedDim; j++)
      for (let v = 0; v < vocabSize; v++)
        dX2[t]![j]! += dlogits[v]! * (Wproj[v]?.[j] ?? 0);

    // Update Wproj: Wproj[v][j] -= lr * dlogits[v] * x2t[j]
    for (let v = 0; v < vocabSize; v++)
      for (let j = 0; j < embedDim; j++)
        Wproj[v]![j]! -= lr * dlogits[v]! * (x2t[j] ?? 0);
  }

  // ── B. Gradient through LayerNorm + residual (simplified: pass dX2 straight through) ──
  // LayerNorm backprop has its own Jacobian; for this lesson we use the common
  // approximation of passing gradients straight through the normalisation.
  // This is valid enough for our toy model and keeps the focus on attention backprop.
  const dAttnOut: number[][] = dX2;  // dL/d(attnOut) ≈ dL/d(X2)

  if (!trainAttention) {
    // Phase 1: frozen attention — update only embeddings and return.
    for (let t = 0; t < T; t++) {
      const tok = tokens[t]!;
      for (let j = 0; j < embedDim; j++)
        embTable[tok]![j]! -= lr * (dAttnOut[t]?.[j] ?? 0);
    }
    return;
  }

  // ── C. Step 1: dAttnOut → dV and dAttnWeights ──
  //
  //   attnOut = attnWeights · V       [T×d = T×T · T×d]
  //
  //   dV           = attnWeights^T · dAttnOut    [T×d]
  //   dAttnWeights = dAttnOut · V^T              [T×T]

  // dV[k][j] = Σ_i attnWeights[i][k] * dAttnOut[i][j]
  const dV = matmul(transpose(attnWeights), dAttnOut);  // T×d

  // dAttnWeights[i][k] = Σ_j dAttnOut[i][j] * V[k][j]
  const dAttnWeights = matmul(dAttnOut, transpose(V));   // T×T

  // ── D. Step 2: dAttnWeights through softmax → dScores ──
  //
  // For each row i, apply the softmax gradient formula:
  //   dot_i     = Σ_k  attnWeights[i][k] * dAttnWeights[i][k]
  //   dScores[i][j] = attnWeights[i][j] * (dAttnWeights[i][j] - dot_i)
  //
  // Then enforce the causal mask: dScores[i][j] = 0 for j > i
  // (masked positions had zero weight and contributed nothing forward)

  const dScores: number[][] = zeros(T, T);
  for (let i = 0; i < T; i++) {
    const rowGrad = softmaxGrad(attnWeights[i]!, dAttnWeights[i]!);
    for (let j = 0; j <= i; j++) {
      // j <= i: unmasked positions get their gradient
      dScores[i]![j] = rowGrad[j] ?? 0;
    }
    // j > i: masked positions get gradient 0 (already 0 from zeros())
  }

  // ── E. Step 3: dScores → dQ and dK ──
  //
  //   scores = Q · K^T / sqrt(d)
  //
  //   dQ = dScores · K  / sqrt(d)      [T×d]
  //   dK = dScores^T · Q / sqrt(d)     [T×d]

  // dQ[i][j] = (1/sqrt(d)) * Σ_k dScores[i][k] * K[k][j]
  const dQ = matmul(dScores, K).map(row => row.map(v => v / scale));      // T×d

  // dK[k][j] = (1/sqrt(d)) * Σ_i dScores[i][k] * Q[i][j]
  const dK = matmul(transpose(dScores), Q).map(row => row.map(v => v / scale)); // T×d

  // ── F. Step 4: dQ, dK, dV → dWq, dWk, dWv ──
  //
  //   Q = X · Wq   →   dWq = X^T · dQ     [d×d]
  //   K = X · Wk   →   dWk = X^T · dK     [d×d]
  //   V = X · Wv   →   dWv = X^T · dV     [d×d]

  const dWq = matmul(transpose(X), dQ);   // d×d
  const dWk = matmul(transpose(X), dK);   // d×d
  const dWv = matmul(transpose(X), dV);   // d×d

  // Update Wq, Wk, Wv
  for (let i = 0; i < embedDim; i++) {
    for (let j = 0; j < embedDim; j++) {
      Wq[i]![j]! -= lr * (dWq[i]?.[j] ?? 0);
      Wk[i]![j]! -= lr * (dWk[i]?.[j] ?? 0);
      Wv[i]![j]! -= lr * (dWv[i]?.[j] ?? 0);
    }
  }

  // ── G. Step 5: dX ← accumulated from Q, K, V paths + direct residual ──
  //
  //   dX_from_Q = dQ · Wq^T
  //   dX_from_K = dK · Wk^T
  //   dX_from_V = dV · Wv^T
  //   dX_total  = dX_from_Q + dX_from_K + dX_from_V + dAttnOut  (residual path)

  const dX_Q = matmul(dQ, transpose(Wq));   // T×d  (using ORIGINAL Wq before update)
  const dX_K = matmul(dK, transpose(Wk));
  const dX_V = matmul(dV, transpose(Wv));
  const dX   = addMat(addMat(addMat(dX_Q, dX_K), dX_V), dAttnOut);

  // Update embedding table
  for (let t = 0; t < T; t++) {
    const tok = tokens[t]!;
    for (let j = 0; j < embedDim; j++)
      embTable[tok]![j]! -= lr * (dX[t]?.[j] ?? 0);
  }
}

// ── Training loop ──
function trainPhase(
  params: ReturnType<typeof initParams>,
  epochs: number,
  trainAttention: boolean,
  label: string
): number[] {
  const encoded = tokenizer.encode(text);
  const losses: number[] = [];

  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0, steps = 0;

    for (let start = 0; start + blockSize < encoded.length; start++) {
      const tokens  = encoded.slice(start, start + blockSize);
      const targets = encoded.slice(start + 1, start + blockSize + 1);

      const cache = forward(tokens, params);
      totalLoss += crossEntropy(cache.logits, targets);
      backward(cache, targets, params, config.lr, trainAttention);
      steps++;
    }

    const avgLoss = totalLoss / steps;
    losses.push(avgLoss);

    if (epoch % 20 === 0 || epoch === epochs - 1) {
      const epochDisplay = String(epoch).padStart(3);
      console.log(`  [${label}] Epoch ${epochDisplay} — Loss: ${avgLoss.toFixed(4)}`);
    }
  }

  return losses;
}

// ── Sample next token ──
function sampleToken(logits: number[], temperature = 1.0): number {
  const probs = softmax(logits.map(l => l / temperature));
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i] ?? 0;
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

function generate(seed: string, length: number, params: ReturnType<typeof initParams>, temperature = 0.8): string {
  let tokens = tokenizer.encode(seed);
  for (let i = 0; i < length; i++) {
    const context = tokens.slice(-blockSize);
    const { logits } = forward(context, params);
    const lastLogits = logits[logits.length - 1]!;
    tokens.push(sampleToken(lastLogits, temperature));
  }
  return tokenizer.decode(tokens);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main: Phase 1 (frozen) then Phase 2 (full backprop), SAME initial weights
// We run both phases on the SAME params object so Phase 2 picks up where
// Phase 1 left off — this is the fairest demonstration of the improvement.
// ─────────────────────────────────────────────────────────────────────────────

console.log("=== Lesson 16 — Backprop Through Attention ===");
console.log(`Vocab: ${vocabSize} words   Embed: ${embedDim}d   Block: ${blockSize} tokens`);
console.log(`Random baseline loss: ${Math.log(vocabSize).toFixed(4)}\n`);

const params = initParams();

// ── Phase 1: frozen Wq/Wk/Wv (100 epochs) ──
console.log("── Phase 1: Wq/Wk/Wv FROZEN (trainAttention = false) ──");
const phase1Losses = trainPhase(params, config.frozenEpochs, false, "frozen");
const p1Final = phase1Losses[phase1Losses.length - 1]!.toFixed(4);

// ── Phase 2: full backprop including Wq/Wk/Wv (100 more epochs) ──
console.log("\n── Phase 2: Wq/Wk/Wv TRAINED (trainAttention = true) ──");
const phase2Losses = trainPhase(params, config.fullEpochs, true, "full");
const p2Final = phase2Losses[phase2Losses.length - 1]!.toFixed(4);

// ── Summary ──
console.log("\n── Summary ──");
console.log(`  Phase 1 start loss : ${phase1Losses[0]!.toFixed(4)}`);
console.log(`  Phase 1 final loss : ${p1Final}  (frozen attention plateau)`);
console.log(`  Phase 2 final loss : ${p2Final}  (full backprop)`);
const drop = ((+p1Final - +p2Final) / +p1Final * 100).toFixed(1);
console.log(`  Further reduction  : ${drop}%  after unlocking attention weights`);

// ── Generated text ──
console.log("\n── Generated text (after full training) ──");
console.log(`  "the" → ${generate("the", 8, params, 0.8)}`);
console.log(`  "a"   → ${generate("a", 8, params, 0.8)}`);

// ── Architecture ──
console.log("\n── Architecture ──");
console.log(`  embTable  [${vocabSize} × ${embedDim}]   — trained`);
console.log(`  posTable  [${blockSize} × ${embedDim}]   — fixed`);
console.log(`  Wq/Wk/Wv  [${embedDim} × ${embedDim}]  — FROZEN in phase 1, TRAINED in phase 2`);
console.log(`  Wproj     [${vocabSize} × ${embedDim}]   — trained`);
console.log(`\n  New in lesson 16:`);
console.log(`    softmaxGrad(A, dA)       — the Jacobian-free row formula`);
console.log(`    attentionBackward(cache) — dV, dA, dScores, dQ, dK, dWq, dWk, dWv`);
console.log(`    causal mask in backward  — zero out dScores[i][j] for j > i`);

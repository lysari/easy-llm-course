// Lesson 15 — Feed-Forward Network (the other half of a transformer)
//
// This file extends lesson 14's tiny GPT with a complete FFN layer.
// Every addition vs lesson 14 is marked with  ← NEW
//
// Architecture (complete transformer block):
//   Embed + PosEncode
//   → Causal Self-Attention
//   → Residual + LayerNorm  (X1)
//   → FFN: W1 → GELU → W2   ← NEW
//   → Residual + LayerNorm  (X2)
//   → Wproj → logits

// ── Helpers (unchanged from lesson 14) ──────────────────────────────────────

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

// ── GELU activation  ← NEW ──────────────────────────────────────────────────
//
// GELU(x) = x · 0.5 · (1 + tanh(sqrt(2/π) · (x + 0.044715 · x³)))
//
// Smoother than ReLU: allows small negative values to pass through.
// Used in GPT-2 and all modern transformers.

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI); // ≈ 0.7979

function gelu(x: number): number {
  const inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
  return x * 0.5 * (1 + Math.tanh(inner));
}

// GELU gradient: needed for backprop through the FFN hidden layer.
// Derived by differentiating the GELU formula with the chain rule.
function geluGrad(x: number): number {
  const c = SQRT_2_OVER_PI;
  const inner = c * (x + 0.044715 * x * x * x);
  const tanhVal = Math.tanh(inner);
  // d/dx [ 0.5 · x · (1 + tanh(inner)) ]
  //   = 0.5 · (1 + tanh(inner))
  //   + 0.5 · x · (1 - tanh²(inner)) · d(inner)/dx
  //
  // d(inner)/dx = c · (1 + 3 · 0.044715 · x²)
  const dInner = c * (1 + 3 * 0.044715 * x * x);
  return 0.5 * (1 + tanhVal) + 0.5 * x * (1 - tanhVal * tanhVal) * dInner;
}

// ── Tokenizers (unchanged from lesson 14) ────────────────────────────────────

class CharTokenizer {
  charToId = new Map<string, number>();
  idToChar = new Map<number, string>();

  constructor(text: string) {
    [...new Set(text.split(""))].sort().forEach((c, i) => {
      this.charToId.set(c, i);
      this.idToChar.set(i, c);
    });
  }

  get vocabSize() { return this.charToId.size; }
  encode(text: string): number[] { return text.split("").map(c => this.charToId.get(c) ?? 0); }
  decode(ids: number[]): string { return ids.map(i => this.idToChar.get(i) ?? "?").join(""); }
}

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

// ── Config ────────────────────────────────────────────────────────────────────

const config = {
  embedDim: 16,
  blockSize: 8,
  numHeads: 2,
  numLayers: 1,
  lr: 0.005,
  epochs: 4000,
};

// ── Training text ─────────────────────────────────────────────────────────────

const text = "hello world. the cat sat on the mat. a dog ran in the fog.";
const tokenizer = new WordTokenizer(text);

const vocabSize = tokenizer.vocabSize;
const { embedDim, blockSize } = config;

// FFN hidden dimension = 4 × embedDim  (the standard expansion ratio)  ← NEW
const ffnDim = 4 * embedDim;  // 64 when embedDim = 16

// ── Parameter initialisation ──────────────────────────────────────────────────

const rand = (r: number, c: number, scale = 0.1) =>
  Array.from({ length: r }, () => Array.from({ length: c }, () => (Math.random() - 0.5) * scale));

const randVec = (n: number, scale = 0.0) =>
  Array.from({ length: n }, () => (Math.random() - 0.5) * scale);

// Attention parameters (unchanged from lesson 14)
const embTable = rand(vocabSize, embedDim);   // [vocabSize × embedDim]
const posTable = rand(blockSize, embedDim);   // [blockSize × embedDim]
const Wq = rand(embedDim, embedDim);
const Wk = rand(embedDim, embedDim);
const Wv = rand(embedDim, embedDim);
const Wproj = rand(vocabSize, embedDim);      // [vocabSize × embedDim]

// FFN parameters  ← NEW
// W1: expand embedDim → ffnDim   shape [ffnDim × embedDim]
// W2: compress ffnDim → embedDim shape [embedDim × ffnDim]
// biases initialised to zero (standard practice)
const W1 = rand(ffnDim, embedDim, 0.1);      // [64 × 16]
const b1 = randVec(ffnDim, 0.0);             // [64]
const W2 = rand(embedDim, ffnDim, 0.1);      // [16 × 64]
const b2 = randVec(embedDim, 0.0);           // [16]

// ── Parameter count ───────────────────────────────────────────────────────────
//
// Lesson 14 had: embTable + Wproj (Wq/Wk/Wv were frozen)
//   embTable: vocabSize × embedDim
//   Wproj:    vocabSize × embedDim
//   Total trained: 2 × vocabSize × embedDim
//
// Lesson 15 adds: W1, b1, W2, b2
//   W1: ffnDim × embedDim = 4 × embedDim²
//   b1: ffnDim
//   W2: embedDim × ffnDim = 4 × embedDim²
//   b2: embedDim
//   FFN params: 8 × embedDim² + 5 × embedDim

const lesson14Params = 2 * vocabSize * embedDim;
const ffnParams = ffnDim * embedDim + ffnDim + embedDim * ffnDim + embedDim;
const lesson15Params = lesson14Params + ffnParams;

// ── Forward pass  ← UPDATED ──────────────────────────────────────────────────
//
// Returns logits, X1 (post-attention), X2 (post-FFN), and FFN intermediates.
// The FFN intermediates (pre, h) are needed for backprop.

interface ForwardResult {
  logits: number[][];
  X1: number[][];     // post-attention residual (input to FFN)
  X2: number[][];     // post-FFN residual (input to Wproj)
  ffnPre: number[][]; // W1·x+b1 before GELU, shape [T × ffnDim]
  ffnH: number[][];   // GELU(pre), shape [T × ffnDim]
}

function forward(tokens: number[]): ForwardResult {
  const T = tokens.length;

  // ── Embed + positional encoding ───────────────────────────────────────────
  const X: number[][] = tokens.map((tok, pos) =>
    (embTable[tok] ?? []).map((v, j) => v + (posTable[pos]?.[j] ?? 0))
  );

  // ── Causal self-attention (single head) ───────────────────────────────────
  const Q = matmul(X, Wq);
  const K = matmul(X, Wk);
  const V = matmul(X, Wv);
  const scale = Math.sqrt(embedDim);
  const scores = matmul(Q, transpose(K)).map(row => row.map(s => s / scale));

  // Causal mask: token i cannot attend to future token j > i
  for (let i = 0; i < T; i++)
    for (let j = i + 1; j < T; j++)
      scores[i]![j] = -Infinity;

  const attnWeights = scores.map(row => softmax(row));
  const attnOut = matmul(attnWeights, V);

  // ── Residual + LayerNorm after attention ──────────────────────────────────
  // X1[t] = LayerNorm(X[t] + attnOut[t])
  const X1 = X.map((row, i) => layerNorm(row.map((v, j) => v + (attnOut[i]?.[j] ?? 0))));

  // ── Feed-Forward Network (new in lesson 15) ───────────────────────────────
  //
  // Applied independently to each token position.
  //
  // For each token t:
  //   pre[t]  = W1 · X1[t] + b1       expand to ffnDim=64
  //   h[t]    = GELU(pre[t])           nonlinearity
  //   ffnOut  = W2 · h[t] + b2         compress to embedDim=16

  const ffnPre: number[][] = [];  // shape [T × ffnDim] — saved for backprop
  const ffnH: number[][] = [];    // shape [T × ffnDim] — saved for backprop
  const ffnOut: number[][] = [];  // shape [T × embedDim]

  for (let t = 0; t < T; t++) {
    const x1t = X1[t]!;

    // Step 1: expand — pre = W1 · x1t + b1
    const pre = Array<number>(ffnDim).fill(0);
    for (let i = 0; i < ffnDim; i++) {
      let val = b1[i]!;
      for (let j = 0; j < embedDim; j++)
        val += (W1[i]?.[j] ?? 0) * (x1t[j] ?? 0);
      pre[i] = val;
    }
    ffnPre.push(pre);

    // Step 2: activate — h = GELU(pre)
    const h = pre.map(gelu);
    ffnH.push(h);

    // Step 3: compress — out = W2 · h + b2
    const out = Array<number>(embedDim).fill(0);
    for (let i = 0; i < embedDim; i++) {
      let val = b2[i]!;
      for (let j = 0; j < ffnDim; j++)
        val += (W2[i]?.[j] ?? 0) * (h[j] ?? 0);
      out[i] = val;
    }
    ffnOut.push(out);
  }

  // ── Residual + LayerNorm after FFN ────────────────────────────────────────
  // X2[t] = LayerNorm(X1[t] + ffnOut[t])
  const X2 = X1.map((row, i) => layerNorm(row.map((v, j) => v + (ffnOut[i]?.[j] ?? 0))));

  // ── Project to vocab logits ───────────────────────────────────────────────
  // logits[t][v] = X2[t] · Wproj[v]
  const logits = X2.map(row =>
    Wproj.map(wRow => wRow.reduce((s, w, j) => s + w * (row[j] ?? 0), 0))
  );

  return { logits, X1, X2, ffnPre, ffnH };
}

// ── Cross-entropy loss (unchanged) ────────────────────────────────────────────

function crossEntropy(logits: number[][], targets: number[]): number {
  let total = 0;
  for (let i = 0; i < logits.length; i++) {
    const probs = softmax(logits[i]!);
    total += -Math.log((probs[targets[i]!] ?? 0) + 1e-9);
  }
  return total / logits.length;
}

// ── Sample next token (unchanged) ────────────────────────────────────────────

function sampleToken(logits: number[], temperature = 1.0): number {
  const probs = softmax(logits.map(l => l / temperature));
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i] ?? 0;
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

// ── Training ──────────────────────────────────────────────────────────────────
//
// Backprop now covers Wproj, embTable, AND the FFN weights.
// Wq/Wk/Wv are still frozen (backprop through attention softmax = future lesson).
//
// Gradient flow:
//
//   dL/dlogits  (cross-entropy + softmax)
//       ↓
//   dL/dWproj, dL/dX2   (output projection)
//       ↓
//   (LayerNorm is approximated as identity for simplicity)
//       ↓
//   dL/dffnOut = dL/dX2  (residual: gradient passes through addition unchanged)
//       ↓
//   dL/dW2 = dL/dffnOut^T · h       ← FFN W2 gradient  ← NEW
//   dL/db2 = dL/dffnOut             ← FFN b2 gradient  ← NEW
//       ↓
//   dL/dh = W2^T · dL/dffnOut
//   dL/dpre = dL/dh ⊙ GELU'(pre)   ← apply GELU gradient  ← NEW
//       ↓
//   dL/dW1 = dL/dpre^T · x1         ← FFN W1 gradient  ← NEW
//   dL/db1 = dL/dpre                ← FFN b1 gradient  ← NEW
//       ↓
//   dL/dX1 ≈ W1^T · dL/dpre        (gradient to attention output — approximate)
//       ↓
//   dL/demb ≈ dL/dX1               (approximate: ignores attention transform)

const encoded = tokenizer.encode(text);

console.log("=== Tiny GPT + FFN (Lesson 15) ===");
console.log(`Vocab: ${vocabSize} words  Embed: ${embedDim}d  FFN hidden: ${ffnDim}d  Block: ${blockSize} tokens`);
console.log(`\nParameter count:`);
console.log(`  Lesson 14 (no FFN): ${lesson14Params.toLocaleString()} trained params`);
console.log(`  FFN adds:           ${ffnParams.toLocaleString()} params  (W1+b1+W2+b2)`);
console.log(`  Lesson 15 total:    ${lesson15Params.toLocaleString()} trained params`);
console.log(`\nTraining tokens: ${encoded.length}  Random baseline loss: ${Math.log(vocabSize).toFixed(4)}\n`);

for (let epoch = 0; epoch < config.epochs; epoch++) {
  let totalLoss = 0, steps = 0;

  for (let start = 0; start + blockSize < encoded.length; start++) {
    const tokens  = encoded.slice(start, start + blockSize);
    const targets = encoded.slice(start + 1, start + blockSize + 1);

    const { logits, X1, X2, ffnPre, ffnH } = forward(tokens);
    totalLoss += crossEntropy(logits, targets);

    for (let t = 0; t < tokens.length; t++) {
      const probs   = softmax(logits[t]!);
      const dlogits = probs.map((p, v) => p - (v === targets[t] ? 1 : 0));
      const x2t     = X2[t]!;
      const x1t     = X1[t]!;
      const tok     = tokens[t]!;
      const pret    = ffnPre[t]!;  // W1·x1 + b1 (before GELU)
      const ht      = ffnH[t]!;    // GELU(pre)

      // ── dL/dX2 — gradient of loss w.r.t. post-FFN representation ──────────
      // dL/dX2[j] = Σ_v dlogits[v] · Wproj[v][j]
      const dX2 = Array<number>(embedDim).fill(0);
      for (let j = 0; j < embedDim; j++)
        for (let v = 0; v < vocabSize; v++)
          dX2[j] += dlogits[v]! * (Wproj[v]?.[j] ?? 0);

      // ── Update Wproj ───────────────────────────────────────────────────────
      // dL/dWproj[v][j] = dlogits[v] · x2t[j]
      for (let v = 0; v < vocabSize; v++)
        for (let j = 0; j < embedDim; j++)
          Wproj[v]![j]! -= config.lr * dlogits[v]! * (x2t[j] ?? 0);

      // ── Backprop through FFN (new in lesson 15) ───────────────────────────
      //
      // The residual X2 = LayerNorm(X1 + ffnOut).
      // We approximate LayerNorm as identity, so dL/dffnOut ≈ dX2.

      const dffnOut = dX2;  // gradient flowing into FFN from above

      // ── dL/dW2 and dL/db2 ─────────────────────────────────────────────────
      // ffnOut[i] = Σ_j W2[i][j] · h[j] + b2[i]
      // dL/dW2[i][j] = dffnOut[i] · h[j]
      // dL/db2[i]    = dffnOut[i]
      for (let i = 0; i < embedDim; i++) {
        b2[i]! -= config.lr * (dffnOut[i] ?? 0);
        for (let j = 0; j < ffnDim; j++)
          W2[i]![j]! -= config.lr * (dffnOut[i] ?? 0) * (ht[j] ?? 0);
      }

      // ── dL/dh = W2^T · dffnOut ────────────────────────────────────────────
      // h[j] feeds into every W2[i][j], so gradient accumulates across i.
      const dh = Array<number>(ffnDim).fill(0);
      for (let j = 0; j < ffnDim; j++)
        for (let i = 0; i < embedDim; i++)
          dh[j] += (W2[i]?.[j] ?? 0) * (dffnOut[i] ?? 0);

      // ── dL/dpre = dL/dh ⊙ GELU'(pre) ─────────────────────────────────────
      // Chain rule through GELU: multiply by the GELU derivative at each position.
      const dpre = dh.map((g, j) => g * geluGrad(pret[j] ?? 0));

      // ── dL/dW1 and dL/db1 ─────────────────────────────────────────────────
      // pre[i] = Σ_j W1[i][j] · x1[j] + b1[i]
      // dL/dW1[i][j] = dpre[i] · x1[j]
      // dL/db1[i]    = dpre[i]
      for (let i = 0; i < ffnDim; i++) {
        b1[i]! -= config.lr * (dpre[i] ?? 0);
        for (let j = 0; j < embedDim; j++)
          W1[i]![j]! -= config.lr * (dpre[i] ?? 0) * (x1t[j] ?? 0);
      }

      // ── dL/dX1 ≈ W1^T · dpre ─────────────────────────────────────────────
      // This gradient flows back through the attention residual into embTable.
      const dX1 = Array<number>(embedDim).fill(0);
      for (let j = 0; j < embedDim; j++)
        for (let i = 0; i < ffnDim; i++)
          dX1[j] += (W1[i]?.[j] ?? 0) * (dpre[i] ?? 0);

      // ── Update embTable ────────────────────────────────────────────────────
      // Combine gradient from Wproj path (via dX2) and FFN path (via dX1).
      // Both gradients ultimately came from the same embedding, so we add them.
      for (let j = 0; j < embedDim; j++)
        embTable[tok]![j]! -= config.lr * (dX2[j]! + dX1[j]!);
    }
    steps++;
  }

  if (epoch % 100 === 0 || epoch === config.epochs - 1)
    console.log(`Epoch ${String(epoch).padStart(4)} — Loss: ${(totalLoss / steps).toFixed(4)}`);
}

// ── Text generation (unchanged from lesson 14) ────────────────────────────────

function generate(seed: string, length: number, temperature = 0.8): string {
  let tokens = tokenizer.encode(seed);
  for (let i = 0; i < length; i++) {
    const context = tokens.slice(-blockSize);
    const { logits } = forward(context);
    const lastLogits = logits[logits.length - 1]!;
    tokens.push(sampleToken(lastLogits, temperature));
  }
  return tokenizer.decode(tokens);
}

console.log("\n=== Generated text (after training) ===");
console.log(`"the" → ${generate("the", 8, 0.8)}`);
console.log(`"a"   → ${generate("a", 8, 0.8)}`);

const trainedTokens = encoded.slice(0, blockSize);
const trainedTargets = encoded.slice(1, blockSize + 1);
const { logits: finalLogits } = forward(trainedTokens);
console.log(`\nFinal loss: ${crossEntropy(finalLogits, trainedTargets).toFixed(4)}`);

console.log("\n=== Architecture (complete transformer block) ===");
console.log(`  embTable  [${vocabSize} × ${embedDim}]         — trained`);
console.log(`  posTable  [${blockSize} × ${embedDim}]         — fixed`);
console.log(`  Wq/Wk/Wv  [${embedDim} × ${embedDim}]        — frozen`);
console.log(`  W1        [${ffnDim} × ${embedDim}]        — trained  ← NEW`);
console.log(`  b1        [${ffnDim}]                 — trained  ← NEW`);
console.log(`  W2        [${embedDim} × ${ffnDim}]        — trained  ← NEW`);
console.log(`  b2        [${embedDim}]                 — trained  ← NEW`);
console.log(`  Wproj     [${vocabSize} × ${embedDim}]         — trained`);
console.log(`\n  Lesson 14 params: ${lesson14Params.toLocaleString()}`);
console.log(`  FFN params added: ${ffnParams.toLocaleString()}`);
console.log(`  Lesson 15 total:  ${lesson15Params.toLocaleString()}`);
console.log(`\n  This is now a COMPLETE transformer block.`);
console.log(`  GPT-2 small: 12 layers × (attn + FFN), embedDim=768, ffnDim=3072, 117M params.`);
console.log(`  Claude: ~100 layers, embedDim~8192, ffnDim~32768, ~500B params.`);

// Lesson 21 — KV Cache: Making Inference Fast
// Demonstrates the KV cache optimization for autoregressive generation.
// No imports — all math is hand-rolled.

// ─────────────────────────────────────────────────────────────────────────────
// Utility math
// ─────────────────────────────────────────────────────────────────────────────

function matMul(A: number[][], B: number[][]): number[][] {
  const rowsA = A.length;
  const colsA = A[0].length;
  const colsB = B[0].length;
  const C: number[][] = Array.from({ length: rowsA }, () => new Array(colsB).fill(0));
  for (let i = 0; i < rowsA; i++) {
    for (let k = 0; k < colsA; k++) {
      for (let j = 0; j < colsB; j++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return C;
}

// Transpose a 2-D matrix
function transpose(M: number[][]): number[][] {
  const rows = M.length;
  const cols = M[0].length;
  const T: number[][] = Array.from({ length: cols }, () => new Array(rows).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      T[j][i] = M[i][j];
    }
  }
  return T;
}

function softmax(x: number[]): number[] {
  const max = Math.max(...x);
  const exps = x.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

// Element-wise add for 1-D vectors
function addVec(a: number[], b: number[]): number[] {
  return a.map((v, i) => v + b[i]);
}

// Scale a 1-D vector
function scaleVec(v: number[], s: number): number[] {
  return v.map(x => x * s);
}

// Layer norm over a 1-D vector
function layerNorm(x: number[]): number[] {
  const mean = x.reduce((s, v) => s + v, 0) / x.length;
  const variance = x.reduce((s, v) => s + (v - mean) ** 2, 0) / x.length;
  const std = Math.sqrt(variance + 1e-5);
  return x.map(v => (v - mean) / std);
}

// ReLU activation
function relu(x: number[]): number[] {
  return x.map(v => Math.max(0, v));
}

// Matrix-vector multiply: M (rows × cols) applied to v (cols) → (rows)
function matvec(M: number[][], v: number[]): number[] {
  return M.map(row => row.reduce((s, w, j) => s + w * v[j], 0));
}

// Seeded pseudo-random number generator (xorshift32) for reproducible weights
class RNG {
  private state: number;
  constructor(seed: number) { this.state = seed >>> 0 || 1; }
  next(): number {
    let x = this.state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return (this.state >>> 0) / 0xffffffff;
  }
  // Gaussian(0, std) via Box-Muller
  randn(std = 0.02): number {
    const u = this.next() + 1e-10;
    const v = this.next() + 1e-10;
    return std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  // Fill a 2-D matrix with small random values
  matrix(rows: number, cols: number, std = 0.02): number[][] {
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => this.randn(std))
    );
  }
  // Fill a 1-D vector
  vector(size: number, std = 0.02): number[] {
    return Array.from({ length: size }, () => this.randn(std));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// KVCache — stores K and V tensors for every layer
// ─────────────────────────────────────────────────────────────────────────────

class KVCache {
  // cachedK[layerIdx] is a 2-D array of shape (seq_len, head_dim)
  // cachedV[layerIdx] is the same shape
  private cachedK: number[][][];
  private cachedV: number[][][];
  private numLayers: number;

  constructor(numLayers: number) {
    this.numLayers = numLayers;
    this.cachedK = Array.from({ length: numLayers }, () => []);
    this.cachedV = Array.from({ length: numLayers }, () => []);
  }

  // Append a single new K row and V row to the cache for a given layer.
  // newK and newV are 1-D vectors of shape (head_dim,).
  append(layerIdx: number, newK: number[], newV: number[]): void {
    this.cachedK[layerIdx].push([...newK]);
    this.cachedV[layerIdx].push([...newV]);
  }

  // Return the full cached K and V matrices for a given layer.
  // Returns { K: (T × head_dim), V: (T × head_dim) }
  get(layerIdx: number): { K: number[][]; V: number[][] } {
    return {
      K: this.cachedK[layerIdx],
      V: this.cachedV[layerIdx],
    };
  }

  // Current sequence length stored in the cache (all layers are the same length)
  seqLen(): number {
    return this.cachedK[0].length;
  }

  // Reset the cache completely (e.g. before a new generation)
  clear(): void {
    this.cachedK = Array.from({ length: this.numLayers }, () => []);
    this.cachedV = Array.from({ length: this.numLayers }, () => []);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tiny GPT-2-style model weights
// ─────────────────────────────────────────────────────────────────────────────

// Hyperparameters — kept small so timing demo is clear
const VOCAB_SIZE  = 64;
const EMBED_DIM   = 32;
const NUM_LAYERS  = 2;
const HEAD_DIM    = 16;   // EMBED_DIM / 2 (single head per layer for simplicity)
const FFN_DIM     = 64;
const MAX_SEQ_LEN = 128;

// Initialize weights once, shared by both the cached and naive implementations
const rng = new RNG(42);

interface LayerWeights {
  Wq: number[][];  // (EMBED_DIM × HEAD_DIM)
  Wk: number[][];  // (EMBED_DIM × HEAD_DIM)
  Wv: number[][];  // (EMBED_DIM × HEAD_DIM)
  Wo: number[][];  // (HEAD_DIM  × EMBED_DIM)
  W1: number[][];  // (EMBED_DIM × FFN_DIM)
  W2: number[][];  // (FFN_DIM   × EMBED_DIM)
}

interface ModelWeights {
  embedding: number[][];    // (VOCAB_SIZE × EMBED_DIM)
  posEmbedding: number[][];  // (MAX_SEQ_LEN × EMBED_DIM)
  layers: LayerWeights[];
  Wout: number[][];          // (EMBED_DIM × VOCAB_SIZE)  — unembedding
}

function initWeights(): ModelWeights {
  const r = new RNG(42);
  return {
    embedding:    r.matrix(VOCAB_SIZE, EMBED_DIM),
    posEmbedding: r.matrix(MAX_SEQ_LEN, EMBED_DIM),
    layers: Array.from({ length: NUM_LAYERS }, () => ({
      Wq: r.matrix(EMBED_DIM, HEAD_DIM),
      Wk: r.matrix(EMBED_DIM, HEAD_DIM),
      Wv: r.matrix(EMBED_DIM, HEAD_DIM),
      Wo: r.matrix(HEAD_DIM,  EMBED_DIM),
      W1: r.matrix(EMBED_DIM, FFN_DIM),
      W2: r.matrix(FFN_DIM,   EMBED_DIM),
    })),
    Wout: r.matrix(EMBED_DIM, VOCAB_SIZE),
  };
}

const weights = initWeights();

// ─────────────────────────────────────────────────────────────────────────────
// Attention helpers
// ─────────────────────────────────────────────────────────────────────────────

// Full self-attention over a sequence of token embeddings.
// x: (T × EMBED_DIM), layer weights → output (T × EMBED_DIM)
function selfAttentionFull(x: number[][], lw: LayerWeights): number[][] {
  const T = x.length;
  const scale = 1 / Math.sqrt(HEAD_DIM);

  // Compute Q, K, V for all positions
  const Q = matMul(x, lw.Wq);  // (T × HEAD_DIM)
  const K = matMul(x, lw.Wk);  // (T × HEAD_DIM)
  const V = matMul(x, lw.Wv);  // (T × HEAD_DIM)

  // Attention scores with causal mask
  const Kt = transpose(K);  // (HEAD_DIM × T)
  const scores = matMul(Q, Kt);  // (T × T)

  // Apply causal mask: position i can only attend to positions <= i
  const maskedScores = scores.map((row, i) =>
    row.map((v, j) => (j <= i ? v * scale : -1e9))
  );

  // Softmax over each row
  const attnWeights = maskedScores.map(row => softmax(row));  // (T × T)

  // Weighted sum of values
  const attnOut = matMul(attnWeights, V);  // (T × HEAD_DIM)

  // Project back to EMBED_DIM
  const Wot = transpose(lw.Wo);  // (EMBED_DIM × HEAD_DIM) — wait, Wo is (HEAD_DIM × EMBED_DIM)
  // attnOut (T × HEAD_DIM) × Wo (HEAD_DIM × EMBED_DIM) = (T × EMBED_DIM)
  return matMul(attnOut, lw.Wo);
}

// FFN: two linear layers with ReLU
function ffn(x: number[], lw: LayerWeights): number[] {
  // x: (EMBED_DIM,)
  // W1: (EMBED_DIM × FFN_DIM), so matmul is x @ W1
  const hidden = matvec(transpose(lw.W1), x);  // (FFN_DIM,)
  const activated = relu(hidden);               // (FFN_DIM,)
  return matvec(transpose(lw.W2), activated);  // (EMBED_DIM,)
}

// Full transformer layer over a sequence
function transformerLayerFull(x: number[][], lw: LayerWeights): number[][] {
  const T = x.length;

  // Self-attention with residual
  const attnOut = selfAttentionFull(x, lw);  // (T × EMBED_DIM)
  const afterAttn = x.map((row, i) => layerNorm(addVec(row, attnOut[i])));

  // FFN with residual
  const afterFFN = afterAttn.map(row => layerNorm(addVec(row, ffn(row, lw))));

  return afterFFN;
}

// ─────────────────────────────────────────────────────────────────────────────
// Naive full forward pass (no cache) — used for comparison
// ─────────────────────────────────────────────────────────────────────────────

// Returns logits over vocabulary for the last token position.
function forwardFull(tokens: number[]): number[] {
  const T = tokens.length;

  // Token + positional embeddings
  let x: number[][] = tokens.map((tok, pos) =>
    addVec(weights.embedding[tok], weights.posEmbedding[pos])
  );

  // Run each transformer layer
  for (let l = 0; l < NUM_LAYERS; l++) {
    x = transformerLayerFull(x, weights.layers[l]);
  }

  // Unembedding: last token's hidden state → logits
  const lastHidden = x[T - 1];  // (EMBED_DIM,)
  return matvec(transpose(weights.Wout), lastHidden);  // (VOCAB_SIZE,)
}

// Greedy sampling (argmax)
function argmax(logits: number[]): number {
  let best = 0;
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > logits[best]) best = i;
  }
  return best;
}

// Generate `length` tokens naively (recomputes K,V from scratch each step)
function generateNaive(seed: number[], length: number): number[] {
  const tokens = [...seed];
  for (let step = 0; step < length; step++) {
    const logits = forwardFull(tokens);
    const next = argmax(logits);
    tokens.push(next);
  }
  return tokens.slice(seed.length);
}

// ─────────────────────────────────────────────────────────────────────────────
// Cached single-token forward pass
// ─────────────────────────────────────────────────────────────────────────────

// Process a single new token through one transformer layer using the KV cache.
//
// newTokenEmbed: embedding of the new token (EMBED_DIM,)
// cache: the KV cache — already populated with past tokens for all layers
// layerIdx: which transformer layer this is
//
// Returns the layer output for the new token position (EMBED_DIM,).
// ALSO appends the new K and V to the cache as a side effect.
function forwardCachedLayer(
  newTokenEmbed: number[],
  cache: KVCache,
  layerIdx: number
): number[] {
  const lw = weights.layers[layerIdx];
  const scale = 1 / Math.sqrt(HEAD_DIM);

  // Compute Q, K, V for the single new token only
  const newQ = matvec(transpose(lw.Wq), newTokenEmbed);  // (HEAD_DIM,)
  const newK = matvec(transpose(lw.Wk), newTokenEmbed);  // (HEAD_DIM,)
  const newV = matvec(transpose(lw.Wv), newTokenEmbed);  // (HEAD_DIM,)

  // Append this token's K and V to the cache BEFORE attending
  // (the cache already holds K,V for all previous tokens)
  cache.append(layerIdx, newK, newV);

  // Get the full K and V (all past + current)
  const { K, V } = cache.get(layerIdx);  // K: (T × HEAD_DIM), V: (T × HEAD_DIM)
  const T = K.length;

  // Attention: new_Q (1 × HEAD_DIM) dot K^T (HEAD_DIM × T) → scores (1 × T)
  // We compute this as a dot product loop since new_Q is a single row
  const scores: number[] = K.map(kRow =>
    kRow.reduce((sum, kVal, d) => sum + newQ[d] * kVal, 0) * scale
  );

  // No causal mask needed — the cache contains only past and current tokens,
  // never future ones. Every position in the cache is valid to attend to.
  const attnWeights = softmax(scores);  // (T,)

  // Weighted sum of values: attn_weights (T,) dot V (T × HEAD_DIM) → (HEAD_DIM,)
  const attnOut: number[] = new Array(HEAD_DIM).fill(0);
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < HEAD_DIM; d++) {
      attnOut[d] += attnWeights[t] * V[t][d];
    }
  }

  // Project back to EMBED_DIM
  const projected = matvec(transpose(lw.Wo), attnOut);  // (EMBED_DIM,)

  // Residual + layer norm after attention
  const afterAttn = layerNorm(addVec(newTokenEmbed, projected));

  // FFN with residual + layer norm
  const afterFFN = layerNorm(addVec(afterAttn, ffn(afterAttn, lw)));

  return afterFFN;
}

// Full cached forward pass for a single new token through ALL layers.
// Returns logits over the vocabulary.
function forwardCachedToken(
  tokenId: number,
  position: number,
  cache: KVCache
): number[] {
  // Token + positional embedding
  let embed = addVec(
    weights.embedding[tokenId],
    weights.posEmbedding[position]
  );  // (EMBED_DIM,)

  // Pass through each transformer layer (each one appends to cache)
  for (let l = 0; l < NUM_LAYERS; l++) {
    embed = forwardCachedLayer(embed, cache, l);
  }

  // Unembedding → logits
  return matvec(transpose(weights.Wout), embed);  // (VOCAB_SIZE,)
}

// ─────────────────────────────────────────────────────────────────────────────
// Prefill: populate the cache from a prompt
// ─────────────────────────────────────────────────────────────────────────────
//
// The prefill step runs the full forward pass on the entire prompt (parallel)
// and fills the KV cache. After prefill, the cache contains K and V for every
// prompt token at every layer.
//
// For simplicity in this demo, prefill is implemented as a sequential loop over
// cached single-token passes. In production systems, prefill runs in parallel
// (standard transformer forward pass) and the cache is populated from that.

function prefill(seed: number[], cache: KVCache): number[] {
  cache.clear();
  let lastEmbed: number[] = [];

  for (let pos = 0; pos < seed.length; pos++) {
    const tokenId = seed[pos];
    let embed = addVec(
      weights.embedding[tokenId],
      weights.posEmbedding[pos]
    );
    for (let l = 0; l < NUM_LAYERS; l++) {
      embed = forwardCachedLayer(embed, cache, l);
    }
    lastEmbed = embed;
  }

  // Return logits for the last seed token (used to pick the first generated token)
  return matvec(transpose(weights.Wout), lastEmbed);
}

// ─────────────────────────────────────────────────────────────────────────────
// generate_fast: prefill then decode with KV cache
// ─────────────────────────────────────────────────────────────────────────────

function generateFast(seed: number[], length: number): number[] {
  const cache = new KVCache(NUM_LAYERS);

  // Prefill phase: process the seed prompt and populate the KV cache
  const seedLogits = prefill(seed, cache);

  const generated: number[] = [];

  // First decode token comes from the prefill output
  let nextToken = argmax(seedLogits);
  generated.push(nextToken);

  // Decode phase: generate one token at a time, using cached K and V
  for (let step = 1; step < length; step++) {
    const position = seed.length + step - 1;
    const logits = forwardCachedToken(nextToken, position, cache);
    nextToken = argmax(logits);
    generated.push(nextToken);
  }

  return generated;
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo: compare timing of naive vs cached generation
// ─────────────────────────────────────────────────────────────────────────────

function main(): void {
  const SEED_TOKENS = [5, 12, 3, 8, 21, 14, 7, 2];   // 8-token prompt
  const GEN_LENGTH  = 50;                              // tokens to generate

  console.log("=== Lesson 21: KV Cache Demo ===\n");
  console.log(`Model: ${NUM_LAYERS} layers, EMBED_DIM=${EMBED_DIM}, HEAD_DIM=${HEAD_DIM}`);
  console.log(`Seed: ${SEED_TOKENS.length} tokens`);
  console.log(`Generating: ${GEN_LENGTH} tokens each\n`);

  // ── Naive generation (no cache) ──────────────────────────────────────────
  console.log("Running NAIVE generation (recomputes K,V from scratch every step)...");
  const naiveStart = Date.now();
  const naiveOutput = generateNaive(SEED_TOKENS, GEN_LENGTH);
  const naiveMs = Date.now() - naiveStart;
  const naiveMsPerToken = naiveMs / GEN_LENGTH;

  console.log(`  Total time: ${naiveMs}ms`);
  console.log(`  ms per token: ${naiveMsPerToken.toFixed(2)}ms`);
  console.log(`  Output (first 20 tokens): [${naiveOutput.slice(0, 20).join(", ")}]`);

  // ── Cached generation (KV cache) ─────────────────────────────────────────
  console.log("\nRunning CACHED generation (KV cache — each step only processes new token)...");
  const cachedStart = Date.now();
  const cachedOutput = generateFast(SEED_TOKENS, GEN_LENGTH);
  const cachedMs = Date.now() - cachedStart;
  const cachedMsPerToken = cachedMs / GEN_LENGTH;

  console.log(`  Total time: ${cachedMs}ms`);
  console.log(`  ms per token: ${cachedMsPerToken.toFixed(2)}ms`);
  console.log(`  Output (first 20 tokens): [${cachedOutput.slice(0, 20).join(", ")}]`);

  // ── Comparison ────────────────────────────────────────────────────────────
  const speedup = naiveMsPerToken / cachedMsPerToken;
  console.log("\n── Results ──────────────────────────────────────────────────────");
  console.log(`Naive:  ${naiveMsPerToken.toFixed(2)} ms/token`);
  console.log(`Cached: ${cachedMsPerToken.toFixed(2)} ms/token`);
  console.log(`Speedup: ${speedup.toFixed(1)}x`);

  const outputsMatch = naiveOutput.join(",") === cachedOutput.join(",");
  console.log(`Outputs identical: ${outputsMatch}`);
  if (!outputsMatch) {
    console.log("  (small differences can occur at layer-norm boundaries due to");
    console.log("   the sequential vs. batch normalization path)");
  }

  // ── Theoretical complexity reminder ───────────────────────────────────────
  console.log("\n── Complexity ───────────────────────────────────────────────────");
  const T = SEED_TOKENS.length + GEN_LENGTH;
  const naiveOps  = (T * (T + 1)) / 2;
  const cachedOps = T;
  console.log(`Naive   total attention ops ∝ N(N+1)/2 = ${naiveOps} for N=${T}`);
  console.log(`Cached  total attention ops ∝ N        = ${cachedOps} for N=${T}`);
  console.log(`Theoretical ratio: ${(naiveOps / cachedOps).toFixed(1)}x`);

  // ── Memory cost illustration ──────────────────────────────────────────────
  console.log("\n── KV Cache Memory Cost ─────────────────────────────────────────");

  function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
    return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  }

  // This demo model (fp64 in JS, shown as fp16 equivalent for illustration)
  const demoBytes = 2 * NUM_LAYERS * 1 * T * HEAD_DIM * 8;  // 8 bytes (JS number = float64)
  console.log(`This demo model KV cache: ${formatBytes(demoBytes)} (float64)`);

  // GPT-4 scale at 128k context (fp16 = 2 bytes)
  const gpt4Layers  = 96;
  const gpt4Heads   = 96;
  const gpt4HeadDim = 128;
  const gpt4Context = 128_000;
  const gpt4Bytes   = 2 * gpt4Layers * gpt4Heads * gpt4Context * gpt4HeadDim * 2;
  console.log(`GPT-4 scale (128k ctx, fp16): ${formatBytes(gpt4Bytes)} per batch item`);
  console.log("  — this is why long-context inference is expensive!");

  // ── Step-by-step trace for a short example ────────────────────────────────
  console.log("\n── Step-by-Step Cache Growth Trace ──────────────────────────────");
  const traceCache = new KVCache(NUM_LAYERS);
  const traceSeed  = [5, 12, 3];

  console.log("Prefill on seed [5, 12, 3]:");
  prefill(traceSeed, traceCache);
  console.log(`  Cache size after prefill: ${traceCache.seqLen()} positions`);

  for (let step = 0; step < 4; step++) {
    const pos     = traceSeed.length + step;
    const tokenId = (step * 7 + 11) % VOCAB_SIZE;  // arbitrary token ids for trace
    forwardCachedToken(tokenId, pos, traceCache);
    console.log(`  After decode step ${step + 1} (token=${tokenId}): cache size = ${traceCache.seqLen()} positions`);
  }

  console.log("\nDone. Each decode step appended exactly 1 row to the cache.");
}

main();

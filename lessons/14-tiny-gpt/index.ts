const { readFileSync } = require("node:fs");
// Tiny GPT: character-level language model — the capstone
// Everything from lessons 00–13 comes together here

// ── Helpers ──
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

// ── Tokenizer (Lesson 09) ──
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

// ── Config ──
const config = {
  embedDim: 16,
  blockSize: 8,   // context window length
  numHeads: 2,    // single-head implemented below; numHeads kept for architecture docs
  numLayers: 1,
  lr: 0.005,
  epochs: 4000,
};

// ── Training text ──
const text = readFileSync("./tinyshakespeare.txt", "utf-8");
const tokenizer = new CharTokenizer(text);

// Fix: vocabSize is a prototype getter — { ...tokenizer } spread only copies own
// properties and loses prototype getters. Access it directly.
const vocabSize = tokenizer.vocabSize;
const { embedDim, blockSize } = config;

// ── Random init ──
const rand = (r: number, c: number, scale = 0.1) =>
  Array.from({ length: r }, () => Array.from({ length: c }, () => (Math.random() - 0.5) * scale));

// ── Parameters ──
const embTable = rand(vocabSize, embedDim);   // [vocabSize × embedDim]
const posTable = rand(blockSize, embedDim);   // [blockSize × embedDim]
const Wq = rand(embedDim, embedDim);
const Wk = rand(embedDim, embedDim);
const Wv = rand(embedDim, embedDim);
const Wproj = rand(vocabSize, embedDim);      // [vocabSize × embedDim]

// ── Forward pass ──
// Returns logits AND X2 — X2 is needed for backprop through Wproj.
function forward(tokens: number[]): { logits: number[][], X2: number[][] } {
  const T = tokens.length;

  // Embed + positional encoding
  const X: number[][] = tokens.map((tok, pos) =>
    (embTable[tok] ?? []).map((v, j) => v + (posTable[pos]?.[j] ?? 0))
  );

  // Causal self-attention (single head)
  const Q = matmul(X, Wq);   // [T × embedDim]
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

  // Residual + LayerNorm
  const X2 = X.map((row, i) => layerNorm(row.map((v, j) => v + (attnOut[i]?.[j] ?? 0))));

  // Project to vocab logits: logits[t][v] = X2[t] · Wproj[v]
  const logits = X2.map(row =>
    Wproj.map(wRow => wRow.reduce((s, w, j) => s + w * (row[j] ?? 0), 0))
  );

  return { logits, X2 };
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

// ── Sample next token (temperature-scaled) ──
function sampleToken(logits: number[], temperature = 1.0): number {
  const probs = softmax(logits.map(l => l / temperature));
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i] ?? 0;
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

// ── Training ──
// Backprop through Wproj and embTable. Wq/Wk/Wv are frozen here —
// their gradients require backprop through the attention softmax (next lesson).
//
// Gradient derivation:
//   logits[t][v]   = X2[t] · Wproj[v]         (output projection)
//   dL/dlogits[v]  = probs[v] − 1(v == target) (cross-entropy + softmax)
//   dL/dWproj[v,j] = dlogits[v] * X2[t][j]
//   dL/dX2[t][j]   = Σ_v dlogits[v] * Wproj[v][j]
//   dL/demb[tok,j] ≈ dX2[t][j]  (approximate: ignores attention transform)

const encoded = tokenizer.encode(text);

console.log("=== Tiny GPT ===");
console.log(`Vocab: ${vocabSize} chars  Embed: ${embedDim}d  Block: ${blockSize} tokens`);
console.log(`Training tokens: ${encoded.length}  Random baseline loss: ${Math.log(vocabSize).toFixed(4)}\n`);

for (let epoch = 0; epoch < config.epochs; epoch++) {
  let totalLoss = 0, steps = 0;

  for (let start = 0; start + blockSize < encoded.length; start++) {
    const tokens  = encoded.slice(start, start + blockSize);
    const targets = encoded.slice(start + 1, start + blockSize + 1);

    const { logits, X2 } = forward(tokens);
    totalLoss += crossEntropy(logits, targets);

    for (let t = 0; t < tokens.length; t++) {
      const probs   = softmax(logits[t]!);
      const dlogits = probs.map((p, v) => p - (v === targets[t] ? 1 : 0));
      const x2t     = X2[t]!;
      const tok     = tokens[t]!;

      // dL/dX2 — compute before updating Wproj so we use the original weights
      const dX2 = Array<number>(embedDim).fill(0);
      for (let j = 0; j < embedDim; j++)
        for (let v = 0; v < vocabSize; v++)
          dX2[j] += dlogits[v]! * (Wproj[v]?.[j] ?? 0);

      // Update Wproj
      for (let v = 0; v < vocabSize; v++)
        for (let j = 0; j < embedDim; j++)
          Wproj[v]![j]! -= config.lr * dlogits[v]! * (x2t[j] ?? 0);

      // Update embTable (approximate gradient through attention)
      for (let j = 0; j < embedDim; j++)
        embTable[tok]![j]! -= config.lr * dX2[j]!;
    }
    steps++;
  }

  if (epoch % 100 === 0 || epoch === config.epochs - 1)
    console.log(`Epoch ${String(epoch).padStart(3)} — Loss: ${(totalLoss / steps).toFixed(4)}`);
}

// ── Text generation ──
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
console.log(`"the" → ${generate("the", 50, 0.8)}`);
console.log(`"a"   → ${generate("a", 50, 0.8)}`);

const trainedTokens = encoded.slice(0, blockSize);
const trainedTargets = encoded.slice(1, blockSize + 1);
const { logits: finalLogits } = forward(trainedTokens);
console.log(`\nFinal loss: ${crossEntropy(finalLogits, trainedTargets).toFixed(4)}`);

console.log("\n=== Architecture ===");
console.log(`  embTable  [${vocabSize} × ${embedDim}]   — trained`);
console.log(`  posTable  [${blockSize} × ${embedDim}]   — fixed (not trained)`);
console.log(`  Wq/Wk/Wv  [${embedDim} × ${embedDim}]  — frozen (backprop through attention = next lesson)`);
console.log(`  Wproj     [${vocabSize} × ${embedDim}]   — trained`);
console.log(`  Trained params: ${(vocabSize * embedDim * 2).toLocaleString()}`);
console.log(`\n  This is the same structure as GPT-1.`);
console.log(`  Claude uses ~100 layers, 8192 dims, 100k vocab, trillions of training tokens.`);

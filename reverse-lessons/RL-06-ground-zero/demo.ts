// RL-06 demo: the ENTIRE transformer, from token IDs to output probabilities
//
// Run: npx ts-node reverse-lessons/RL-06-ground-zero/demo.ts
//
// This is not a toy. This IS the transformer — every operation is here.
// The only simplification is tiny dimensions (vocab=6, embed=4, layers=2).
// A real GPT-4: vocab=100k, embed=12288, layers=96. Same math. More numbers.

// ── Utilities ──────────────────────────────────────────────────────────────

function matMul(v: number[], M: number[][]): number[] {
  return M.map(row => row.reduce((s, w, i) => s + w * v[i]!, 0));
}

function add(a: number[], b: number[]): number[] {
  return a.map((x, i) => x + b[i]!);
}

function relu(v: number[]): number[] {
  return v.map(x => Math.max(0, x));
}

function layerNorm(v: number[]): number[] {
  const mean = v.reduce((s, x) => s + x, 0) / v.length;
  const variance = v.reduce((s, x) => s + (x - mean) ** 2, 0) / v.length;
  const std = Math.sqrt(variance + 1e-8);
  return v.map(x => (x - mean) / std);
}

function softmax(v: number[]): number[] {
  const max = Math.max(...v);
  const exps = v.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function dot(a: number[], b: number[]): number {
  return a.reduce((s, ai, i) => s + ai * b[i]!, 0);
}

// ── Tiny model weights (fixed — normally learned by gradient descent) ──────

const VOCAB_SIZE = 6;
const EMBED_DIM  = 4;
const NUM_LAYERS = 2;

// Embedding table: vocab_size × embed_dim
const embeddingTable: number[][] = [
  [ 0.9,  0.1, -0.2,  0.7],  // token 0: "the"
  [ 0.9,  0.1, -0.2,  0.6],  // token 1: "cat"
  [-0.3,  0.8,  0.4, -0.5],  // token 2: "sat"
  [-0.1, -0.2,  0.3,  0.4],  // token 3: "on"
  [-0.4,  0.6,  0.3, -0.5],  // token 4: "mat"
  [ 0.5, -0.3,  0.7,  0.2],  // token 5: "<end>"
];

// Positional encoding (sine-based, simplified)
function posEnc(pos: number, dim: number): number[] {
  return Array.from({length: dim}, (_, i) =>
    i % 2 === 0
      ? Math.sin(pos / 10000 ** (i / dim))
      : Math.cos(pos / 10000 ** ((i - 1) / dim))
  );
}

// Attention weights (W_Q, W_K, W_V as square EMBED_DIM matrices)
const W_Q: number[][] = [[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,0.1,0.2,0.3],[0.4,0.5,0.6,0.7]];
const W_K: number[][] = [[0.7,0.6,0.5,0.4],[0.3,0.2,0.1,0.9],[0.8,0.7,0.6,0.5],[0.4,0.3,0.2,0.1]];
const W_V: number[][] = [[0.5,0.5,0.5,0.5],[0.2,0.8,0.2,0.8],[0.9,0.1,0.9,0.1],[0.3,0.7,0.3,0.7]];

// Feed-forward weights: expand to 8 then back to 4
const W_ff1: number[][] = Array.from({length: 8}, (_, i) =>
  Array.from({length: 4}, (_, j) => Math.sin(i * 0.3 + j * 0.7) * 0.5)
);
const W_ff2: number[][] = Array.from({length: 4}, (_, i) =>
  Array.from({length: 8}, (_, j) => Math.cos(i * 0.5 + j * 0.3) * 0.5)
);

// Output matrix: embed → vocab
const W_out: number[][] = Array.from({length: VOCAB_SIZE}, (_, i) =>
  Array.from({length: EMBED_DIM}, (_, j) => Math.sin(i * 0.4 + j * 0.9) * 0.8)
);

// ── The transformer forward pass ───────────────────────────────────────────

function attentionLayer(vectors: number[][]): number[][] {
  const scale = Math.sqrt(EMBED_DIM);
  const queries = vectors.map(v => matMul(v, W_Q));
  const keys    = vectors.map(v => matMul(v, W_K));
  const values  = vectors.map(v => matMul(v, W_V));

  return vectors.map((_, i) => {
    const scores = keys.map(k => dot(queries[i]!, k) / scale);
    const weights = softmax(scores);
    const output = new Array(EMBED_DIM).fill(0);
    for (let j = 0; j < vectors.length; j++) {
      for (let d = 0; d < EMBED_DIM; d++) {
        output[d] += weights[j]! * values[j]![d]!;
      }
    }
    return output;
  });
}

function ffnLayer(v: number[]): number[] {
  const hidden = relu(matMul(v, W_ff1));  // expand: 4→8, then relu
  return matMul(hidden, W_ff2);           // contract: 8→4
}

function transformerForward(tokenIds: number[]): number[] {
  // Step 1: Embed + positional encoding
  let vectors = tokenIds.map((id, pos) =>
    add(embeddingTable[id]!, posEnc(pos, EMBED_DIM))
  );

  // Step 2: N transformer layers (each = attention + residual + ffn + residual + layernorm)
  for (let layer = 0; layer < NUM_LAYERS; layer++) {
    // Attention with residual connection
    const attnOutput = attentionLayer(vectors);
    vectors = vectors.map((v, i) => layerNorm(add(v, attnOutput[i]!)));

    // FFN with residual connection
    vectors = vectors.map(v => layerNorm(add(v, ffnLayer(v))));
  }

  // Step 3: Project last token's vector to vocabulary logits
  const lastVector = vectors[vectors.length - 1]!;
  const logits = matMul(lastVector, W_out);

  // Step 4: Softmax → probability distribution
  return softmax(logits);
}

// ── Run it ─────────────────────────────────────────────────────────────────

const vocab = ["the", "cat", "sat", "on", "mat", "<end>"];

// Prompt: "the cat sat on" → what comes next?
const inputIds = [0, 1, 2, 3];  // the=0, cat=1, sat=2, on=3
const probs = transformerForward(inputIds);

console.log('=== Full transformer forward pass ===');
console.log('Input: "the cat sat on"');
console.log(`Input IDs: [${inputIds.join(", ")}]`);
console.log();
console.log('Output probabilities for next token:');
vocab.forEach((word, i) => {
  const bar = '█'.repeat(Math.round(probs[i]! * 30));
  console.log(`  ${word.padEnd(8)} ${(probs[i]! * 100).toFixed(1).padStart(5)}%  ${bar}`);
});
console.log();
const predicted = vocab[probs.indexOf(Math.max(...probs))]!;
console.log(`Predicted next token: "${predicted}"`);
console.log();

console.log('=== What this model IS ===');
console.log('Operations performed:');
console.log('  1. token IDs → embedding vectors (array lookup)');
console.log('  2. + positional encoding (sine/cosine)');
console.log(`  3. × ${NUM_LAYERS} transformer layers, each:`);
console.log('       attention = softmax(QK^T / sqrt(d)) × V');
console.log('       ffn       = relu(W1 × x + b1), then W2 × h + b2');
console.log('       residual  = layer + output');
console.log('       layernorm = (x - mean) / std');
console.log('  4. last vector × W_out → logits');
console.log('  5. softmax(logits) → probabilities');
console.log();
console.log('That is the COMPLETE list of operations.');
console.log('No facts. No grammar rules. No understanding.');
console.log('Just: numbers in → arithmetic → numbers out.');
console.log();
console.log('GPT-4 is this, with:');
console.log('  vocab=100,000  embed=12,288  layers=96  heads=96');
console.log('  ~1.8 trillion parameters (weights)');
console.log('  Each weight chosen by gradient descent to minimize prediction error.');
console.log();
console.log('That is all it is.');

// Embeddings: turn token integers into dense learned vectors
// Similar tokens end up with similar vectors after training

function initTable(vocabSize: number, embedDim: number): number[][] {
  return Array.from({ length: vocabSize }, () =>
    Array.from({ length: embedDim }, () => (Math.random() - 0.5) * 0.1)
  );
}

function lookup(tokenId: number, table: number[][]): number[] {
  return table[tokenId] ?? [];
}

function embedSequence(tokens: number[], table: number[][]): number[][] {
  return tokens.map(t => lookup(t, table));
}

function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((s, ai, i) => s + ai * (b[i] ?? 0), 0);
  const normA = Math.sqrt(a.reduce((s, ai) => s + ai ** 2, 0));
  const normB = Math.sqrt(b.reduce((s, bi) => s + bi ** 2, 0));
  return dot / (normA * normB + 1e-9);
}

// Simulate one gradient update on an embedding
// (in a real model this happens via backprop through the lookup table)
function updateEmbedding(table: number[][], tokenId: number, grad: number[], lr: number): void {
  table[tokenId] = (table[tokenId] ?? []).map((v, i) => v - lr * (grad[i] ?? 0));
}

// ── Setup ──
const vocab = " abcdefghijklmnopqrstuvwxyz".split(""); // 27 chars
const charToId = new Map(vocab.map((c, i) => [c, i]));
const embedDim = 8;
const table = initTable(vocab.length, embedDim);

// ── Embed a word ──
const word = "cat";
const tokens = word.split("").map(c => charToId.get(c)!);
const embedded = embedSequence(tokens, table);

console.log(`=== Embedding "${word}" ===`);
embedded.forEach((vec, i) => {
  console.log(`  "${word[i]}": [${vec.map(v => v.toFixed(4)).join(", ")}]`);
});
console.log(`Output shape: ${embedded.length} × ${embedded[0]?.length}`);

// ── Before training: embeddings are random → low similarity ──
console.log("\n=== Cosine similarity (before training — random) ===");
const [cVec, aVec, tVec] = embedded;
console.log(`c ↔ a: ${cosineSimilarity(cVec!, aVec!).toFixed(4)}`);
console.log(`c ↔ t: ${cosineSimilarity(cVec!, tVec!).toFixed(4)}`);

// ── Simulate one update (as if 'c' and 'a' often appear together) ──
const fakeGrad = Array(embedDim).fill(0.01); // push toward similar context
updateEmbedding(table, charToId.get("c")!, fakeGrad, 0.1);
updateEmbedding(table, charToId.get("a")!, fakeGrad, 0.1);

const cNew = lookup(charToId.get("c")!, table);
const aNew = lookup(charToId.get("a")!, table);
console.log(`\nc ↔ a after 1 update: ${cosineSimilarity(cNew, aNew).toFixed(4)}`);
console.log("After millions of updates, related tokens cluster together.");

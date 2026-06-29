// RL-02 demo: embeddings are coordinates, not meanings
//
// Run: npx ts-node reverse-lessons/RL-02-meaning-is-position/demo.ts

// Tiny embedding table (vocab_size=6, embed_dim=4)
// These are random — just like a freshly initialized model has random embeddings.
// The model would learn to adjust these during training.
const EMBED_DIM = 4;
const embeddingTable: Record<string, number[]> = {
  "the":  [ 0.23, -0.41,  0.88,  0.12],
  "cat":  [ 0.91,  0.13, -0.22,  0.67],
  "dog":  [ 0.88,  0.17, -0.19,  0.71],  // ← close to "cat" (similar contexts)
  "sat":  [-0.33,  0.79,  0.44, -0.55],
  "on":   [-0.11, -0.22,  0.33,  0.44],
  "mat":  [-0.45,  0.62,  0.31, -0.48],
};

function embed(token: string): number[] {
  return embeddingTable[token] ?? new Array(EMBED_DIM).fill(0);
}

function dotProduct(a: number[], b: number[]): number {
  return a.reduce((sum, ai, i) => sum + ai * b[i], 0);
}

function magnitude(v: number[]): number {
  return Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
}

function cosineSimilarity(a: number[], b: number[]): number {
  return dotProduct(a, b) / (magnitude(a) * magnitude(b));
}

console.log("=== Embedding table (each word = coordinate vector) ===");
for (const [word, vec] of Object.entries(embeddingTable)) {
  console.log(`  "${word}"  →  [${vec.map(x => x.toFixed(2)).join(", ")}]`);
}
console.log();

// Show that "cat" and "dog" are close, while "cat" and "sat" are far
const pairs = [
  ["cat", "dog"],   // both animals → should be close
  ["cat", "sat"],   // one animal, one verb → should be far
  ["cat", "mat"],   // one animal, one object → should be far
];

console.log("=== Cosine similarity (1.0 = identical, 0.0 = unrelated) ===");
for (const [a, b] of pairs) {
  const sim = cosineSimilarity(embed(a), embed(b));
  console.log(`  similarity("${a}", "${b}") = ${sim.toFixed(3)}`);
}
console.log();

console.log("Key insight:");
console.log("  'cat' and 'dog' are close NOT because the model knows they're animals.");
console.log("  They're close because they appear in similar sentence positions.");
console.log("  ('The cat sat', 'The dog sat', 'A cat ran', 'A dog ran' ...)");
console.log();

// Prove the point: if we renamed everything, the model would work the same
console.log("=== What if we renamed everything? ===");
const renamed: Record<string, number[]> = {
  "XQVBZM": embeddingTable["cat"]!,  // "cat" renamed to "XQVBZM"
  "RFPKLA": embeddingTable["dog"]!,  // "dog" renamed to "RFPKLA"
};
const sim = cosineSimilarity(renamed["XQVBZM"]!, renamed["RFPKLA"]!);
console.log(`  similarity("XQVBZM", "RFPKLA") = ${sim.toFixed(3)}`);
console.log("  Same similarity. Same geometry.");
console.log("  The model doesn't care what the words mean to YOU.");
console.log("  It only cares about the coordinates.");

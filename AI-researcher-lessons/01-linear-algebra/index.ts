// Linear algebra from scratch: the operations every transformer runs on.
// dot product, matmul, transpose, norms, cosine similarity, and
// power iteration (finding the dominant eigenvector by just... multiplying).

// ── Core operations ──────────────────────────────────────────────

// Dot product: multiply matching entries, add them up.
// Geometrically: a similarity meter (positive = similar direction).
function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] ?? 0) * (b[i] ?? 0);
  return sum;
}

// L2 norm: the length of the arrow (Pythagoras in n dimensions).
function norm(v: number[]): number {
  return Math.sqrt(dot(v, v));
}

// Cosine similarity: dot product with lengths divided out.
// Always in [-1, 1]. THE standard way to compare embeddings.
function cosineSimilarity(a: number[], b: number[]): number {
  return dot(a, b) / (norm(a) * norm(b));
}

// Matrix multiply: C[i][j] = (row i of A) · (col j of B).
// Shapes: (n×m) @ (m×p) → (n×p). Inner dimensions must match.
function matmul(A: number[][], B: number[][]): number[][] {
  const rows = A.length, cols = B[0]?.length ?? 0, inner = B.length;
  const C = Array.from({ length: rows }, () => Array<number>(cols).fill(0));
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++)
      for (let k = 0; k < inner; k++)
        C[i]![j]! += (A[i]?.[k] ?? 0) * (B[k]?.[j] ?? 0);
  return C;
}

// Transpose: flip rows and columns. (Aᵀ)[i][j] = A[j][i].
function transpose(A: number[][]): number[][] {
  return (A[0] ?? []).map((_, j) => A.map(row => row[j] ?? 0));
}

// Matrix-vector product: transform a vector through the matrix-machine.
// Each output entry = dot product of one row of M with v.
function matvec(M: number[][], v: number[]): number[] {
  return M.map(row => dot(row, v));
}

const fmt = (v: number[]) => `[${v.map(x => x.toFixed(4)).join(", ")}]`;

// ── Demo 1: dot product as similarity ────────────────────────────
console.log("=== 1. Dot product: the similarity meter ===");
const a = [1, 2, 3];
const b = [4, 5, 6];
console.log(`a = [1,2,3], b = [4,5,6]`);
console.log(`a·b = 1×4 + 2×5 + 3×6 = ${dot(a, b)}  (positive → similar direction)`);
console.log(`[1,0]·[0,1] = ${dot([1, 0], [0, 1])}  (zero → orthogonal, "unrelated")`);
console.log(`[1,2]·[-1,-2] = ${dot([1, 2], [-1, -2])}  (negative → opposite direction)`);

// ── Demo 2: norms and cosine similarity on toy "embeddings" ─────
console.log("\n=== 2. Norms and cosine similarity ===");
console.log(`‖[3,4]‖ = √(9+16) = ${norm([3, 4])}  (Pythagoras)`);

// Toy 4-dim embeddings. Dims could mean: [furry, purrs, has-engine, has-wheels]
const cat = [0.9, 0.8, 0.0, 0.1];
const kitten = [1.0, 0.9, 0.0, 0.0];
const truck = [0.0, 0.0, 0.9, 1.0];
console.log(`cos(cat, kitten) = ${cosineSimilarity(cat, kitten).toFixed(4)}  (≈1 → same concept)`);
console.log(`cos(cat, truck)  = ${cosineSimilarity(cat, truck).toFixed(4)}  (≈0 → unrelated)`);
// Scaling doesn't change cosine — direction is what matters:
const shoutedCat = cat.map(x => x * 100);
console.log(`cos(cat, 100×cat) = ${cosineSimilarity(cat, shoutedCat).toFixed(4)}  (length ignored)`);

// ── Demo 3: matmul and transpose ─────────────────────────────────
console.log("\n=== 3. Matrix multiply: chaining machines ===");
const A = [
  [1, 2],
  [3, 4],
];
const B = [
  [5, 6],
  [7, 8],
];
const C = matmul(A, B);
console.log("A = [[1,2],[3,4]], B = [[5,6],[7,8]]");
console.log(`A@B = [${C.map(r => `[${r.join(",")}]`).join(", ")}]   (C[0][0] = 1×5+2×7 = 19)`);
const BA = matmul(B, A);
console.log(`B@A = [${BA.map(r => `[${r.join(",")}]`).join(", ")}]   (different! order matters)`);
const At = transpose([[1, 2, 3], [4, 5, 6]]);
console.log(`transpose([[1,2,3],[4,5,6]]) = [${At.map(r => `[${r.join(",")}]`).join(", ")}]  (2×3 → 3×2)`);

// ── Demo 4: matrices transform vectors ───────────────────────────
console.log("\n=== 4. A matrix is a machine: vector in, vector out ===");
const stretch = [
  [2, 0],
  [0, 3],
];
console.log(`[[2,0],[0,3]] · [1,1] = ${fmt(matvec(stretch, [1, 1]))}  (stretch x by 2, y by 3)`);
const rotate90 = [
  [0, -1],
  [1, 0],
];
console.log(`[[0,-1],[1,0]] · [1,0] = ${fmt(matvec(rotate90, [1, 0]))}  (rotated 90°: x-axis → y-axis)`);

// ── Demo 5: power iteration — finding the dominant eigenvector ──
// Idea: multiply ANY starting vector by M over and over.
// The component along the biggest-|eigenvalue| direction grows fastest,
// so the vector swings toward the dominant eigenvector. Normalize each
// step so numbers don't explode. That's the whole algorithm.
// (Google's original PageRank was exactly this, on the web-link matrix.)
console.log("\n=== 5. Power iteration: what stays pointing the same way ===");
const M = [
  [2, 1],
  [1, 2],
];
// True answer (solvable by hand for 2×2): eigenvalues 3 and 1,
// dominant eigenvector [1,1]/√2 ≈ [0.7071, 0.7071], eigenvalue 3.
console.log("M = [[2,1],[1,2]]   (true dominant: eigenvector [0.7071, 0.7071], eigenvalue 3)");

let v = [1, 0]; // deliberately start pointing the "wrong" way
console.log(`start: v = ${fmt(v)}`);
let lambda = 0;
for (let step = 1; step <= 8; step++) {
  const Mv = matvec(M, v);
  // Rayleigh quotient: current best estimate of the eigenvalue.
  // λ ≈ v·(Mv) / v·v — "how much did M stretch v along itself?"
  lambda = dot(v, Mv) / dot(v, v);
  const len = norm(Mv);
  v = Mv.map(x => x / len); // renormalize to unit length
  console.log(`step ${step}: v = ${fmt(v)}   λ estimate = ${lambda.toFixed(6)}`);
}

// Verify: M·v should equal λ·v (same direction, stretched by λ)
const Mv = matvec(M, v);
console.log(`\ncheck  M·v = ${fmt(Mv)}`);
console.log(`check  λ·v = ${fmt(v.map(x => x * lambda))}`);
console.log("→ M·v ≈ λ·v: the machine no longer rotates v, only stretches it.");
console.log("→ Repeated matmul converges to the dominant eigen-direction —");
console.log("  the same math behind exploding/vanishing gradients in deep nets.");

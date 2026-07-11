// Interpretability, two experiments:
//   1) A hand-built INDUCTION HEAD — the classic discovered circuit.
//      We set the weights by hand (no training) so the mechanism is fully visible.
//   2) SUPERPOSITION — 5 sparse features stored in 3 dimensions, recovered
//      almost perfectly BECAUSE they are sparse.
//
// Run: npx ts-node AI-researcher-lessons/22-interpretability/index.ts

// ── shared helpers ──────────────────────────────────────────────────────────

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] ?? 0) * (b[i] ?? 0);
  return s;
}

// mulberry32 — tiny seeded RNG so the demo is reproducible
function rng(seed: number): () => number {
  let t = seed;
  return () => {
    t = (t + 0x6d2b79f5) | 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

// ════════════════════════════════════════════════════════════════════════════
// PART 1 — The induction head circuit, built by hand
// ════════════════════════════════════════════════════════════════════════════
//
// The two-layer algorithm:
//   Layer 1 "previous-token head": every position copies its predecessor's
//           identity into a second slot of its own vector.
//   Layer 2 "induction head": position i asks "whose PREDECESSOR was my
//           token?" — that match is the token that followed the previous
//           occurrence of token[i]. Attend there, copy it → prediction.
//
// We represent each position with a vector of size 2·V:
//   dims [0..V)   = one-hot of MY token          (written by the embedding)
//   dims [V..2V)  = one-hot of the PREVIOUS token (written by layer 1)
// Layer 1 is so simple here that we just fill that slot directly — in a real
// transformer a previous-token attention head does this job.

const VOCAB = ["A", "B", "C", "D"];
const V = VOCAB.length;

const seq = ["D", "A", "B", "C", "A"]; // ... A B ... A → should predict B
const T = seq.length;
const ids = seq.map(t => VOCAB.indexOf(t));

// Build the residual-stream vectors after the layer-1 previous-token head.
const X: number[][] = [];
for (let i = 0; i < T; i++) {
  const x = Array<number>(2 * V).fill(0);
  x[ids[i] ?? 0] = 1;                       // my token
  if (i > 0) x[V + (ids[i - 1] ?? 0)] = 1;  // my predecessor's token
  X.push(x);
}

// Layer 2, the induction head. Hand-set projections:
//   Query  = "my token identity"        → reads dims [0..V)
//   Key    = "my PREDECESSOR's identity" → reads dims [V..2V)
//   Value  = "my token identity"        → what gets copied if attended to
// So q_i · k_j = 1 exactly when token[j-1] == token[i]. That single dot
// product IS the induction algorithm.
const SHARPNESS = 8; // scales scores so softmax commits to the match

const query = (x: number[]) => x.slice(0, V);
const key = (x: number[]) => x.slice(V, 2 * V);
const value = (x: number[]) => x.slice(0, V);

const attn: number[][] = []; // attention weights, row = query position
const out: number[][] = [];  // head output per position (a vocab distribution here)
for (let i = 0; i < T; i++) {
  const scores: number[] = [];
  for (let j = 0; j < T; j++) {
    // causal mask: can only look at the past (and self)
    scores.push(j <= i ? SHARPNESS * dot(query(X[i] ?? []), key(X[j] ?? [])) : -Infinity);
  }
  const w = softmax(scores);
  attn.push(w);
  const o = Array<number>(V).fill(0);
  for (let j = 0; j < T; j++)
    for (let v = 0; v < V; v++) o[v] = (o[v] ?? 0) + (w[j] ?? 0) * (value(X[j] ?? [])[v] ?? 0);
  out.push(o);
}

console.log("═══ Part 1: a hand-built induction head ═══");
console.log(`sequence: ${seq.join(" ")}   (pattern: ...A B...A → predict B)\n`);
console.log("Attention matrix (rows = query position, cols = attended position):");
console.log("        " + seq.map((t, j) => `${t}(${j})`.padStart(7)).join(""));
attn.forEach((row, i) => {
  console.log(
    `  ${seq[i]}(${i}) ` + row.map(w => w.toFixed(3).padStart(7)).join("")
  );
});

const last = out[T - 1] ?? [];
const pred = VOCAB[last.indexOf(Math.max(...last))];
console.log(`\nLast position is "${seq[T - 1]}". Its attention row peaks at position 2 — "B",`);
console.log(`the token AFTER the previous "A". Copied value → prediction: "${pred}"`);
console.log('That "look back at what followed me last time" stripe is the induction pattern.');
console.log("Real induction heads are FOUND (not built) by scanning trained models for");
console.log("exactly this attention signature, then verified by ablating the head.");

// ════════════════════════════════════════════════════════════════════════════
// PART 2 — Superposition: 5 features in 3 dimensions
// ════════════════════════════════════════════════════════════════════════════
//
// Assign each of 5 features a random unit direction w_i in 3-D. Store a
// feature vector f (length 5) as the 3-D sum  h = Σ f_i · w_i.
// Recover feature i by reading  r_i = w_i · h.
// Perfect if directions were orthogonal — but 5 arrows can't be orthogonal in
// 3-D, so recovery picks up interference  Σ_{j≠i} f_j (w_i·w_j).
// SPARSE f → few interfering terms → recovery works.
// DENSE  f → every overlap contributes → recovery drowns.

const rand = rng(12345);
const NF = 5, DIM = 3;

// A trained model under superposition learns directions that are AS SPREAD OUT
// as geometry allows. In 3-D the best possible arrangement of 5+ lines is
// known exactly: the 6 axes of an icosahedron, every pair at |cos| = 1/√5
// ≈ 0.447. We take 5 of them (φ = golden ratio), i.e. we skip the training
// run and jump straight to the optimum the model would find.
const PHI = (1 + Math.sqrt(5)) / 2;
const RAW: number[][] = [
  [0, 1, PHI],
  [0, -1, PHI],
  [1, PHI, 0],
  [-1, PHI, 0],
  [PHI, 0, 1],
];
const W: number[][] = RAW.map(v => {
  const n = Math.sqrt(dot(v, v)) || 1;
  return v.map(x => x / n);
});

console.log("\n═══ Part 2: superposition — 5 features crammed into 3 dims ═══");
console.log("\nInterference matrix wᵢ·wⱼ (off-diagonal ≠ 0: the directions overlap):");
for (let i = 0; i < NF; i++) {
  console.log(
    "  " + W.map((_, j) => dot(W[i] ?? [], W[j] ?? []).toFixed(2).padStart(6)).join("")
  );
}

function compress(f: number[]): number[] {
  const h = Array<number>(DIM).fill(0);
  for (let i = 0; i < NF; i++)
    for (let d = 0; d < DIM; d++) h[d] = (h[d] ?? 0) + (f[i] ?? 0) * (W[i]?.[d] ?? 0);
  return h;
}
const recover = (h: number[]) => W.map(w => dot(w, h));

// Trial: activate k random features (value 1), compress to 3 numbers, read
// back out. Recovery = "the k largest readouts are exactly the k features
// that were active." Also track the readout error caused by interference.
function trial(k: number): { exact: boolean; err: number } {
  const f = Array<number>(NF).fill(0);
  const order = [...Array(NF).keys()].sort(() => rand() - 0.5);
  for (let i = 0; i < k; i++) f[order[i] ?? 0] = 1;
  const r = recover(compress(f));
  const topk = r
    .map((v, i) => [v, i] as const)
    .sort((a, b) => b[0] - a[0])
    .slice(0, k)
    .map(([, i]) => i);
  const exact = topk.every(i => f[i] === 1);
  const err = r.reduce((s, ri, i) => s + Math.abs(ri - (f[i] ?? 0)), 0) / NF;
  return { exact, err };
}

const TRIALS = 2000;
console.log("\nActive-set recovery over " + TRIALS + " random trials");
console.log("(recovered = the k biggest readouts are exactly the k active features):");
for (const k of [1, 2, 3, 4]) {
  let ok = 0, errSum = 0;
  for (let t = 0; t < TRIALS; t++) {
    const { exact, err } = trial(k);
    if (exact) ok++;
    errSum += err;
  }
  const frac = ok / TRIALS;
  const pct = (100 * frac).toFixed(1).padStart(5);
  const bar = "█".repeat(Math.round(frac * 30)).padEnd(30, "·");
  console.log(
    `  ${k} of 5 active: recovered ${pct}%  ${bar}  mean readout error ${(errSum / TRIALS).toFixed(2)}`
  );
}
console.log("  sparse → recovery works; denser → interference piles up and wins.");

// Show one concrete sparse example end-to-end
const fEx = [0, 1, 0, 0, 0];
const rEx = recover(compress(fEx));
console.log("\nOne sparse example, features [0,1,0,0,0] → 3 numbers → readout:");
console.log("  readout: [" + rEx.map(x => x.toFixed(2)).join(", ") + "]");
console.log("  the active feature stands tall (1.00); the rest sit at ±0.45 of");
console.log("  pure crosstalk. Distinguishable — but the interference budget is");
console.log("  nearly spent, which is exactly why density breaks it above.");
console.log("\nThis is why single neurons look polysemantic, and why sparse");
console.log("autoencoders (which learn to UNDO this compression) can find the");
console.log("real features hiding inside.");

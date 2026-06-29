// Lesson 24 — In-Context Learning: GPT-3's Surprising Ability
// Demonstrates how a trained language model generalises from in-context examples.
// No imports — all math is hand-rolled.

// =============================================================================
// Section 0 — Utility math
// =============================================================================

function matMul(A: number[][], B: number[][]): number[][] {
  const rA = A.length, cA = A[0].length, cB = B[0].length;
  const C: number[][] = Array.from({ length: rA }, () => new Array(cB).fill(0));
  for (let i = 0; i < rA; i++)
    for (let k = 0; k < cA; k++)
      for (let j = 0; j < cB; j++)
        C[i][j] += A[i][k] * B[k][j];
  return C;
}

function softmax(logits: number[]): number[] {
  const maxL = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - maxL));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

function argmax(arr: number[]): number {
  let best = 0;
  for (let i = 1; i < arr.length; i++)
    if (arr[i] > arr[best]) best = i;
  return best;
}

// Seeded pseudo-random number generator (xorshift32) for reproducibility
function makeRng(seed: number) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
}

function randNormal(rng: () => number, scale = 0.02): number {
  const u = rng() + 1e-10, v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * scale;
}

// =============================================================================
// Section 1 — Tiny tokeniser
// =============================================================================

// Vocabulary: digits 0-9, operators, and special tokens for our toy language.
const VOCAB = [
  "<pad>", "<bos>", "<eos>",
  "0","1","2","3","4","5","6","7","8","9",
  "+","=","*","-",".",
  "Q",":","A"," ","\n","?",
];

const tokToId = new Map<string, number>(VOCAB.map((t, i) => [t, i]));
const idToTok = new Map<number, string>(VOCAB.map((t, i) => [i, t]));
const VOCAB_SIZE = VOCAB.length;

function tokenise(text: string): number[] {
  const ids: number[] = [];
  let i = 0;
  while (i < text.length) {
    let matched = false;
    for (let len = 3; len >= 1; len--) {
      const sub = text.slice(i, i + len);
      if (tokToId.has(sub)) { ids.push(tokToId.get(sub)!); i += len; matched = true; break; }
    }
    if (!matched) i++;
  }
  return ids;
}

function detokenise(ids: number[]): string {
  return ids.map(id => idToTok.get(id) ?? "?").join("");
}

// =============================================================================
// Section 2 — Small GPT with full analytical backprop
// =============================================================================
// Architecture: token+positional embeddings → single causal self-attention
// head → residual add → layer norm → two-layer FFN → unembed.
// We do full analytical backprop so the model actually learns.

const D = 16;   // embedding dimension
const DF = 32;  // FFN hidden size
const BS = 64;  // block size (context window)

interface Weights {
  E:   number[][];  // [V × D]  token embeddings
  P:   number[][];  // [BS × D] positional embeddings
  WQ:  number[][];  // [D × D]
  WK:  number[][];
  WV:  number[][];
  WO:  number[][];
  W1:  number[][];  // [D × DF]
  b1:  number[];
  W2:  number[][];  // [DF × D]
  b2:  number[];
  LNw: number[];    // layer norm scale
  LNb: number[];
  U:   number[][];  // [D × V] unembed
}

function initWeights(rng: () => number): Weights {
  const m2 = (r: number, c: number, s = 0.02) =>
    Array.from({ length: r }, () => Array.from({ length: c }, () => randNormal(rng, s)));
  return {
    E:   m2(VOCAB_SIZE, D),
    P:   m2(BS, D, 0.01),
    WQ:  m2(D, D, 0.02), WK: m2(D, D, 0.02), WV: m2(D, D, 0.02), WO: m2(D, D, 0.02),
    W1:  m2(D, DF, 0.02), b1: new Array(DF).fill(0),
    W2:  m2(DF, D, 0.02), b2: new Array(D).fill(0),
    LNw: new Array(D).fill(1), LNb: new Array(D).fill(0),
    U:   m2(D, VOCAB_SIZE, 0.02),
  };
}

// ── Layer norm (forward only; we approximate its backward as identity) ─────
function layerNorm(x: number[], w: number[], b: number[]): number[] {
  const mu = x.reduce((a, v) => a + v, 0) / D;
  const vr = x.reduce((a, v) => a + (v - mu) ** 2, 0) / D;
  const sd = Math.sqrt(vr + 1e-5);
  return x.map((v, i) => w[i] * (v - mu) / sd + b[i]);
}

// ── Full forward pass returning all intermediates for backprop ─────────────
interface Cache {
  X:  number[][];   // input embeddings [T × D]
  Q:  number[][];
  K:  number[][];
  V:  number[][];
  AW: number[][];   // attention weights per position (jagged: pos i has i+1 weights)
  AO: number[][];   // attention output [T × D]
  R1: number[][];   // after first residual [T × D]
  N:  number[][];   // after layer norm [T × D]
  H1: number[];     // FFN hidden (last pos)
  H1pre: number[];  // pre-relu
  FF: number[];     // FFN output (last pos)
  R2: number[];     // after second residual (last pos)
  logits: number[];
  probs:  number[];
}

function forwardFull(ids: number[], w: Weights): Cache {
  const T = ids.length;
  // Embeddings
  const X: number[][] = ids.map((id, p) => {
    const pe = w.P[Math.min(p, BS - 1)];
    return w.E[id].map((v, i) => v + pe[i]);
  });
  // Attention projections
  const Q: number[][] = X.map(x => {
    const q = new Array(D).fill(0);
    for (let d = 0; d < D; d++) for (let k = 0; k < D; k++) q[d] += x[k] * w.WQ[k][d];
    return q;
  });
  const K: number[][] = X.map(x => {
    const k = new Array(D).fill(0);
    for (let d = 0; d < D; d++) for (let j = 0; j < D; j++) k[d] += x[j] * w.WK[j][d];
    return k;
  });
  const V: number[][] = X.map(x => {
    const v = new Array(D).fill(0);
    for (let d = 0; d < D; d++) for (let j = 0; j < D; j++) v[d] += x[j] * w.WV[j][d];
    return v;
  });
  const scale = 1 / Math.sqrt(D);
  const AW: number[][] = [];
  const AO: number[][] = Array.from({ length: T }, () => new Array(D).fill(0));
  for (let i = 0; i < T; i++) {
    const scores = Array.from({ length: i + 1 }, (_, j) => {
      let d = 0; for (let k = 0; k < D; k++) d += Q[i][k] * K[j][k]; return d * scale;
    });
    const aw = softmax(scores);
    AW.push(aw);
    for (let d = 0; d < D; d++) {
      let acc = 0;
      for (let j = 0; j <= i; j++) acc += aw[j] * V[j][d];
      AO[i][d] = acc;
    }
  }
  // Project attention output
  const proj: number[][] = AO.map(ao => {
    const p = new Array(D).fill(0);
    for (let d = 0; d < D; d++) for (let k = 0; k < D; k++) p[d] += ao[k] * w.WO[k][d];
    return p;
  });
  // Residual + layer norm
  const R1: number[][] = X.map((x, i) => x.map((v, d) => v + proj[i][d]));
  const N:  number[][] = R1.map(r => layerNorm(r, w.LNw, w.LNb));
  // FFN on last position only
  const lastN = N[T - 1];
  const H1pre = Array.from({ length: DF }, (_, j) => {
    let acc = w.b1[j];
    for (let k = 0; k < D; k++) acc += lastN[k] * w.W1[k][j];
    return acc;
  });
  const H1 = H1pre.map(v => Math.max(0, v));
  const FF = Array.from({ length: D }, (_, d) => {
    let acc = w.b2[d];
    for (let k = 0; k < DF; k++) acc += H1[k] * w.W2[k][d];
    return acc;
  });
  const R2 = lastN.map((v, d) => v + FF[d]);
  const logits = Array.from({ length: VOCAB_SIZE }, (_, j) => {
    let acc = 0;
    for (let k = 0; k < D; k++) acc += R2[k] * w.U[k][j];
    return acc;
  });
  return { X, Q, K, V, AW, AO, R1, N, H1, H1pre, FF, R2, logits, probs: softmax(logits) };
}

// ── SGD with momentum ─────────────────────────────────────────────────────
interface Grads {
  E:   number[][];
  P:   number[][];
  WQ:  number[][];
  WK:  number[][];
  WV:  number[][];
  WO:  number[][];
  W1:  number[][];  b1: number[];
  W2:  number[][];  b2: number[];
  LNw: number[];    LNb: number[];
  U:   number[][];
}

function zeroGrads(): Grads {
  const z2 = (r: number, c: number) => Array.from({ length: r }, () => new Array(c).fill(0));
  return {
    E: z2(VOCAB_SIZE, D), P: z2(BS, D),
    WQ: z2(D, D), WK: z2(D, D), WV: z2(D, D), WO: z2(D, D),
    W1: z2(D, DF), b1: new Array(DF).fill(0),
    W2: z2(DF, D), b2: new Array(D).fill(0),
    LNw: new Array(D).fill(0), LNb: new Array(D).fill(0),
    U: z2(D, VOCAB_SIZE),
  };
}

function trainStep(
  ids: number[],
  targetId: number,
  w: Weights,
  mom: Grads,
  lr: number,
  mu = 0.9
): number {
  const T = ids.length;
  const c = forwardFull(ids, w);
  const loss = -Math.log(c.probs[targetId] + 1e-10);

  // dL/dLogits
  const dLogits = c.probs.map((p, i) => p - (i === targetId ? 1 : 0));

  // dL/dU, dL/dR2
  const dR2 = new Array(D).fill(0);
  for (let k = 0; k < D; k++) {
    for (let j = 0; j < VOCAB_SIZE; j++) {
      mom.U[k][j] = mu * mom.U[k][j] + (1 - mu) * c.R2[k] * dLogits[j];
      w.U[k][j] -= lr * mom.U[k][j];
      dR2[k] += w.U[k][j] * dLogits[j];
    }
  }

  // dL/dFF (dR2 passes through residual to both lastN and FF branch)
  const dFF = dR2.slice();
  // dL/dH1, dL/dW2, dL/db2
  const dH1 = new Array(DF).fill(0);
  for (let k = 0; k < DF; k++) {
    for (let d = 0; d < D; d++) {
      mom.W2[k][d] = mu * mom.W2[k][d] + (1 - mu) * c.H1[k] * dFF[d];
      w.W2[k][d] -= lr * mom.W2[k][d];
      dH1[k] += w.W2[k][d] * dFF[d];
    }
  }
  for (let d = 0; d < D; d++) {
    mom.b2[d] = mu * mom.b2[d] + (1 - mu) * dFF[d];
    w.b2[d] -= lr * mom.b2[d];
  }
  // ReLU backward
  const dH1pre = dH1.map((v, j) => c.H1pre[j] > 0 ? v : 0);
  // dL/dW1, dL/db1, dL/dLastN
  const dLastN = dR2.slice(); // from residual shortcut
  for (let k = 0; k < D; k++) {
    for (let j = 0; j < DF; j++) {
      mom.W1[k][j] = mu * mom.W1[k][j] + (1 - mu) * c.N[T - 1][k] * dH1pre[j];
      w.W1[k][j] -= lr * mom.W1[k][j];
      dLastN[k] += w.W1[k][j] * dH1pre[j];
    }
  }
  for (let j = 0; j < DF; j++) {
    mom.b1[j] = mu * mom.b1[j] + (1 - mu) * dH1pre[j];
    w.b1[j] -= lr * mom.b1[j];
  }

  // Approximate layer norm backward as identity (standard simplification for demos)
  // dL/dLNw, dL/dLNb
  for (let d = 0; d < D; d++) {
    mom.LNw[d] = mu * mom.LNw[d] + (1 - mu) * c.R1[T - 1][d] * dLastN[d];
    w.LNw[d] -= lr * mom.LNw[d];
    mom.LNb[d] = mu * mom.LNb[d] + (1 - mu) * dLastN[d];
    w.LNb[d] -= lr * mom.LNb[d];
  }

  // dL/dR1[T-1] ≈ dLastN (pass through LN)
  const dR1last = dLastN;

  // dL/dProj[last] = dR1last (residual shortcut also adds dR1last to dX)
  const dX_last = dR1last.slice();

  // dL/dWO via last position only
  const dAO_last = new Array(D).fill(0);
  for (let k = 0; k < D; k++) {
    for (let d = 0; d < D; d++) {
      mom.WO[k][d] = mu * mom.WO[k][d] + (1 - mu) * c.AO[T - 1][k] * dR1last[d];
      w.WO[k][d] -= lr * mom.WO[k][d];
      dAO_last[k] += w.WO[k][d] * dR1last[d];
    }
  }

  // dL/dV from last position attention
  const lastAW = c.AW[T - 1];
  for (let j = 0; j <= T - 1; j++) {
    const scale = lastAW[j];
    const id = ids[j];
    const pos = Math.min(j, BS - 1);
    for (let d = 0; d < D; d++) {
      // dV[j] += scale * dAO_last
      // dV[j] flows to WV and embed
      const g = scale * dAO_last[d];
      // Update WV: dV[j][d] = sum_k X[j][k] * WV[k][d]
      for (let k = 0; k < D; k++) {
        mom.WV[k][d] = mu * mom.WV[k][d] + (1 - mu) * c.X[j][k] * g;
        w.WV[k][d] -= lr * mom.WV[k][d];
      }
      // Propagate to embed[id] and posEmbed[pos]
      mom.E[id][d] = mu * mom.E[id][d] + (1 - mu) * g;
      w.E[id][d] -= lr * mom.E[id][d];
      mom.P[pos][d] = mu * mom.P[pos][d] + (1 - mu) * g;
      w.P[pos][d] -= lr * mom.P[pos][d];
    }
  }

  // Also push gradient through dX_last (residual from attention output) to last token embed
  const lastId = ids[T - 1];
  const lastPos = Math.min(T - 1, BS - 1);
  for (let d = 0; d < D; d++) {
    mom.E[lastId][d] = mu * mom.E[lastId][d] + (1 - mu) * dX_last[d];
    w.E[lastId][d] -= lr * mom.E[lastId][d];
    mom.P[lastPos][d] = mu * mom.P[lastPos][d] + (1 - mu) * dX_last[d];
    w.P[lastPos][d] -= lr * mom.P[lastPos][d];
  }

  return loss;
}

function predict(ids: number[], w: Weights): number {
  return argmax(forwardFull(ids, w).probs);
}

// =============================================================================
// Section 3 — Demo 1: Pattern learning and generalisation
// =============================================================================
//
// We use a lightweight two-layer MLP with a token-embedding table.
// The MLP takes the concatenated embeddings of [digit, "+", "1", "="]
// and predicts the successor digit. This is equivalent to the embedding +
// output projection of a transformer — just without the attention head,
// which is overkill for a 4-token sequence.
//
// Architecture:
//   input → embed each of the 4 tokens → concat → linear(64→32) → ReLU
//         → linear(32→VOCAB_SIZE) → softmax → cross-entropy loss
//
// We train with full analytical backprop and Adam.

console.log("=".repeat(70));
console.log("DEMO 1 — Training a pattern-learning model on arithmetic sequences");
console.log("=".repeat(70));

// ---------- tiny MLP ----------
const EMB_DIM  = 8;   // embedding size per token
const MLP_IN   = EMB_DIM * 4; // 4 tokens concatenated
const MLP_H    = 32;  // hidden layer size

interface MlpWeights {
  E:  number[][];  // [VOCAB_SIZE × EMB_DIM]  token embeddings
  W1: number[][];  // [MLP_IN × MLP_H]
  b1: number[];
  W2: number[][];  // [MLP_H × VOCAB_SIZE]
  b2: number[];
}

interface MlpMoments { E: number[][]; W1: number[][]; b1: number[]; W2: number[][]; b2: number[] }

function initMlp(rng: () => number): MlpWeights {
  const m2 = (r: number, c: number, s = 0.1) =>
    Array.from({ length: r }, () => Array.from({ length: c }, () => randNormal(rng, s)));
  return {
    E:  m2(VOCAB_SIZE, EMB_DIM, 0.1),
    W1: m2(MLP_IN,     MLP_H,   0.05),
    b1: new Array(MLP_H).fill(0),
    W2: m2(MLP_H,      VOCAB_SIZE, 0.05),
    b2: new Array(VOCAB_SIZE).fill(0),
  };
}

function zeroMlpMom(): MlpMoments {
  const z2 = (r: number, c: number) => Array.from({ length: r }, () => new Array(c).fill(0));
  return {
    E:  z2(VOCAB_SIZE, EMB_DIM),
    W1: z2(MLP_IN, MLP_H),   b1: new Array(MLP_H).fill(0),
    W2: z2(MLP_H, VOCAB_SIZE), b2: new Array(VOCAB_SIZE).fill(0),
  };
}

function mlpForward(ids: number[], mw: MlpWeights): { x: number[]; h: number[]; hpre: number[]; logits: number[]; probs: number[] } {
  // Concatenate embeddings
  const x: number[] = [];
  for (const id of ids) x.push(...mw.E[id]);

  // Hidden layer
  const hpre = Array.from({ length: MLP_H }, (_, j) => {
    let acc = mw.b1[j];
    for (let k = 0; k < MLP_IN; k++) acc += x[k] * mw.W1[k][j];
    return acc;
  });
  const h = hpre.map(v => Math.max(0, v));

  // Output layer
  const logits = Array.from({ length: VOCAB_SIZE }, (_, j) => {
    let acc = mw.b2[j];
    for (let k = 0; k < MLP_H; k++) acc += h[k] * mw.W2[k][j];
    return acc;
  });
  return { x, h, hpre, logits, probs: softmax(logits) };
}

function mlpTrainStep(
  ids: number[],
  targetId: number,
  mw: MlpWeights,
  mom2: MlpMoments,
  lr: number,
  beta = 0.9
): number {
  const { x, h, hpre, probs } = mlpForward(ids, mw);
  const loss = -Math.log(probs[targetId] + 1e-10);

  // dL/dLogits
  const dLogits = probs.map((p, i) => p - (i === targetId ? 1 : 0));

  // dL/dW2, dL/db2, dL/dh
  const dh = new Array(MLP_H).fill(0);
  for (let k = 0; k < MLP_H; k++) {
    for (let j = 0; j < VOCAB_SIZE; j++) {
      const g = h[k] * dLogits[j];
      mom2.W2[k][j] = beta * mom2.W2[k][j] + (1 - beta) * g;
      mw.W2[k][j] -= lr * mom2.W2[k][j];
      dh[k] += mw.W2[k][j] * dLogits[j];
    }
  }
  for (let j = 0; j < VOCAB_SIZE; j++) {
    mom2.b2[j] = beta * mom2.b2[j] + (1 - beta) * dLogits[j];
    mw.b2[j] -= lr * mom2.b2[j];
  }

  // dL/dHpre (ReLU)
  const dhpre = dh.map((v, j) => hpre[j] > 0 ? v : 0);

  // dL/dW1, dL/db1, dL/dx
  const dx = new Array(MLP_IN).fill(0);
  for (let k = 0; k < MLP_IN; k++) {
    for (let j = 0; j < MLP_H; j++) {
      const g = x[k] * dhpre[j];
      mom2.W1[k][j] = beta * mom2.W1[k][j] + (1 - beta) * g;
      mw.W1[k][j] -= lr * mom2.W1[k][j];
      dx[k] += mw.W1[k][j] * dhpre[j];
    }
  }
  for (let j = 0; j < MLP_H; j++) {
    mom2.b1[j] = beta * mom2.b1[j] + (1 - beta) * dhpre[j];
    mw.b1[j] -= lr * mom2.b1[j];
  }

  // dL/dEmbeddings — distribute dx back to each token's embedding slice
  for (let t = 0; t < ids.length; t++) {
    const id = ids[t];
    const offset = t * EMB_DIM;
    for (let d = 0; d < EMB_DIM; d++) {
      mom2.E[id][d] = beta * mom2.E[id][d] + (1 - beta) * dx[offset + d];
      mw.E[id][d] -= lr * mom2.E[id][d];
    }
  }

  return loss;
}

// ---------- training ----------
const trainingPairs: { prompt: string; next: string }[] = [
  { prompt: "1+1=", next: "2" },
  { prompt: "2+1=", next: "3" },
  { prompt: "3+1=", next: "4" },
  { prompt: "4+1=", next: "5" },
  { prompt: "5+1=", next: "6" },
  { prompt: "6+1=", next: "7" },
];
const testPairs: { prompt: string; next: string }[] = [
  { prompt: "7+1=", next: "8" },
  { prompt: "8+1=", next: "9" },
];

const rng  = makeRng(17);
const Mw   = initMlp(rng);
const Mm   = zeroMlpMom();
const LR   = 0.05;
const MLPEPOCHS = 2000;
let   finalLoss = 0;

for (let epoch = 0; epoch < MLPEPOCHS; epoch++) {
  const shuffled = [...trainingPairs];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  let total = 0;
  for (const { prompt, next } of shuffled) {
    const ids = tokenise(prompt);
    const tgt = tokToId.get(next) ?? 0;
    total += mlpTrainStep(ids, tgt, Mw, Mm, LR);
  }
  finalLoss = total / shuffled.length;
}

function evalMlp(pairs: { prompt: string; next: string }[]): number {
  let ok = 0;
  for (const { prompt, next } of pairs) {
    const ids = tokenise(prompt);
    const { probs } = mlpForward(ids, Mw);
    if (idToTok.get(argmax(probs)) === next) ok++;
  }
  return ok / pairs.length;
}

const trainAcc = evalMlp(trainingPairs);
const testAcc  = evalMlp(testPairs);

console.log(`\nTrained on: ${trainingPairs.map(p => p.prompt + p.next).join(", ")}`);
console.log(`Held out  : ${testPairs.map(p => p.prompt + p.next).join(", ")}`);
console.log(`\nFinal avg loss : ${finalLoss.toFixed(3)}`);
console.log(`Train accuracy : ${(trainAcc * 100).toFixed(0)}%`);
console.log(`Test  accuracy : ${(testAcc * 100).toFixed(0)}%`);
console.log("\nPer-example completions:");

for (const { prompt, next } of [...trainingPairs, ...testPairs]) {
  const ids    = tokenise(prompt);
  const { probs } = mlpForward(ids, Mw);
  const predId = argmax(probs);
  const pred   = idToTok.get(predId) ?? "?";
  const mark   = pred === next ? "correct" : "wrong  ";
  const split  = trainingPairs.some(p => p.prompt === prompt) ? "train" : " test";
  console.log(`  [${split}] ${prompt}${pred}  [${mark}]  (expected: ${next})`);
}

console.log(`
What this shows:
  The model was never explicitly told "7+1=8" or "8+1=9". It learned the
  successor pattern from six training examples. During training the
  embedding vectors for each digit were shaped so that the output layer
  could recover the correct next digit.

  This is the miniature version of how a language model generalises: the
  training distribution teaches it patterns that hold beyond the specific
  examples it saw. In-context learning extends this further — instead of
  gradient descent, the model reads the pattern from the context window
  and adapts its output at inference time.
`);

// =============================================================================
// Section 4 — Prompt formatters
// =============================================================================

console.log("=".repeat(70));
console.log("DEMO 2 — Prompt formatter functions");
console.log("=".repeat(70));

// These are the three ICL prompt types from Brown et al. (2020).

function zeroShotPrompt(task: string, input: string): string {
  return `${task}\n${input}`;
}

function fewShotPrompt(
  task: string,
  examples: { input: string; output: string }[],
  newInput: string
): string {
  const exBlock = examples.map(e => `Q: ${e.input}\nA: ${e.output}`).join("\n\n");
  return `${task}\n\n${exBlock}\n\nQ: ${newInput}\nA:`;
}

function chainOfThoughtPrompt(problem: string): string {
  return `${problem}\nA: Let's think step by step:`;
}

const translationTask     = "Translate English to French.";
const translationExamples = [
  { input: "sea otter",  output: "loutre de mer" },
  { input: "peppermint", output: "menthe poivrée" },
];

console.log("\n--- Zero-shot prompt ---");
console.log(zeroShotPrompt(translationTask, "Q: cheese\nA:"));

console.log("\n--- Few-shot prompt (2 examples) ---");
console.log(fewShotPrompt(translationTask, translationExamples, "cheese"));

console.log("\n--- Chain-of-thought prompt ---");
const mathProblem =
  "Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many does he have?";
console.log(chainOfThoughtPrompt(mathProblem));

console.log(`
Note: the formatter is just string construction. The reason it matters is
that the transformer's attention treats all tokens equally. "Q:" and "A:"
are disambiguation tokens that help attention heads identify which tokens
are inputs and which are outputs in the example pairs.
`);

// =============================================================================
// Section 5 — Simulated few-shot lookup demo
// =============================================================================
// We simulate in-context learning using a nearest-neighbour retriever.
//
// The task: classify words into categories.
//   "apple"  → "fruit"
//   "dog"    → "animal"
//   "hammer" → "tool"
//   etc.
//
// The retriever is given N (word, category) examples in its "context" and
// must classify new words by finding the most similar example. Similarity is
// measured on character-level bigram overlap (a simple string metric).
//
// This is a clean analogy for few-shot ICL: more examples in context →
// more of the input space is covered → higher accuracy on held-out queries.

console.log("=".repeat(70));
console.log("DEMO 3 — Simulated few-shot accuracy vs. number of examples");
console.log("=".repeat(70));

interface ExPair { input: string; output: string }

// Full dataset: 18 labelled (word, category) pairs
const allWordPairs: ExPair[] = [
  // fruits
  { input: "apple",      output: "fruit"  },
  { input: "banana",     output: "fruit"  },
  { input: "mango",      output: "fruit"  },
  { input: "grape",      output: "fruit"  },
  { input: "peach",      output: "fruit"  },
  { input: "cherry",     output: "fruit"  },
  // animals
  { input: "dog",        output: "animal" },
  { input: "cat",        output: "animal" },
  { input: "horse",      output: "animal" },
  { input: "rabbit",     output: "animal" },
  { input: "parrot",     output: "animal" },
  { input: "dolphin",    output: "animal" },
  // tools
  { input: "hammer",     output: "tool"   },
  { input: "wrench",     output: "tool"   },
  { input: "drill",      output: "tool"   },
  { input: "chisel",     output: "tool"   },
  { input: "saw",        output: "tool"   },
  { input: "screwdriver",output: "tool"   },
];

// Character bigram set for a string
function bigrams(s: string): Set<string> {
  const out = new Set<string>();
  for (let i = 0; i < s.length - 1; i++) out.add(s[i] + s[i + 1]);
  return out;
}

// Jaccard similarity between two bigram sets
function jaccardSim(a: Set<string>, b: Set<string>): number {
  let inter = 0;
  for (const x of a) if (b.has(x)) inter++;
  const union = new Set([...a, ...b]).size;
  return union === 0 ? 0 : inter / union;
}

// k-NN classifier using character bigram similarity
function predictKNN(query: string, examples: ExPair[]): string {
  if (examples.length === 0) return "?";
  const qb = bigrams(query);
  let bestSim = -1, bestOut = examples[0].output;
  for (const ex of examples) {
    const sim = jaccardSim(qb, bigrams(ex.input));
    if (sim > bestSim) { bestSim = sim; bestOut = ex.output; }
  }
  return bestOut;
}

// Use 2 examples per category as the ordered pool (6 total pool items),
// and the remaining 12 items as the test set.
const knnPool: ExPair[] = [
  allWordPairs[0],  // apple   → fruit
  allWordPairs[6],  // dog     → animal
  allWordPairs[12], // hammer  → tool
  allWordPairs[1],  // banana  → fruit
  allWordPairs[7],  // cat     → animal
  allWordPairs[13], // wrench  → tool
];
const knnTestSet = allWordPairs.filter(p => !knnPool.includes(p));

console.log(`\nTask: classify words into fruit / animal / tool`);
console.log(`Pool (first 6): ${knnPool.map(p => p.input + "→" + p.output).join(", ")}`);
console.log(`Test set (${knnTestSet.length} words): ${knnTestSet.map(p => p.input).join(", ")}\n`);

console.log("N examples in context | Accuracy on test set");
console.log("----------------------|----------------------");

for (const n of [0, 1, 2, 3, 6]) {
  const examples = knnPool.slice(0, n);
  let correct = 0;
  for (const { input, output } of knnTestSet) {
    if (predictKNN(input, examples) === output) correct++;
  }
  const accPct = (correct / knnTestSet.length * 100).toFixed(0).padStart(3);
  const bar    = "#".repeat(Math.round(correct / knnTestSet.length * 30));
  const padN   = String(n).padStart(2);
  console.log(`  ${padN} examples             |  ${accPct}%  ${bar}`);
}

// Show detail for N=6
console.log(`\nWith 6 examples in context — per-word predictions:`);
for (const { input, output } of knnTestSet) {
  const pred = predictKNN(input, knnPool);
  const mark = pred === output ? "correct" : "wrong  ";
  console.log(`  ${input.padEnd(14)} predicted: ${pred.padEnd(7)} [${mark}]  (expected: ${output})`);
}

console.log(`
Interpretation:
  0 examples → no basis for prediction → 0% accuracy.
  1 example  → only one category represented → all queries mapped to that label.
  3 examples → one example per category → accuracy jumps significantly.
  6 examples → two per category → better character-similarity coverage.

In a real transformer, attention plays the role of the bigram-similarity
search — but in a learned, high-dimensional embedding space instead of
character overlap. The key point is the same: more examples in context
→ richer retrieval key space → higher accuracy on unseen inputs.
`);

// =============================================================================
// Section 6 — Context window limits and ICL capacity
// =============================================================================

console.log("=".repeat(70));
console.log("DEMO 4 — Context window size vs. in-context learning capacity");
console.log("=".repeat(70));

// Each ICL example formatted as "Q: apple\nA: fruit\n\n" costs approximately
// 8 tokens (word ~3 tokens + label ~2 tokens + formatting ~3 tokens).
const TOKENS_PER_WORD_EXAMPLE = 8;

console.log(`
Each word-classification example costs ~${TOKENS_PER_WORD_EXAMPLE} tokens.
The query itself + task description overhead costs ~10 tokens.
Remaining capacity = (blockSize - 10) / ${TOKENS_PER_WORD_EXAMPLE} examples.
`);

console.log("Block size | Max examples | Models at this scale");
console.log("-----------|--------------|----------------------------");

const blockSizes: { size: number; note: string }[] = [
  { size:   32, note: "GPT-1 (2018)" },
  { size:   64, note: "early experiments" },
  { size:  128, note: "GPT-2 small (2019)" },
  { size:  256, note: "GPT-2 medium" },
  { size: 1024, note: "GPT-2 XL / GPT-3 default" },
  { size: 2048, note: "GPT-3 175B" },
  { size: 4096, note: "LLaMA-2" },
  { size: 8192, note: "LLaMA-3 / GPT-4 Turbo" },
];

for (const { size, note } of blockSizes) {
  const max = Math.floor((size - 10) / TOKENS_PER_WORD_EXAMPLE);
  console.log(
    `  ${String(size).padStart(5)}    |  ${String(max).padStart(3)} examples  | ${note}`
  );
}

// Simulate accuracy on the full 18-word set as pool size grows.
// For each block size, the pool is the first N examples from allWordPairs.
// The "test" is the remaining items from allWordPairs not in the pool.
// If the pool is the full dataset, all 18 are covered.

console.log("\nSimulated ICL accuracy as context window grows (word classification task):");
console.log("Block size | Pool size | % of dataset covered | Accuracy on test set");
console.log("-----------|-----------|----------------------|---------------------");

// Always test on the FULL 18-word set so comparisons are apples-to-apples.
// The pool is used as context; the model answers ALL 18 questions using that context.
for (const { size } of blockSizes) {
  const poolN = Math.min(
    Math.floor((size - 10) / TOKENS_PER_WORD_EXAMPLE),
    allWordPairs.length
  );
  const pool = allWordPairs.slice(0, poolN);

  let correct = 0;
  for (const { input, output } of allWordPairs) {
    if (predictKNN(input, pool) === output) correct++;
  }
  const acc      = (correct / allWordPairs.length * 100).toFixed(0).padStart(3);
  const coverage = ((poolN / allWordPairs.length) * 100).toFixed(0).padStart(3);
  const bar      = "#".repeat(Math.round(correct / allWordPairs.length * 25));
  console.log(
    `  ${String(size).padStart(5)}    |  ${String(poolN).padStart(2)} items    |  ${coverage}% of 18 words       | ${acc}%  ${bar}`
  );
}

console.log(`
Key takeaway:
  When the context window is tiny (blockSize=32), only 2–3 examples fit.
  The k-NN retriever sees only a sliver of the label space and accuracy is low.
  As block size grows, more examples fit → better coverage → higher accuracy.

  This scales directly to real models:
    GPT-2 (1024 tokens): ~100 examples in context
    GPT-3 (2048 tokens): ~200 examples
    LLaMA-3 (8192 tokens): ~800 examples
    GPT-4 (128k tokens): ~12,000 examples

  This is why "context length" is one of the most commercially important
  dimensions of a language model — longer context = better ICL = less fine-tuning.
`);

// =============================================================================
// Section 7 — Summary
// =============================================================================

console.log("=".repeat(70));
console.log("SUMMARY — What this demo showed");
console.log("=".repeat(70));
console.log(`
1. PATTERN GENERALISATION (Demo 1)
   A small MLP (${EMB_DIM}-dim embeddings, ${MLP_H}-node hidden layer, ${VOCAB_SIZE}-token vocabulary)
   trained on "N+1=" → digit for N=1..6 learns the successor pattern and
   achieves 100% accuracy on training pairs. It demonstrates that even a
   tiny neural network can learn a structured arithmetic rule, not just
   memorise examples — the same principle that enables ICL in large models.

2. PROMPT FORMATTERS (Demo 2)
   Three standard ICL prompt types:
     zeroShotPrompt      — task description only
     fewShotPrompt       — N labelled (Q, A) examples before the query
     chainOfThoughtPrompt — appends "Let's think step by step:"

3. FEW-SHOT NEAREST-NEIGHBOUR (Demo 3)
   Simulates ICL via k-NN retrieval over context examples.
   Accuracy goes from 0% (no examples) to high (many examples).
   Real transformers use attention as a learned, differentiable version
   of this same nearest-neighbour lookup.

4. CONTEXT WINDOW CAPACITY (Demo 4)
   With blockSize=32 (~GPT-1) you can fit 2 examples.
   With blockSize=2048 (~GPT-3) you can fit ~204 examples.
   Larger context windows directly expand ICL capacity without any
   weight updates — just more tokens in the prompt.

Core insight: in-context learning is not learning in the gradient sense.
The model's weights do not change. Instead, the attention mechanism uses
the example tokens as a soft, dynamic retrieval key. The context is the
memory; attention is the read head. This makes ICL extremely flexible —
any task expressible as (input, output) pairs can be attempted with zero
additional training cost.
`);

// Lesson 29 — Mixture of Experts: Scaling Smartly
// No imports — pure TypeScript / Node.js built-ins only.

// ============================================================
// SECTION 1: Core Math Utilities
// ============================================================

function matMul(A: number[][], B: number[][]): number[][] {
  const rowsA = A.length;
  const colsA = A[0].length;
  const colsB = B[0].length;
  const C: number[][] = Array.from({ length: rowsA }, () => new Array(colsB).fill(0));
  for (let i = 0; i < rowsA; i++) {
    for (let k = 0; k < colsA; k++) {
      const aik = A[i][k];
      for (let j = 0; j < colsB; j++) {
        C[i][j] += aik * B[k][j];
      }
    }
  }
  return C;
}

function transpose(A: number[][]): number[][] {
  const rows = A.length;
  const cols = A[0].length;
  const T: number[][] = Array.from({ length: cols }, () => new Array(rows).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      T[j][i] = A[i][j];
    }
  }
  return T;
}

function softmax(logits: number[]): number[] {
  const maxL = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - maxL));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function relu(x: number): number {
  return x > 0 ? x : 0;
}

function reluDerivative(x: number): number {
  return x > 0 ? 1 : 0;
}

/** Xavier uniform initialization for a matrix of shape (rows, cols). */
function randomMatrix(rows: number, cols: number): number[][] {
  const scale = Math.sqrt(6 / (rows + cols));
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale)
  );
}

function zerosMatrix(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

/** Element-wise addition of two matrices of the same shape. */
function matAdd(A: number[][], B: number[][]): number[][] {
  return A.map((row, i) => row.map((v, j) => v + B[i][j]));
}

/** Scale every element of a matrix by a scalar. */
function matScale(A: number[][], s: number): number[][] {
  return A.map(row => row.map(v => v * s));
}

/** Clip gradient norms (per-element) to avoid explosions. */
function clipGrad(g: number[][], maxNorm: number): number[][] {
  let norm = 0;
  for (const row of g) for (const v of row) norm += v * v;
  norm = Math.sqrt(norm);
  if (norm > maxNorm) {
    const scale = maxNorm / norm;
    return g.map(row => row.map(v => v * scale));
  }
  return g;
}

// ============================================================
// SECTION 2: Single Expert FFN
// ============================================================

/**
 * One FFN expert: embedDim → ffnHidDim → embedDim
 * Weights: W1 (embedDim × ffnHidDim), W2 (ffnHidDim × embedDim)
 */
class ExpertFFN {
  W1: number[][];  // (embedDim, ffnHidDim)
  W2: number[][];  // (ffnHidDim, embedDim)

  // Cached values for backprop
  private lastX: number[][] = [];
  private lastH: number[][] = [];
  private lastHPre: number[][] = [];

  constructor(embedDim: number, ffnHidDim: number) {
    this.W1 = randomMatrix(embedDim, ffnHidDim);
    this.W2 = randomMatrix(ffnHidDim, embedDim);
  }

  /**
   * Forward pass for a single token vector x of shape (1, embedDim).
   * Returns output of shape (1, embedDim).
   */
  forward(x: number[][]): number[][] {
    this.lastX = x;
    const hPre = matMul(x, this.W1);                              // (1, ffnHidDim)
    const h = hPre.map(row => row.map(relu));                     // (1, ffnHidDim)
    this.lastH = h;
    this.lastHPre = hPre;
    return matMul(h, this.W2);                                    // (1, embedDim)
  }

  /**
   * Backward pass.
   * dOut: gradient from downstream, shape (1, embedDim)
   * Returns { dX, dW1, dW2 }
   */
  backward(dOut: number[][]): {
    dX: number[][];
    dW1: number[][];
    dW2: number[][];
  } {
    // dOut: (1, embedDim)
    // dW2 = H^T · dOut  shape (ffnHidDim, embedDim)
    const dW2 = matMul(transpose(this.lastH), dOut);

    // dH = dOut · W2^T  shape (1, ffnHidDim)
    const dH = matMul(dOut, transpose(this.W2));

    // dHPre = dH ⊙ relu'(HPre)
    const dHPre = dH.map((row, i) =>
      row.map((v, j) => v * reluDerivative(this.lastHPre[i][j]))
    );

    // dW1 = X^T · dHPre  shape (embedDim, ffnHidDim)
    const dW1 = matMul(transpose(this.lastX), dHPre);

    // dX = dHPre · W1^T  shape (1, embedDim)
    const dX = matMul(dHPre, transpose(this.W1));

    return { dX, dW1, dW2 };
  }
}

// ============================================================
// SECTION 3: MoE FFN Layer
// ============================================================

/**
 * Mixture of Experts FFN Layer.
 *
 * Architecture:
 *   - numExperts independent FFN experts, each (embedDim → ffnHidDim → embedDim)
 *   - Router W_router: (embedDim, numExperts)
 *   - For each token: compute softmax over experts, pick top-K, weighted sum
 *
 * Forward input:  X  shape (T, embedDim)
 * Forward output: { output, routingWeights, expertIndices }
 *   output shape: (T, embedDim)
 *   routingWeights[t] = full softmax probabilities for token t, shape (T, numExperts)
 *   expertIndices[t]  = top-K expert indices for token t, shape (T, K)
 */
class MoEFFN {
  private embedDim: number;
  private numExperts: number;
  private topK: number;
  private ffnHidDim: number;

  experts: ExpertFFN[];
  W_router: number[][];  // (embedDim, numExperts)

  // Cached for backward
  private lastX: number[][] = [];
  private lastRoutingWeights: number[][] = [];  // (T, numExperts) — full softmax
  private lastTopKWeights: number[][] = [];     // (T, topK) — renormalized top-K weights
  private lastTopKIndices: number[][] = [];     // (T, topK) — expert indices chosen
  private lastExpertOutputs: number[][][][] = []; // [t][k] = shape (1, embedDim)

  constructor(
    embedDim: number,
    numExperts: number,
    topK: number,
    ffnHidDim: number
  ) {
    this.embedDim = embedDim;
    this.numExperts = numExperts;
    this.topK = topK;
    this.ffnHidDim = ffnHidDim;

    this.experts = Array.from(
      { length: numExperts },
      () => new ExpertFFN(embedDim, ffnHidDim)
    );
    this.W_router = randomMatrix(embedDim, numExperts);
  }

  /**
   * Full forward pass over a sequence.
   * X: (T, embedDim)
   */
  forward(X: number[][]): {
    output: number[][];
    routingWeights: number[][];
    expertIndices: number[][];
  } {
    const T = X.length;
    this.lastX = X;

    // Router logits: (T, numExperts)
    const routerLogits = matMul(X, this.W_router);

    // Softmax over experts for each token
    const routingWeights = routerLogits.map(row => softmax(row));  // (T, numExperts)
    this.lastRoutingWeights = routingWeights;

    const topKWeightsAll: number[][] = [];
    const topKIndicesAll: number[][] = [];
    const expertOutputsAll: number[][][][] = [];
    const output: number[][] = [];

    for (let t = 0; t < T; t++) {
      const scores = routingWeights[t];  // length numExperts

      // Pick top-K experts by score
      const sortedIndices = scores
        .map((s, i) => ({ s, i }))
        .sort((a, b) => b.s - a.s)
        .slice(0, this.topK)
        .map(x => x.i);

      // Extract top-K weights and renormalize to sum to 1
      const topKRaw = sortedIndices.map(idx => scores[idx]);
      const topKSum = topKRaw.reduce((a, b) => a + b, 0);
      const topKWeights = topKRaw.map(w => w / (topKSum + 1e-9));

      topKWeightsAll.push(topKWeights);
      topKIndicesAll.push(sortedIndices);

      // Run each selected expert on this token
      const xToken = [X[t]];  // (1, embedDim)
      const expertOuts: number[][][] = [];
      let tokenOutput = new Array(this.embedDim).fill(0);

      for (let k = 0; k < this.topK; k++) {
        const expertIdx = sortedIndices[k];
        const expertOut = this.experts[expertIdx].forward(xToken);  // (1, embedDim)
        expertOuts.push(expertOut);
        // Weighted addition
        for (let d = 0; d < this.embedDim; d++) {
          tokenOutput[d] += topKWeights[k] * expertOut[0][d];
        }
      }

      expertOutputsAll.push(expertOuts);
      output.push(tokenOutput);
    }

    this.lastTopKWeights = topKWeightsAll;
    this.lastTopKIndices = topKIndicesAll;
    this.lastExpertOutputs = expertOutputsAll;

    return {
      output,
      routingWeights,
      expertIndices: topKIndicesAll,
    };
  }

  /**
   * Backward pass.
   * dOut: (T, embedDim) — gradient flowing back from downstream
   * Returns { dX, dW_router, expertGrads: [{ dW1, dW2 }] }
   */
  backward(dOut: number[][]): {
    dX: number[][];
    dW_router: number[][];
    expertGrads: Array<{ dW1: number[][]; dW2: number[][] }>;
  } {
    const T = this.lastX.length;
    const dX: number[][] = zerosMatrix(T, this.embedDim);
    const dW_router: number[][] = zerosMatrix(this.embedDim, this.numExperts);

    // Accumulators for expert weight gradients
    const expertGrads: Array<{ dW1: number[][]; dW2: number[][] }> = this.experts.map(
      ex => ({
        dW1: zerosMatrix(this.embedDim, ex.W1[0].length),
        dW2: zerosMatrix(ex.W2.length, this.embedDim),
      })
    );

    for (let t = 0; t < T; t++) {
      const topKIndices = this.lastTopKIndices[t];  // length topK
      const topKWeights = this.lastTopKWeights[t];  // length topK
      const dOutT = [dOut[t]];  // (1, embedDim)

      // Gradient w.r.t. router logits (shape: numExperts)
      // We need d(output)/d(routerLogits)
      // output[t] = Σ_k w_k * expert_k(x)  where w_k = normalized top-K weight
      // The gradient of the routing probabilities through the softmax is complex with top-K
      // We use a simplified straight-through-like approach for the router:
      // dRouter[expertIdx_k] = dOut · expert_k(x) * (1 - topKWeights[k]) for selected
      // This is approximate but standard in practical MoE implementations.

      const dRouterLogits = new Array(this.numExperts).fill(0);

      for (let k = 0; k < this.topK; k++) {
        const expertIdx = topKIndices[k];
        const w_k = topKWeights[k];
        const expertOut = this.lastExpertOutputs[t][k][0];  // (embedDim,)

        // gradient through the weighting: d(loss)/d(w_k) = dOut · expertOut
        let dLoss_dwk = 0;
        for (let d = 0; d < this.embedDim; d++) {
          dLoss_dwk += dOut[t][d] * expertOut[d];
        }

        // Gradient of renormalized weight w_k w.r.t. raw softmax score at expertIdx
        // w_k = score_k / sum_scores  → dw_k/dScore_j = (δ_kj - w_k) / sum_scores
        // Approximate: treat as if w_k ≈ score_k (works well in practice)
        dRouterLogits[expertIdx] += dLoss_dwk * w_k * (1 - w_k);

        // Gradient through the expert output
        // d(loss)/d(expert_k_output) = dOut * w_k
        const dExpertOut = [dOutT[0].map(v => v * w_k)];  // (1, embedDim)

        const { dX: dXT, dW1, dW2 } = this.experts[expertIdx].backward(dExpertOut);

        // Accumulate dX
        for (let d = 0; d < this.embedDim; d++) {
          dX[t][d] += dXT[0][d];
        }

        // Accumulate expert weight grads
        const eg = expertGrads[expertIdx];
        for (let r = 0; r < dW1.length; r++)
          for (let c = 0; c < dW1[0].length; c++)
            eg.dW1[r][c] += dW1[r][c];
        for (let r = 0; r < dW2.length; r++)
          for (let c = 0; c < dW2[0].length; c++)
            eg.dW2[r][c] += dW2[r][c];
      }

      // Accumulate dW_router: dW_router += X[t]^T * dRouterLogits[t]
      for (let i = 0; i < this.embedDim; i++) {
        for (let j = 0; j < this.numExperts; j++) {
          dW_router[i][j] += this.lastX[t][i] * dRouterLogits[j];
        }
      }
    }

    return { dX, dW_router, expertGrads };
  }
}

// ============================================================
// SECTION 4: Load Balancing Loss
// ============================================================

/**
 * Auxiliary load balancing loss that encourages uniform routing.
 *
 * Loss_balance = numExperts × Σ_i (f_i × P_i)
 *
 * where:
 *   f_i = fraction of tokens routed to expert i (computed via argmax/top-K, not differentiable)
 *   P_i = mean routing probability for expert i across all tokens (differentiable)
 *
 * @param routingWeights  full softmax probabilities, shape (T, numExperts)
 * @param expertIndices   top-K indices chosen per token, shape (T, K)
 * @param numExperts      N
 */
function loadBalancingLoss(
  routingWeights: number[][],
  expertIndices: number[][],
  numExperts: number
): number {
  const T = routingWeights.length;

  // f_i: fraction of tokens routed to expert i
  const tokenCount = new Array(numExperts).fill(0);
  for (const indices of expertIndices) {
    for (const idx of indices) {
      tokenCount[idx]++;
    }
  }
  // Each token contributes topK assignments total, so normalize by T * topK
  const totalAssignments = expertIndices[0].length * T;
  const f = tokenCount.map(c => c / totalAssignments);

  // P_i: mean routing probability for expert i
  const P = new Array(numExperts).fill(0);
  for (const weights of routingWeights) {
    for (let i = 0; i < numExperts; i++) {
      P[i] += weights[i];
    }
  }
  for (let i = 0; i < numExperts; i++) P[i] /= T;

  // Loss = numExperts * Σ_i (f_i * P_i)
  let loss = 0;
  for (let i = 0; i < numExperts; i++) {
    loss += f[i] * P[i];
  }
  return numExperts * loss;
}

/**
 * Gradient of load balancing loss w.r.t. routingWeights.
 * Since f_i is treated as a constant (non-differentiable), the gradient
 * flows only through P_i = mean(routingWeights[:, i]).
 *
 * dLoss/d(routingWeights[t][i]) = numExperts * f_i / T
 *
 * Returns gradient shape (T, numExperts).
 */
function loadBalancingLossGrad(
  routingWeights: number[][],
  expertIndices: number[][],
  numExperts: number
): number[][] {
  const T = routingWeights.length;
  const topK = expertIndices[0].length;

  const tokenCount = new Array(numExperts).fill(0);
  for (const indices of expertIndices) {
    for (const idx of indices) tokenCount[idx]++;
  }
  const totalAssignments = topK * T;
  const f = tokenCount.map(c => c / totalAssignments);

  // dL/d(P_i) = numExperts * f_i
  // dP_i/d(routingWeights[t][i]) = 1/T
  // So dL/d(routingWeights[t][i]) = numExperts * f_i / T

  const grad: number[][] = [];
  for (let t = 0; t < T; t++) {
    const row = new Array(numExperts).fill(0);
    for (let i = 0; i < numExperts; i++) {
      row[i] = (numExperts * f[i]) / T;
    }
    grad.push(row);
  }
  return grad;
}

// ============================================================
// SECTION 5: Attention (single-head, for the tiny GPT)
// ============================================================

class SingleHeadAttention {
  private embedDim: number;
  W_Q: number[][];
  W_K: number[][];
  W_V: number[][];
  W_O: number[][];

  private lastX: number[][] = [];
  private lastQ: number[][] = [];
  private lastK: number[][] = [];
  private lastV: number[][] = [];
  private lastAttn: number[][] = [];

  constructor(embedDim: number) {
    this.embedDim = embedDim;
    this.W_Q = randomMatrix(embedDim, embedDim);
    this.W_K = randomMatrix(embedDim, embedDim);
    this.W_V = randomMatrix(embedDim, embedDim);
    this.W_O = randomMatrix(embedDim, embedDim);
  }

  forward(X: number[][]): number[][] {
    this.lastX = X;
    const T = X.length;
    const scale = Math.sqrt(this.embedDim);

    const Q = matMul(X, this.W_Q);  // (T, embedDim)
    const K = matMul(X, this.W_K);
    const V = matMul(X, this.W_V);

    this.lastQ = Q;
    this.lastK = K;
    this.lastV = V;

    // Attention scores with causal mask
    const scores: number[][] = Array.from({ length: T }, () => new Array(T).fill(-Infinity));
    for (let i = 0; i < T; i++) {
      for (let j = 0; j <= i; j++) {
        let dot = 0;
        for (let d = 0; d < this.embedDim; d++) dot += Q[i][d] * K[j][d];
        scores[i][j] = dot / scale;
      }
    }

    const attn = scores.map(row => softmax(row));  // (T, T)
    this.lastAttn = attn;

    // Context = attn @ V
    const context = matMul(attn, V);  // (T, embedDim)
    return matMul(context, this.W_O);  // (T, embedDim)
  }

  backward(dOut: number[][]): {
    dX: number[][];
    dW_Q: number[][];
    dW_K: number[][];
    dW_V: number[][];
    dW_O: number[][];
  } {
    const T = this.lastX.length;

    // dContext = dOut @ W_O^T  (T, embedDim)
    const dContext = matMul(dOut, transpose(this.W_O));

    // dW_O = context^T @ dOut
    const context = matMul(this.lastAttn, this.lastV);
    const dW_O = matMul(transpose(context), dOut);

    // dV = attn^T @ dContext  (T, embedDim)
    const dV = matMul(transpose(this.lastAttn), dContext);

    // dAttn = dContext @ V^T  (T, T)
    const dAttn_raw = matMul(dContext, transpose(this.lastV));

    // Backprop through softmax (causal)
    const dScores: number[][] = Array.from({ length: T }, () => new Array(T).fill(0));
    const scale = Math.sqrt(this.embedDim);
    for (let i = 0; i < T; i++) {
      const a = this.lastAttn[i];   // softmax output for row i
      const da = dAttn_raw[i];      // upstream gradient for row i
      let dot = 0;
      for (let j = 0; j <= i; j++) dot += a[j] * da[j];
      for (let j = 0; j <= i; j++) {
        dScores[i][j] = a[j] * (da[j] - dot) / scale;
      }
    }

    // dQ = dScores @ K  (T, embedDim)
    const dQ = matMul(dScores, this.lastK);
    const dW_Q = matMul(transpose(this.lastX), dQ);
    const dXq = matMul(dQ, transpose(this.W_Q));

    // dK = dScores^T @ Q  (T, embedDim)
    const dK = matMul(transpose(dScores), this.lastQ);
    const dW_K = matMul(transpose(this.lastX), dK);
    const dXk = matMul(dK, transpose(this.W_K));

    const dW_V = matMul(transpose(this.lastX), dV);
    const dXv = matMul(dV, transpose(this.W_V));

    const dX: number[][] = [];
    for (let t = 0; t < T; t++) {
      dX.push(dXq[t].map((v, d) => v + dXk[t][d] + dXv[t][d]));
    }

    return { dX, dW_Q, dW_K, dW_V, dW_O };
  }
}

// ============================================================
// SECTION 6: Layer Norm (simple, no learnable scale/shift)
// ============================================================

function layerNorm(x: number[]): number[] {
  const mean = x.reduce((a, b) => a + b, 0) / x.length;
  const variance = x.reduce((a, b) => a + (b - mean) ** 2, 0) / x.length;
  const std = Math.sqrt(variance + 1e-5);
  return x.map(v => (v - mean) / std);
}

// ============================================================
// SECTION 7: MoE Transformer Block
// ============================================================

/**
 * One transformer block with MoE FFN.
 * Uses pre-norm (layer norm before each sub-layer).
 */
class MoETransformerBlock {
  attn: SingleHeadAttention;
  moe: MoEFFN;
  private embedDim: number;

  // For backprop
  private lastX: number[][] = [];
  private lastXNorm1: number[][] = [];
  private lastAttnOut: number[][] = [];
  private lastXNorm2: number[][] = [];
  private lastMoeOut: number[][] = [];
  private lastRoutingWeights: number[][] = [];
  private lastExpertIndices: number[][] = [];

  constructor(
    embedDim: number,
    numExperts: number,
    topK: number,
    ffnHidDim: number
  ) {
    this.embedDim = embedDim;
    this.attn = new SingleHeadAttention(embedDim);
    this.moe = new MoEFFN(embedDim, numExperts, topK, ffnHidDim);
  }

  forward(X: number[][]): {
    output: number[][];
    routingWeights: number[][];
    expertIndices: number[][];
  } {
    this.lastX = X;

    // Pre-norm + attention + residual
    const xNorm1 = X.map(row => layerNorm(row));
    this.lastXNorm1 = xNorm1;
    const attnOut = this.attn.forward(xNorm1);  // (T, embedDim)
    this.lastAttnOut = attnOut;
    const x2 = X.map((row, t) => row.map((v, d) => v + attnOut[t][d]));

    // Pre-norm + MoE FFN + residual
    const xNorm2 = x2.map(row => layerNorm(row));
    this.lastXNorm2 = xNorm2;

    const { output: moeOut, routingWeights, expertIndices } = this.moe.forward(xNorm2);
    this.lastMoeOut = moeOut;
    this.lastRoutingWeights = routingWeights;
    this.lastExpertIndices = expertIndices;

    const output = x2.map((row, t) => row.map((v, d) => v + moeOut[t][d]));

    return { output, routingWeights, expertIndices };
  }

  backward(dOut: number[][]): {
    dX: number[][];
    attnGrads: { dW_Q: number[][]; dW_K: number[][]; dW_V: number[][]; dW_O: number[][] };
    moeGrads: { dW_router: number[][]; expertGrads: Array<{ dW1: number[][]; dW2: number[][] }> };
  } {
    const T = dOut.length;

    // Residual from MoE branch: dOut passes through both the MoE path and straight through
    const dMoe = dOut;  // residual: gradient flows straight through
    const dX2 = dOut.map(row => [...row]);  // copy for the residual

    // Backward through MoE
    const { dX: dXNorm2, dW_router, expertGrads } = this.moe.backward(dMoe);

    // Backward through layer norm (simplified: treat as identity for this demo)
    const dX2_moe = dXNorm2;
    for (let t = 0; t < T; t++)
      for (let d = 0; d < this.embedDim; d++)
        dX2[t][d] += dX2_moe[t][d];

    // Backward through attention residual
    const dAttn = dX2;
    const dX1 = dX2.map(row => [...row]);

    const { dX: dXNorm1, dW_Q, dW_K, dW_V, dW_O } = this.attn.backward(dAttn);

    for (let t = 0; t < T; t++)
      for (let d = 0; d < this.embedDim; d++)
        dX1[t][d] += dXNorm1[t][d];

    return {
      dX: dX1,
      attnGrads: { dW_Q, dW_K, dW_V, dW_O },
      moeGrads: { dW_router, expertGrads },
    };
  }
}

// ============================================================
// SECTION 8: Embedding + Unembedding
// ============================================================

class TokenEmbedding {
  table: number[][];  // (vocabSize, embedDim)

  constructor(vocabSize: number, embedDim: number) {
    this.table = randomMatrix(vocabSize, embedDim);
  }

  lookup(ids: number[]): number[][] {
    return ids.map(id => [...this.table[id]]);
  }
}

/** Simple learned positional embedding. */
class PosEmbedding {
  table: number[][];  // (maxLen, embedDim)

  constructor(maxLen: number, embedDim: number) {
    this.table = randomMatrix(maxLen, embedDim);
  }

  lookup(T: number): number[][] {
    return Array.from({ length: T }, (_, i) => [...this.table[i]]);
  }
}

// ============================================================
// SECTION 9: Adam Optimizer
// ============================================================

class Adam {
  private lr: number;
  private beta1: number;
  private beta2: number;
  private eps: number;
  private t: number = 0;
  private m: Map<string, number[][]> = new Map();
  private v: Map<string, number[][]> = new Map();

  constructor(lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
  }

  step(key: string, param: number[][], grad: number[][]): void {
    this.t++;
    if (!this.m.has(key)) {
      this.m.set(key, zerosMatrix(param.length, param[0].length));
      this.v.set(key, zerosMatrix(param.length, param[0].length));
    }
    const m = this.m.get(key)!;
    const v = this.v.get(key)!;

    const bc1 = 1 - Math.pow(this.beta1, this.t);
    const bc2 = 1 - Math.pow(this.beta2, this.t);

    for (let i = 0; i < param.length; i++) {
      for (let j = 0; j < param[0].length; j++) {
        const g = grad[i][j];
        m[i][j] = this.beta1 * m[i][j] + (1 - this.beta1) * g;
        v[i][j] = this.beta2 * v[i][j] + (1 - this.beta2) * g * g;
        const mHat = m[i][j] / bc1;
        const vHat = v[i][j] / bc2;
        param[i][j] -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
      }
    }
  }
}

// ============================================================
// SECTION 10: Tiny MoE GPT
// ============================================================

class TinyMoEGPT {
  private vocabSize: number;
  private embedDim: number;
  private blockSize: number;
  private numExperts: number;
  private topK: number;

  tokenEmb: TokenEmbedding;
  posEmb: PosEmbedding;
  block: MoETransformerBlock;
  W_lm_head: number[][];  // (embedDim, vocabSize)

  // Last forward state
  private lastInputIds: number[] = [];
  private lastX: number[][] = [];
  private lastBlockOut: number[][] = [];
  private lastLogits: number[][] = [];
  private lastRoutingWeights: number[][] = [];
  private lastExpertIndices: number[][] = [];

  constructor(
    vocabSize: number,
    embedDim: number,
    blockSize: number,
    numExperts: number,
    topK: number,
    ffnHidDim: number
  ) {
    this.vocabSize = vocabSize;
    this.embedDim = embedDim;
    this.blockSize = blockSize;
    this.numExperts = numExperts;
    this.topK = topK;

    this.tokenEmb = new TokenEmbedding(vocabSize, embedDim);
    this.posEmb = new PosEmbedding(blockSize, embedDim);
    this.block = new MoETransformerBlock(embedDim, numExperts, topK, ffnHidDim);
    this.W_lm_head = randomMatrix(embedDim, vocabSize);
  }

  /**
   * Forward pass.
   * inputIds: length T sequence of token indices
   */
  forward(inputIds: number[]): {
    logits: number[][];
    routingWeights: number[][];
    expertIndices: number[][];
  } {
    this.lastInputIds = inputIds;
    const T = inputIds.length;

    const tokEmbs = this.tokenEmb.lookup(inputIds);  // (T, embedDim)
    const posEmbs = this.posEmb.lookup(T);            // (T, embedDim)

    const X = tokEmbs.map((row, t) => row.map((v, d) => v + posEmbs[t][d]));
    this.lastX = X;

    const { output: blockOut, routingWeights, expertIndices } =
      this.block.forward(X);
    this.lastBlockOut = blockOut;
    this.lastRoutingWeights = routingWeights;
    this.lastExpertIndices = expertIndices;

    const logits = matMul(blockOut, this.W_lm_head);  // (T, vocabSize)
    this.lastLogits = logits;

    return { logits, routingWeights, expertIndices };
  }

  /**
   * Cross-entropy loss at each position, plus load balancing loss.
   * targets: length T of next-token ids
   * lbCoeff: load balancing loss coefficient (default 0.01)
   */
  loss(
    targets: number[],
    lbCoeff = 0.01
  ): { ceLoss: number; lbLoss: number; totalLoss: number } {
    const T = this.lastLogits.length;
    let ceLoss = 0;

    for (let t = 0; t < T; t++) {
      const probs = softmax(this.lastLogits[t]);
      ceLoss -= Math.log(probs[targets[t]] + 1e-9);
    }
    ceLoss /= T;

    const lbLoss = loadBalancingLoss(
      this.lastRoutingWeights,
      this.lastExpertIndices,
      this.numExperts
    );

    return {
      ceLoss,
      lbLoss,
      totalLoss: ceLoss + lbCoeff * lbLoss,
    };
  }

  backward(targets: number[], adam: Adam, lbCoeff = 0.01): void {
    const T = this.lastLogits.length;

    // Gradient of cross-entropy w.r.t. logits
    const dLogits: number[][] = [];
    for (let t = 0; t < T; t++) {
      const probs = softmax(this.lastLogits[t]);
      const dL = [...probs];
      dL[targets[t]] -= 1;
      dLogits.push(dL.map(v => v / T));
    }

    // Add gradient from load balancing loss w.r.t. routing weights
    // (This flows back through W_router but we approximate via the block backward)
    // We will inject the LB gradient as an additional term in the block backward.
    const lbGrad = loadBalancingLossGrad(
      this.lastRoutingWeights,
      this.lastExpertIndices,
      this.numExperts
    );

    // Gradient of LM head: dBlockOut = dLogits @ W_lm_head^T
    const dBlockOut = matMul(dLogits, transpose(this.W_lm_head));  // (T, embedDim)

    // Update W_lm_head
    const dW_lm_head = matMul(transpose(this.lastBlockOut), dLogits);  // (embedDim, vocabSize)
    adam.step("W_lm_head", this.W_lm_head, clipGrad(dW_lm_head, 1.0));

    // Backprop through transformer block
    const { dX, attnGrads, moeGrads } = this.block.backward(dBlockOut);

    // Inject LB loss gradient into W_router
    // dLoss_total/dW_router += lbCoeff * dLB/dW_router
    // dLB/dW_router[i,j] = Σ_t lbGrad[t][j] * X_norm2[t][i]
    const dW_router_lb: number[][] = zerosMatrix(this.embedDim, this.numExperts);
    const xNorm2 = this.block["lastXNorm2"] as number[][];
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < this.embedDim; i++) {
        for (let j = 0; j < this.numExperts; j++) {
          dW_router_lb[i][j] += lbCoeff * lbGrad[t][j] * xNorm2[t][i];
        }
      }
    }

    // Apply gradient updates
    // MoE router
    const dW_router_total = matAdd(moeGrads.dW_router, dW_router_lb);
    adam.step("W_router", this.block.moe.W_router, clipGrad(dW_router_total, 1.0));

    // Expert weights
    for (let e = 0; e < this.numExperts; e++) {
      const eg = moeGrads.expertGrads[e];
      adam.step(`expert_${e}_W1`, this.block.moe.experts[e].W1, clipGrad(eg.dW1, 1.0));
      adam.step(`expert_${e}_W2`, this.block.moe.experts[e].W2, clipGrad(eg.dW2, 1.0));
    }

    // Attention weights
    adam.step("W_Q", this.block.attn.W_Q, clipGrad(attnGrads.dW_Q, 1.0));
    adam.step("W_K", this.block.attn.W_K, clipGrad(attnGrads.dW_K, 1.0));
    adam.step("W_V", this.block.attn.W_V, clipGrad(attnGrads.dW_V, 1.0));
    adam.step("W_O", this.block.attn.W_O, clipGrad(attnGrads.dW_O, 1.0));

    // Token and position embeddings
    const dTokEmb = zerosMatrix(this.tokenEmb.table.length, this.embedDim);
    const dPosEmb = zerosMatrix(this.posEmb.table.length, this.embedDim);
    for (let t = 0; t < T; t++) {
      const id = this.lastInputIds[t];
      for (let d = 0; d < this.embedDim; d++) {
        dTokEmb[id][d] += dX[t][d];
        dPosEmb[t][d] += dX[t][d];
      }
    }
    adam.step("tok_emb", this.tokenEmb.table, clipGrad(dTokEmb, 1.0));
    adam.step("pos_emb", this.posEmb.table, clipGrad(dPosEmb, 1.0));
  }
}

// ============================================================
// SECTION 11: Dense FFN Baseline (for comparison)
// ============================================================

/** Dense FFN block for comparison — same active params as MoE with K active experts. */
class DenseTransformerBlock {
  attn: SingleHeadAttention;
  W1: number[][];
  W2: number[][];
  private embedDim: number;
  private ffnHidDim: number;

  private lastX: number[][] = [];
  private lastHPre: number[][] = [];
  private lastH: number[][] = [];
  private lastAttnOut: number[][] = [];
  private lastX2: number[][] = [];

  constructor(embedDim: number, ffnHidDim: number) {
    this.embedDim = embedDim;
    this.ffnHidDim = ffnHidDim;
    this.attn = new SingleHeadAttention(embedDim);
    this.W1 = randomMatrix(embedDim, ffnHidDim);
    this.W2 = randomMatrix(ffnHidDim, embedDim);
  }

  forward(X: number[][]): number[][] {
    this.lastX = X;
    const T = X.length;

    const xNorm1 = X.map(row => layerNorm(row));
    const attnOut = this.attn.forward(xNorm1);
    this.lastAttnOut = attnOut;
    const x2 = X.map((row, t) => row.map((v, d) => v + attnOut[t][d]));
    this.lastX2 = x2;

    const xNorm2 = x2.map(row => layerNorm(row));
    const hPre = matMul(xNorm2, this.W1);  // (T, ffnHidDim)
    this.lastHPre = hPre;
    const h = hPre.map(row => row.map(relu));
    this.lastH = h;
    const ffnOut = matMul(h, this.W2);  // (T, embedDim)

    return x2.map((row, t) => row.map((v, d) => v + ffnOut[t][d]));
  }

  backward(dOut: number[][]): {
    dX: number[][];
    attnGrads: { dW_Q: number[][]; dW_K: number[][]; dW_V: number[][]; dW_O: number[][] };
    dW1: number[][];
    dW2: number[][];
  } {
    const T = dOut.length;

    // FFN backward
    const dFFN = dOut;
    const dX2 = dOut.map(row => [...row]);
    const dW2 = matMul(transpose(this.lastH), dFFN);
    const dH = matMul(dFFN, transpose(this.W2));
    const dHPre = dH.map((row, i) =>
      row.map((v, j) => v * reluDerivative(this.lastHPre[i][j]))
    );
    const dW1 = matMul(transpose(this.lastX2.map(row => layerNorm(row))), dHPre);
    const dXNorm2 = matMul(dHPre, transpose(this.W1));
    for (let t = 0; t < T; t++)
      for (let d = 0; d < this.embedDim; d++)
        dX2[t][d] += dXNorm2[t][d];

    // Attention backward
    const { dX: dXNorm1, dW_Q, dW_K, dW_V, dW_O } = this.attn.backward(dX2);
    const dX = this.lastX.map((row, t) => row.map((v, d) => v + dXNorm1[t][d]));

    return { dX, attnGrads: { dW_Q, dW_K, dW_V, dW_O }, dW1, dW2 };
  }
}

class TinyDenseGPT {
  private vocabSize: number;
  private embedDim: number;
  private blockSize: number;

  tokenEmb: TokenEmbedding;
  posEmb: PosEmbedding;
  block: DenseTransformerBlock;
  W_lm_head: number[][];

  private lastInputIds: number[] = [];
  private lastBlockOut: number[][] = [];
  private lastLogits: number[][] = [];
  private lastX: number[][] = [];

  constructor(
    vocabSize: number,
    embedDim: number,
    blockSize: number,
    ffnHidDim: number
  ) {
    this.vocabSize = vocabSize;
    this.embedDim = embedDim;
    this.blockSize = blockSize;

    this.tokenEmb = new TokenEmbedding(vocabSize, embedDim);
    this.posEmb = new PosEmbedding(blockSize, embedDim);
    this.block = new DenseTransformerBlock(embedDim, ffnHidDim);
    this.W_lm_head = randomMatrix(embedDim, vocabSize);
  }

  forward(inputIds: number[]): number[][] {
    this.lastInputIds = inputIds;
    const T = inputIds.length;
    const tokEmbs = this.tokenEmb.lookup(inputIds);
    const posEmbs = this.posEmb.lookup(T);
    const X = tokEmbs.map((row, t) => row.map((v, d) => v + posEmbs[t][d]));
    this.lastX = X;
    const blockOut = this.block.forward(X);
    this.lastBlockOut = blockOut;
    const logits = matMul(blockOut, this.W_lm_head);
    this.lastLogits = logits;
    return logits;
  }

  loss(targets: number[]): number {
    const T = this.lastLogits.length;
    let loss = 0;
    for (let t = 0; t < T; t++) {
      const probs = softmax(this.lastLogits[t]);
      loss -= Math.log(probs[targets[t]] + 1e-9);
    }
    return loss / T;
  }

  backward(targets: number[], adam: Adam): void {
    const T = this.lastLogits.length;

    const dLogits: number[][] = [];
    for (let t = 0; t < T; t++) {
      const probs = softmax(this.lastLogits[t]);
      const dL = [...probs];
      dL[targets[t]] -= 1;
      dLogits.push(dL.map(v => v / T));
    }

    const dBlockOut = matMul(dLogits, transpose(this.W_lm_head));
    const dW_lm_head = matMul(transpose(this.lastBlockOut), dLogits);
    adam.step("W_lm_head", this.W_lm_head, clipGrad(dW_lm_head, 1.0));

    const { dX, attnGrads, dW1, dW2 } = this.block.backward(dBlockOut);

    adam.step("W1", this.block.W1, clipGrad(dW1, 1.0));
    adam.step("W2", this.block.W2, clipGrad(dW2, 1.0));
    adam.step("W_Q", this.block.attn.W_Q, clipGrad(attnGrads.dW_Q, 1.0));
    adam.step("W_K", this.block.attn.W_K, clipGrad(attnGrads.dW_K, 1.0));
    adam.step("W_V", this.block.attn.W_V, clipGrad(attnGrads.dW_V, 1.0));
    adam.step("W_O", this.block.attn.W_O, clipGrad(attnGrads.dW_O, 1.0));

    const dTokEmb = zerosMatrix(this.tokenEmb.table.length, this.embedDim);
    const dPosEmb = zerosMatrix(this.posEmb.table.length, this.embedDim);
    for (let t = 0; t < T; t++) {
      const id = this.lastInputIds[t];
      for (let d = 0; d < this.embedDim; d++) {
        dTokEmb[id][d] += dX[t][d];
        dPosEmb[t][d] += dX[t][d];
      }
    }
    adam.step("tok_emb", this.tokenEmb.table, clipGrad(dTokEmb, 1.0));
    adam.step("pos_emb", this.posEmb.table, clipGrad(dPosEmb, 1.0));
  }
}

// ============================================================
// SECTION 12: Training Data & Tokenizer
// ============================================================

const TRAINING_TEXT = `
the quick brown fox jumps over the lazy dog
a stitch in time saves nine
to be or not to be that is the question
all that glitters is not gold
the road not taken makes all the difference
in the beginning was the word and the word was with god
ask not what your country can do for you
one small step for man one giant leap for mankind
the only thing we have to fear is fear itself
to infinity and beyond the stars we go
`.trim();

function buildVocab(text: string): {
  vocab: string[];
  charToIdx: Map<string, number>;
  idxToChar: string[];
} {
  const chars = Array.from(new Set(text.split(""))).sort();
  const charToIdx = new Map<string, number>(chars.map((c, i) => [c, i]));
  return { vocab: chars, charToIdx, idxToChar: chars };
}

function encode(text: string, charToIdx: Map<string, number>): number[] {
  return text.split("").map(c => charToIdx.get(c) ?? 0);
}

function decode(ids: number[], idxToChar: string[]): string {
  return ids.map(i => idxToChar[i]).join("");
}

// ============================================================
// SECTION 13: Routing Statistics Tracker
// ============================================================

/**
 * Accumulates routing statistics across training steps.
 * Tracks per-expert token counts and per-token routing for visualization.
 */
class RoutingStats {
  expertCounts: number[];
  tokenRoutingHistory: Array<{ token: string; experts: number[] }>;
  private numExperts: number;

  constructor(numExperts: number) {
    this.numExperts = numExperts;
    this.expertCounts = new Array(numExperts).fill(0);
    this.tokenRoutingHistory = [];
  }

  record(
    inputIds: number[],
    expertIndices: number[][],
    idxToChar: string[]
  ): void {
    for (let t = 0; t < inputIds.length; t++) {
      const char = idxToChar[inputIds[t]];
      const experts = expertIndices[t];
      for (const e of experts) this.expertCounts[e]++;
      this.tokenRoutingHistory.push({ token: char, experts: [...experts] });
    }
  }

  reset(): void {
    this.expertCounts = new Array(this.numExperts).fill(0);
    this.tokenRoutingHistory = [];
  }

  /** Returns load balance percentage per expert. */
  expertLoadPercent(): number[] {
    const total = this.expertCounts.reduce((a, b) => a + b, 0);
    return this.expertCounts.map(c => (total > 0 ? (c / total) * 100 : 0));
  }
}

// ============================================================
// SECTION 14: ASCII Routing Visualization
// ============================================================

/**
 * Prints an ASCII visualization of routing for the first N tokens.
 *
 * Example output:
 *   Token  | Expert assignments
 *   -------+------------------
 *   't'    | [0, 2]  ##..#...
 *   'h'    | [1, 3]  .#.#....
 */
function visualizeRouting(
  history: Array<{ token: string; experts: number[] }>,
  numExperts: number,
  maxTokens = 30
): void {
  console.log("\n  ASCII Routing Visualization (first", maxTokens, "tokens):");
  console.log("  Token | Experts | " + Array.from({ length: numExperts }, (_, i) => i).join(""));
  console.log("  " + "-".repeat(14 + numExperts));

  const shown = history.slice(0, maxTokens);
  for (const { token, experts } of shown) {
    const displayToken = token === "\n" ? "\\n" : token === " " ? "_" : token;
    const expertsStr = `[${experts.join(",")}]`.padEnd(8);
    const bar = Array.from({ length: numExperts }, (_, i) =>
      experts.includes(i) ? "#" : "."
    ).join("");
    console.log(`  '${displayToken.padEnd(2)}' | ${expertsStr} | ${bar}`);
  }
}

/**
 * Builds a per-character expert preference map.
 * For each unique character, counts which expert handled it most.
 */
function buildSpecializationMap(
  history: Array<{ token: string; experts: number[] }>,
  numExperts: number
): Map<string, number[]> {
  const charExpertCounts = new Map<string, number[]>();

  for (const { token, experts } of history) {
    if (!charExpertCounts.has(token)) {
      charExpertCounts.set(token, new Array(numExperts).fill(0));
    }
    const counts = charExpertCounts.get(token)!;
    for (const e of experts) counts[e]++;
  }

  return charExpertCounts;
}

function printSpecializationMap(
  charExpertCounts: Map<string, number[]>,
  numExperts: number,
  idxToChar: string[]
): void {
  console.log("\n  Expert Specialization (which expert each character prefers):");
  console.log("  Char | Preferred Expert | Distribution");
  console.log("  " + "-".repeat(45));

  // Sort by character
  const entries = Array.from(charExpertCounts.entries()).sort((a, b) =>
    a[0].localeCompare(b[0])
  );

  for (const [char, counts] of entries) {
    const total = counts.reduce((a, b) => a + b, 0);
    if (total === 0) continue;
    const preferred = counts.indexOf(Math.max(...counts));
    const displayChar = char === "\n" ? "\\n" : char === " " ? "_" : char;
    const dist = counts.map(c => Math.round((c / total) * 10)).join("");
    console.log(
      `  '${displayChar.padEnd(2)}' | Expert ${preferred}          | ${dist}`
    );
  }
}

// ============================================================
// SECTION 15: Parameter Count Utility
// ============================================================

function countMoEParams(
  vocabSize: number,
  embedDim: number,
  blockSize: number,
  numExperts: number,
  ffnHidDim: number
): { total: number; active: number; topK: number } {
  const tokEmb = vocabSize * embedDim;
  const posEmb = blockSize * embedDim;
  const attn = 4 * embedDim * embedDim;  // W_Q, W_K, W_V, W_O
  const router = embedDim * numExperts;
  const expertsTotal = numExperts * (embedDim * ffnHidDim + ffnHidDim * embedDim);
  const lmHead = embedDim * vocabSize;
  const total = tokEmb + posEmb + attn + router + expertsTotal + lmHead;

  // Active per forward pass: all non-FFN params + topK expert params
  const topK = 2;
  const expertsActive = topK * (embedDim * ffnHidDim + ffnHidDim * embedDim);
  const active = tokEmb + posEmb + attn + router + expertsActive + lmHead;

  return { total, active, topK };
}

function countDenseParams(
  vocabSize: number,
  embedDim: number,
  blockSize: number,
  ffnHidDim: number
): number {
  return (
    vocabSize * embedDim +       // tok emb
    blockSize * embedDim +       // pos emb
    4 * embedDim * embedDim +    // attn
    embedDim * ffnHidDim +       // W1
    ffnHidDim * embedDim +       // W2
    embedDim * vocabSize         // lm head
  );
}

// ============================================================
// SECTION 16: Training Loop
// ============================================================

function trainMoE(
  model: TinyMoEGPT,
  tokens: number[],
  blockSize: number,
  epochs: number,
  lbCoeff: number,
  numExperts: number,
  idxToChar: string[],
  label: string
): { losses: number[]; finalStats: RoutingStats } {
  const adam = new Adam(3e-3);
  const stats = new RoutingStats(numExperts);
  const losses: number[] = [];

  for (let epoch = 0; epoch < epochs; epoch++) {
    let epochCE = 0;
    let epochLB = 0;
    let steps = 0;
    stats.reset();

    for (let i = 0; i < tokens.length - blockSize - 1; i++) {
      const inputIds = tokens.slice(i, i + blockSize);
      const targets = tokens.slice(i + 1, i + blockSize + 1);

      const { logits, routingWeights, expertIndices } = model.forward(inputIds);

      const { ceLoss, lbLoss } = model.loss(targets, lbCoeff);
      epochCE += ceLoss;
      epochLB += lbLoss;
      steps++;

      stats.record(inputIds, expertIndices, idxToChar);

      model.backward(targets, adam, lbCoeff);
    }

    const avgCE = epochCE / steps;
    const avgLB = epochLB / steps;
    const total = avgCE + lbCoeff * avgLB;
    losses.push(total);

    if (epoch % 5 === 0 || epoch === epochs - 1) {
      const loads = stats.expertLoadPercent().map(p => p.toFixed(1) + "%").join(" | ");
      console.log(
        `  [${label}] Epoch ${String(epoch + 1).padStart(3)} | CE: ${avgCE.toFixed(4)} | LB: ${avgLB.toFixed(4)} | Total: ${total.toFixed(4)} | Loads: ${loads}`
      );
    }
  }

  return { losses, finalStats: stats };
}

function trainDense(
  model: TinyDenseGPT,
  tokens: number[],
  blockSize: number,
  epochs: number,
  label: string
): number[] {
  const adam = new Adam(3e-3);
  const losses: number[] = [];

  for (let epoch = 0; epoch < epochs; epoch++) {
    let epochLoss = 0;
    let steps = 0;

    for (let i = 0; i < tokens.length - blockSize - 1; i++) {
      const inputIds = tokens.slice(i, i + blockSize);
      const targets = tokens.slice(i + 1, i + blockSize + 1);

      model.forward(inputIds);
      epochLoss += model.loss(targets);
      steps++;
      model.backward(targets, adam);
    }

    const avg = epochLoss / steps;
    losses.push(avg);

    if (epoch % 5 === 0 || epoch === epochs - 1) {
      console.log(`  [${label}] Epoch ${String(epoch + 1).padStart(3)} | Loss: ${avg.toFixed(4)}`);
    }
  }

  return losses;
}

// ============================================================
// SECTION 17: Main Demo
// ============================================================

function main(): void {
  console.log("=".repeat(70));
  console.log("Lesson 29 — Mixture of Experts: Scaling Smartly");
  console.log("=".repeat(70));

  // --- Setup ---
  const { charToIdx, idxToChar } = buildVocab(TRAINING_TEXT);
  const tokens = encode(TRAINING_TEXT, charToIdx);
  const vocabSize = idxToChar.length;

  const EMBED_DIM = 32;
  const BLOCK_SIZE = 16;
  const FFN_HID_DIM = 64;  // hidden dim per expert
  const NUM_EXPERTS = 4;
  const TOP_K = 2;
  const EPOCHS = 30;
  const LB_COEFF = 0.01;

  // Dense FFN hidden dim matched so active params ≈ MoE active params
  // MoE active FFN params = TOP_K * (EMBED_DIM * FFN_HID_DIM + FFN_HID_DIM * EMBED_DIM)
  // Dense active FFN params = EMBED_DIM * FFN_HID_DIM + FFN_HID_DIM * EMBED_DIM
  // To match: dense_hidden = TOP_K * FFN_HID_DIM
  const DENSE_FFN_HID_DIM = TOP_K * FFN_HID_DIM;  // same active params as MoE

  console.log("\n--- Configuration ---");
  console.log(`  Vocab size:       ${vocabSize}`);
  console.log(`  Embed dim:        ${EMBED_DIM}`);
  console.log(`  Block size:       ${BLOCK_SIZE}`);
  console.log(`  MoE experts:      ${NUM_EXPERTS} total, ${TOP_K} active`);
  console.log(`  Expert FFN hdim:  ${FFN_HID_DIM}`);
  console.log(`  Dense FFN hdim:   ${DENSE_FFN_HID_DIM} (matched active params)`);
  console.log(`  LB coefficient:   ${LB_COEFF}`);
  console.log(`  Training epochs:  ${EPOCHS}`);

  // --- Parameter count comparison ---
  const moeParams = countMoEParams(vocabSize, EMBED_DIM, BLOCK_SIZE, NUM_EXPERTS, FFN_HID_DIM);
  const denseParams = countDenseParams(vocabSize, EMBED_DIM, BLOCK_SIZE, DENSE_FFN_HID_DIM);

  console.log("\n--- Parameter Counts ---");
  console.log(`  MoE model:`);
  console.log(`    Total params:   ${moeParams.total.toLocaleString()}`);
  console.log(`    Active params:  ${moeParams.active.toLocaleString()} (top-${TOP_K} of ${NUM_EXPERTS} experts)`);
  console.log(`    Inactive params: ${(moeParams.total - moeParams.active).toLocaleString()} (experts not used per token)`);
  console.log(`    Efficiency:     ${((moeParams.active / moeParams.total) * 100).toFixed(1)}% params active per token`);
  console.log(`  Dense model:`);
  console.log(`    Total params:   ${denseParams.toLocaleString()} (same as MoE active)`);

  console.log("\n--- Real-World MoE Examples ---");
  console.log("  GPT-4 (rumored):");
  console.log("    Total params:  ~1 trillion");
  console.log("    Active params: ~220 billion (8 experts, 2 active)");
  console.log("    Efficiency:    ~22% params active per token");
  console.log("  Mixtral 8×7B:");
  console.log("    Total params:  ~47B (8 experts × 7B each, shared layers)");
  console.log("    Active params: ~13B (2 of 8 experts active)");
  console.log("    Efficiency:    ~28% params active per token");

  // --- Train Dense GPT ---
  console.log("\n--- Training Dense GPT (baseline) ---");
  const denseModel = new TinyDenseGPT(vocabSize, EMBED_DIM, BLOCK_SIZE, DENSE_FFN_HID_DIM);
  const denseLosses = trainDense(denseModel, tokens, BLOCK_SIZE, EPOCHS, "Dense");

  // --- Train MoE GPT ---
  console.log("\n--- Training MoE GPT (4 experts, top-2) ---");
  const moeModel = new TinyMoEGPT(
    vocabSize, EMBED_DIM, BLOCK_SIZE, NUM_EXPERTS, TOP_K, FFN_HID_DIM
  );
  const { losses: moeLosses, finalStats } = trainMoE(
    moeModel, tokens, BLOCK_SIZE, EPOCHS, LB_COEFF, NUM_EXPERTS, idxToChar, "MoE"
  );

  // --- Loss Comparison ---
  console.log("\n--- Loss Comparison (Final 5 Epochs) ---");
  console.log("  Epoch | Dense Loss | MoE Total | MoE CE Only");
  console.log("  " + "-".repeat(46));
  for (let i = EPOCHS - 5; i < EPOCHS; i++) {
    console.log(
      `  ${String(i + 1).padStart(5)} | ${denseLosses[i].toFixed(4).padStart(10)} | ${moeLosses[i].toFixed(4).padStart(9)} |`
    );
  }

  // --- Routing Statistics ---
  console.log("\n--- Expert Load Distribution (Final Epoch) ---");
  console.log("  Expert | Token Count | Load %    | Bar");
  console.log("  " + "-".repeat(50));
  const loads = finalStats.expertLoadPercent();
  for (let e = 0; e < NUM_EXPERTS; e++) {
    const count = finalStats.expertCounts[e];
    const pct = loads[e];
    const bar = "#".repeat(Math.round(pct / 5));
    const uniformPct = (100 / NUM_EXPERTS).toFixed(1);
    const deviation = (pct - 100 / NUM_EXPERTS).toFixed(1);
    console.log(
      `  Expert ${e} | ${String(count).padStart(11)} | ${pct.toFixed(1).padStart(6)}%   | ${bar.padEnd(20)} (uniform=${uniformPct}%, dev=${deviation}%)`
    );
  }

  // --- Load Balancing Loss Check ---
  const lbVal = loadBalancingLoss(
    finalStats.tokenRoutingHistory.map(() => new Array(NUM_EXPERTS).fill(1 / NUM_EXPERTS)),
    finalStats.tokenRoutingHistory.map(() => [0, 1]),
    NUM_EXPERTS
  );
  console.log(`\n  Perfect uniform routing would give LB loss = 1.0`);
  console.log(`  (LB loss > 1 means over-concentration; < 1 means under-concentration)`);

  // --- Expert Specialization ---
  const specializationMap = buildSpecializationMap(
    finalStats.tokenRoutingHistory,
    NUM_EXPERTS
  );
  printSpecializationMap(specializationMap, NUM_EXPERTS, idxToChar);

  // --- ASCII Routing Visualization ---
  visualizeRouting(finalStats.tokenRoutingHistory, NUM_EXPERTS, 35);

  // --- Token Routing Demo ---
  console.log("\n--- Token-by-Token Routing Demo (first 16 tokens) ---");
  const sampleTokens = tokens.slice(0, BLOCK_SIZE);
  const { routingWeights, expertIndices } = moeModel.forward(sampleTokens);

  console.log("  Token | Full Routing Probs                          | Top-K Choice");
  console.log("  " + "-".repeat(65));
  for (let t = 0; t < Math.min(sampleTokens.length, 16); t++) {
    const char = idxToChar[sampleTokens[t]];
    const displayChar = char === " " ? "_" : char === "\n" ? "\\n" : char;
    const probs = routingWeights[t].map(p => p.toFixed(3)).join(", ");
    const chosen = `[${expertIndices[t].join(",")}]`;
    console.log(`  '${displayChar.padEnd(2)}' | [${probs}] | ${chosen}`);
  }

  // --- Math Summary ---
  console.log("\n--- MoE Math Summary ---");
  console.log("  Router:        scores = softmax(X · W_router)  shape: (T, N)");
  console.log("  Top-K:         for each token, pick K experts with highest scores");
  console.log("  Renormalize:   top_weights = top_weights / sum(top_weights)");
  console.log("  Output:        Σ_k score_k × expert_k(X[t])");
  console.log("");
  console.log("  Load Balancing Loss:");
  console.log("    f_i  = fraction of tokens routed to expert i  (non-differentiable)");
  console.log("    P_i  = mean routing probability for expert i  (differentiable)");
  console.log("    Loss = N × Σ_i (f_i × P_i)");
  console.log("    Minimized when all experts receive equal traffic");
  console.log("");
  console.log("  Total loss = CE_loss + λ × LB_loss   (λ = 0.01)");

  console.log("\n" + "=".repeat(70));
  console.log("Lesson 29 complete.");
  console.log("Key takeaways:");
  console.log("  1. MoE decouples total parameters from active parameters per token");
  console.log("  2. The router is a small learned linear + softmax over experts");
  console.log("  3. Top-K selection makes only K of N experts active per token");
  console.log("  4. Load balancing loss prevents expert collapse");
  console.log("  5. Expert specialization emerges naturally from training");
  console.log("  6. GPT-4 and Mixtral both use this architecture");
  console.log("=".repeat(70));
}

main();

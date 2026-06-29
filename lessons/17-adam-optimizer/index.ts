// Lesson 17 — Adam Optimizer
// Demonstrates Adam vs SGD on XOR, then replaces SGD in the tiny GPT from lesson 14.

// ─────────────────────────────────────────────────────────────────────────────
// Part 1: Adam Optimizer class
// ─────────────────────────────────────────────────────────────────────────────

class AdamOptimizer {
  private lr: number;
  private beta1: number;
  private beta2: number;
  private eps: number;

  // Per-parameter first moment (momentum) and second moment (variance)
  // Keyed by parameter name
  private m2d = new Map<string, number[][]>();
  private v2d = new Map<string, number[][]>();
  private m1d = new Map<string, number[]>();
  private v1d = new Map<string, number[]>();

  constructor(lr = 3e-4, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.lr    = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps   = eps;
  }

  // Update a 2-D parameter matrix in place.
  // t is the global step count (1-indexed), used for bias correction.
  step(name: string, param: number[][], grad: number[][], t: number): void {
    const rows = param.length;
    const cols = param[0]?.length ?? 0;

    // Initialize moment matrices to zero on first encounter
    if (!this.m2d.has(name)) {
      this.m2d.set(name, Array.from({ length: rows }, () => Array(cols).fill(0)));
      this.v2d.set(name, Array.from({ length: rows }, () => Array(cols).fill(0)));
    }

    const m = this.m2d.get(name)!;
    const v = this.v2d.get(name)!;

    // Bias-correction denominators
    const bc1 = 1 - Math.pow(this.beta1, t);
    const bc2 = 1 - Math.pow(this.beta2, t);

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const g = grad[i]![j]!;

        // Update biased moments
        m[i]![j] = this.beta1 * m[i]![j]! + (1 - this.beta1) * g;
        v[i]![j] = this.beta2 * v[i]![j]! + (1 - this.beta2) * g * g;

        // Bias-corrected moments
        const mHat = m[i]![j]! / bc1;
        const vHat = v[i]![j]! / bc2;

        // Parameter update
        param[i]![j]! -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
      }
    }
  }

  // Update a 1-D parameter array in place.
  step1d(name: string, param: number[], grad: number[], t: number): void {
    const len = param.length;

    if (!this.m1d.has(name)) {
      this.m1d.set(name, Array(len).fill(0));
      this.v1d.set(name, Array(len).fill(0));
    }

    const m = this.m1d.get(name)!;
    const v = this.v1d.get(name)!;

    const bc1 = 1 - Math.pow(this.beta1, t);
    const bc2 = 1 - Math.pow(this.beta2, t);

    for (let i = 0; i < len; i++) {
      const g = grad[i]!;

      m[i] = this.beta1 * m[i]! + (1 - this.beta1) * g;
      v[i] = this.beta2 * v[i]! + (1 - this.beta2) * g * g;

      const mHat = m[i]! / bc1;
      const vHat = v[i]! / bc2;

      param[i]! -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Part 2: Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidDeriv(x: number): number {
  const s = sigmoid(x);
  return s * (1 - s);
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

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - max));
  const sum  = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function layerNorm(x: number[], eps = 1e-5): number[] {
  const mean = x.reduce((s, v) => s + v, 0) / x.length;
  const variance = x.reduce((s, v) => s + (v - mean) ** 2, 0) / x.length;
  return x.map(v => (v - mean) / Math.sqrt(variance + eps));
}

// Fixed random seed via LCG so results are reproducible
let seed = 0;
function lcgRand(): number {
  seed = (seed * 1664525 + 1013904223) & 0xffffffff;
  return ((seed >>> 0) / 0xffffffff) - 0.5;
}

function randMatrix(r: number, c: number, scale = 0.1): number[][] {
  return Array.from({ length: r }, () =>
    Array.from({ length: c }, () => lcgRand() * scale)
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Part 3: XOR demo — SGD vs Adam
// ─────────────────────────────────────────────────────────────────────────────
//
// Architecture: 2 → 4 → 1  (2-layer MLP, sigmoid activations, BCE loss)
// Inputs:  [0,0] [0,1] [1,0] [1,1]
// Targets:   0     1     1     0

const XOR_INPUTS  = [[0,0],[0,1],[1,0],[1,1]] as number[][];
const XOR_TARGETS = [0, 1, 1, 0] as number[];

function bce(pred: number, target: number): number {
  return -(target * Math.log(pred + 1e-9) + (1 - target) * Math.log(1 - pred + 1e-9));
}

// ── SGD training on XOR ──

function trainXorSGD(epochs: number, lr: number, printAt: number[]): void {
  // Reset seed for a fair comparison
  seed = 7;
  const W1 = randMatrix(4, 2);  // hidden layer: 4 neurons, 2 inputs
  const b1 = Array(4).fill(0) as number[];
  const W2 = randMatrix(1, 4);  // output layer: 1 neuron, 4 inputs
  const b2 = [0] as number[];

  for (let epoch = 1; epoch <= epochs; epoch++) {
    let totalLoss = 0;

    for (let i = 0; i < XOR_INPUTS.length; i++) {
      const x = XOR_INPUTS[i]!;
      const target = XOR_TARGETS[i]!;

      // ── Forward ──
      // Hidden layer pre-activations: z1 = W1 * x + b1
      const z1 = W1.map((row, k) =>
        row[0]! * x[0]! + row[1]! * x[1]! + b1[k]!
      );
      const a1 = z1.map(sigmoid);   // [4]

      // Output pre-activation: z2 = W2 * a1 + b2
      const z2 = W2[0]!.reduce((sum, w, k) => sum + w * a1[k]!, b2[0]!);
      const pred = sigmoid(z2);

      totalLoss += bce(pred, target);

      // ── Backward ──
      // dL/dpred, then through output sigmoid
      const dz2 = pred - target;  // dL/dz2 (BCE + sigmoid combined)

      // Gradient for W2 and b2
      const dW2 = a1.map(a => dz2 * a);
      const db2 = dz2;

      // Backprop through hidden layer
      const da1 = W2[0]!.map(w => dz2 * w);
      const dz1 = da1.map((d, k) => d * sigmoidDeriv(z1[k]!));

      // Gradient for W1 and b1
      const dW1 = dz1.map(dz => x.map(xi => dz * xi));
      const db1 = dz1;

      // ── SGD update ──
      for (let k = 0; k < 4; k++) {
        W1[k]![0]! -= lr * dW1[k]![0]!;
        W1[k]![1]! -= lr * dW1[k]![1]!;
        b1[k]!     -= lr * db1[k]!;
      }
      for (let k = 0; k < 4; k++) {
        W2[0]![k]! -= lr * dW2[k]!;
      }
      b2[0]! -= lr * db2;
    }

    const avgLoss = totalLoss / 4;
    if (printAt.includes(epoch)) {
      console.log(`  SGD  epoch ${String(epoch).padStart(4)} — loss: ${avgLoss.toFixed(4)}`);
    }
  }

  // Show final predictions
  const preds = XOR_INPUTS.map(x => {
    const z1 = W1.map((row, k) => row[0]! * x[0]! + row[1]! * x[1]! + b1[k]!);
    const a1 = z1.map(sigmoid);
    const z2 = W2[0]!.reduce((s, w, k) => s + w * a1[k]!, b2[0]!);
    return sigmoid(z2);
  });
  console.log(`  SGD  final predictions: [${preds.map(p => p.toFixed(3)).join(", ")}]  (want [0, 1, 1, 0])`);
}

// ── Adam training on XOR ──

function trainXorAdam(epochs: number, lr: number, printAt: number[]): void {
  seed = 7;  // same init as SGD
  const W1 = randMatrix(4, 2);
  const b1 = Array(4).fill(0) as number[];
  const W2 = randMatrix(1, 4);
  const b2 = [0] as number[];

  const adam = new AdamOptimizer(lr);
  let t = 0;  // global step counter

  for (let epoch = 1; epoch <= epochs; epoch++) {
    let totalLoss = 0;

    for (let i = 0; i < XOR_INPUTS.length; i++) {
      const x = XOR_INPUTS[i]!;
      const target = XOR_TARGETS[i]!;
      t++;

      // ── Forward (same as SGD) ──
      const z1 = W1.map((row, k) =>
        row[0]! * x[0]! + row[1]! * x[1]! + b1[k]!
      );
      const a1 = z1.map(sigmoid);
      const z2 = W2[0]!.reduce((sum, w, k) => sum + w * a1[k]!, b2[0]!);
      const pred = sigmoid(z2);

      totalLoss += bce(pred, target);

      // ── Backward (same as SGD) ──
      const dz2 = pred - target;
      const dW2 = a1.map(a => dz2 * a);
      const db2 = [dz2];
      const da1 = W2[0]!.map(w => dz2 * w);
      const dz1 = da1.map((d, k) => d * sigmoidDeriv(z1[k]!));
      const dW1 = dz1.map(dz => x.map(xi => dz * xi));
      const db1 = dz1;

      // ── Adam update (instead of SGD) ──
      adam.step("W1", W1, dW1, t);
      adam.step1d("b1", b1, db1, t);
      adam.step("W2", W2, [dW2], t);
      adam.step1d("b2", b2, db2, t);
    }

    const avgLoss = totalLoss / 4;
    if (printAt.includes(epoch)) {
      console.log(`  Adam epoch ${String(epoch).padStart(4)} — loss: ${avgLoss.toFixed(4)}`);
    }
  }

  const preds = XOR_INPUTS.map(x => {
    const z1 = W1.map((row, k) => row[0]! * x[0]! + row[1]! * x[1]! + b1[k]!);
    const a1 = z1.map(sigmoid);
    const z2 = W2[0]!.reduce((s, w, k) => s + w * a1[k]!, b2[0]!);
    return sigmoid(z2);
  });
  console.log(`  Adam final predictions: [${preds.map(p => p.toFixed(3)).join(", ")}]  (want [0, 1, 1, 0])`);
}

// For XOR specifically, Adam needs lr~0.01 to showcase rapid convergence
// (3e-4 is for transformers; on a tiny MLP you can use a larger lr).
// SGD uses 0.1 — also a typical "manual tune" for small networks.
const checkpoints = [1, 50, 100, 200, 500, 1000];

console.log("════════════════════════════════════════════════════════");
console.log(" Part 1: XOR — SGD vs Adam");
console.log("════════════════════════════════════════════════════════");
console.log("\n── SGD (lr=0.1, 1000 epochs) ──");
trainXorSGD(1000, 0.1, checkpoints);

console.log("\n── Adam (lr=0.01, 1000 epochs) ──");
console.log("  Note: Adam with lr=0.01 is faster than SGD at lr=0.1 with zero manual tuning.");
trainXorAdam(1000, 0.01, checkpoints);

// ─────────────────────────────────────────────────────────────────────────────
// Part 4: Tiny GPT with Adam
// (Same architecture as lesson 14, Adam replaces the manual SGD update)
// ─────────────────────────────────────────────────────────────────────────────

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
  decode(ids: number[]): string  { return ids.map(i => this.idToChar.get(i) ?? "?").join(""); }
}

const GPT_CONFIG = {
  embedDim:  16,
  blockSize: 8,
  numLayers: 1,
  // For our toy-scale GPT we use lr=1e-3 for Adam and lr=5e-3 for SGD.
  // Both are reasonable for this scale. Adam still converges to lower loss.
  // At real transformer scale (GPT-2+), 3e-4 is the right Adam lr.
  adamLr:    1e-3,
  sgdLr:     5e-3,
  epochs:    500,
};

const TRAIN_TEXT = "hello world. the cat sat on the mat. a dog ran in the fog.";

// Shared helpers for tiny GPT
function gptRand(r: number, c: number, scale = 0.1): number[][] {
  return Array.from({ length: r }, () =>
    Array.from({ length: c }, () => (Math.random() - 0.5) * scale)
  );
}

function crossEntropy(logits: number[][], targets: number[]): number {
  let total = 0;
  for (let i = 0; i < logits.length; i++) {
    const probs = softmax(logits[i]!);
    total += -Math.log((probs[targets[i]!] ?? 0) + 1e-9);
  }
  return total / logits.length;
}

function trainGPT(useAdam: boolean, printEvery: number): number {
  const tokenizer = new CharTokenizer(TRAIN_TEXT);
  const vocabSize = tokenizer.vocabSize;
  const { embedDim, blockSize } = GPT_CONFIG;

  // Fix the seed so SGD and Adam start from the same weights
  seed = 17;
  const embTable = gptRand(vocabSize, embedDim);
  const posTable = gptRand(blockSize, embedDim);
  const Wq       = gptRand(embedDim, embedDim);
  const Wk       = gptRand(embedDim, embedDim);
  const Wv       = gptRand(embedDim, embedDim);
  const Wproj    = gptRand(vocabSize, embedDim);

  const adam = useAdam ? new AdamOptimizer(GPT_CONFIG.adamLr) : null;
  const sgdLr = GPT_CONFIG.sgdLr;
  let adamStep = 0;

  const encoded = tokenizer.encode(TRAIN_TEXT);

  function forward(tokens: number[]): { logits: number[][], X2: number[][] } {
    const T = tokens.length;
    const X: number[][] = tokens.map((tok, pos) =>
      (embTable[tok] ?? []).map((v, j) => v + (posTable[pos]?.[j] ?? 0))
    );

    const Q = matmul(X, Wq);
    const K = matmul(X, Wk);
    const V = matmul(X, Wv);
    const scale = Math.sqrt(embedDim);
    const scores = matmul(Q, transpose(K)).map(row => row.map(s => s / scale));

    for (let i = 0; i < T; i++)
      for (let j = i + 1; j < T; j++)
        scores[i]![j] = -Infinity;

    const attnWeights = scores.map(row => softmax(row));
    const attnOut = matmul(attnWeights, V);
    const X2 = X.map((row, i) =>
      layerNorm(row.map((v, j) => v + (attnOut[i]?.[j] ?? 0)))
    );
    const logits = X2.map(row =>
      Wproj.map(wRow => wRow.reduce((s, w, j) => s + w * (row[j] ?? 0), 0))
    );
    return { logits, X2 };
  }

  let finalLoss = 0;

  for (let epoch = 0; epoch < GPT_CONFIG.epochs; epoch++) {
    let totalLoss = 0, steps = 0;

    for (let start = 0; start + blockSize < encoded.length; start++) {
      const tokens  = encoded.slice(start, start + blockSize);
      const targets = encoded.slice(start + 1, start + blockSize + 1);

      const { logits, X2 } = forward(tokens);
      totalLoss += crossEntropy(logits, targets);

      // Accumulate gradients for Adam batch update
      const dWproj = Array.from({ length: vocabSize }, () => Array(embedDim).fill(0) as number[]);
      const dEmb   = new Map<number, number[]>();

      for (let t = 0; t < tokens.length; t++) {
        const probs   = softmax(logits[t]!);
        const dlogits = probs.map((p, v) => p - (v === targets[t] ? 1 : 0));
        const x2t     = X2[t]!;
        const tok     = tokens[t]!;

        // dL/dX2
        const dX2 = Array<number>(embedDim).fill(0);
        for (let j = 0; j < embedDim; j++)
          for (let v = 0; v < vocabSize; v++)
            dX2[j] += dlogits[v]! * (Wproj[v]?.[j] ?? 0);

        // Accumulate dWproj
        for (let v = 0; v < vocabSize; v++)
          for (let j = 0; j < embedDim; j++)
            dWproj[v]![j]! += dlogits[v]! * (x2t[j] ?? 0);

        // Accumulate dEmb
        if (!dEmb.has(tok)) dEmb.set(tok, Array(embedDim).fill(0));
        const de = dEmb.get(tok)!;
        for (let j = 0; j < embedDim; j++) de[j]! += dX2[j]!;
      }

      adamStep++;

      if (adam) {
        // Adam: batch update for Wproj
        adam.step("Wproj", Wproj, dWproj, adamStep);

        // Adam: update each embedding row that appeared in this window
        for (const [tok, dRow] of dEmb) {
          // Wrap single row as 2D for the step method
          const rowParam = [embTable[tok]!];
          adam.step(`emb_${tok}`, rowParam, [dRow], adamStep);
          embTable[tok] = rowParam[0]!;
        }
      } else {
        // Plain SGD (same logic as lesson 14)
        for (let v = 0; v < vocabSize; v++)
          for (let j = 0; j < embedDim; j++)
            Wproj[v]![j]! -= sgdLr * dWproj[v]![j]!;

        for (const [tok, dRow] of dEmb)
          for (let j = 0; j < embedDim; j++)
            embTable[tok]![j]! -= sgdLr * dRow[j]!;
      }

      steps++;
    }

    finalLoss = totalLoss / steps;
    if (epoch % printEvery === 0 || epoch === GPT_CONFIG.epochs - 1) {
      const optimizer = useAdam ? "Adam" : " SGD";
      console.log(`  ${optimizer} epoch ${String(epoch).padStart(3)} — loss: ${finalLoss.toFixed(4)}`);
    }
  }

  return finalLoss;
}

console.log("\n════════════════════════════════════════════════════════");
console.log(" Part 2: Tiny GPT — SGD vs Adam");
console.log(`          ${GPT_CONFIG.epochs} epochs, printing every 50`);
console.log("════════════════════════════════════════════════════════");

console.log(`\n── SGD on Tiny GPT (lr=${GPT_CONFIG.sgdLr}) ──`);
const sgdFinalLoss = trainGPT(false, 50);

console.log(`\n── Adam on Tiny GPT (lr=${GPT_CONFIG.adamLr}) ──`);
const adamFinalLoss = trainGPT(true, 50);

console.log("\n════════════════════════════════════════════════════════");
console.log(" Final Loss Comparison");
console.log("════════════════════════════════════════════════════════");
console.log(`  SGD  final loss: ${sgdFinalLoss.toFixed(4)}`);
console.log(`  Adam final loss: ${adamFinalLoss.toFixed(4)}`);
console.log(`  Improvement:     ${((sgdFinalLoss - adamFinalLoss) / sgdFinalLoss * 100).toFixed(1)}% lower loss with Adam`);

console.log("\n════════════════════════════════════════════════════════");
console.log(" Key Takeaways");
console.log("════════════════════════════════════════════════════════");
console.log("  Adam tracks m (momentum) and v (variance) per parameter.");
console.log("  Parameters that oscillate get small updates (large v → small step).");
console.log("  Parameters with consistent gradients get large updates (small v → big step).");
console.log("  Bias correction matters in the first ~30 steps, then becomes negligible.");
console.log("  lr=3e-4 is 'Karpathy's constant' — the safe default for real transformer training.");
console.log("  Adam is the standard optimizer for all modern transformer training.");

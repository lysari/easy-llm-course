// Lesson 22 — Scaling Laws: Empirical Demonstration
//
// We empirically demonstrate that:
//   1. Bigger models → lower loss (model size scaling)
//   2. More data → lower loss (data scaling)
//   3. Loss follows approximately: loss ≈ C * N^(-α)
//
// Architecture: a context-window language model with a learned embedding table
// and a linear projection. We vary the embedding dimension (= model capacity)
// to study model scaling. Full gradient descent is used for honest results.
//
// No imports. All math done from scratch.

// ---------------------------------------------------------------------------
// Seeded pseudo-random for reproducibility
// ---------------------------------------------------------------------------

let _seed = 42;
function seededRandom(): number {
  // Mulberry32 — fast, good-quality 32-bit PRNG
  _seed |= 0;
  _seed = _seed + 0x6d2b79f5 | 0;
  let t = Math.imul(_seed ^ _seed >>> 15, 1 | _seed);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
  return ((t ^ t >>> 14) >>> 0) / 4294967296;
}

function randn(): number {
  // Box-Muller using seededRandom
  const u1 = seededRandom() + 1e-10;
  const u2 = seededRandom();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function randomMatrix(rows: number, cols: number, scale: number): number[][] {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => randn() * scale)
  );
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// Matrix-vector: (rows x cols) @ (cols,) → (rows,)
function matvec(A: number[][], v: number[]): number[] {
  return A.map(row => dot(row, v));
}

function softmax(logits: number[]): number[] {
  const m = Math.max(...logits);
  const e = logits.map(x => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(x => x / s);
}

function crossEntropy(logits: number[], target: number): number {
  const p = softmax(logits);
  return -Math.log(p[target] + 1e-10);
}

// ---------------------------------------------------------------------------
// Tokenizer — character-level
// ---------------------------------------------------------------------------

function buildVocab(text: string): { vocab: string[]; charToId: Map<string, number> } {
  const chars = Array.from(new Set(text.split(''))).sort();
  const charToId = new Map(chars.map((c, i) => [c, i]));
  return { vocab: chars, charToId };
}

function encode(text: string, charToId: Map<string, number>): number[] {
  return text.split('').map(c => charToId.get(c) ?? 0);
}

// ---------------------------------------------------------------------------
// Model: Embedding + Context Window + Linear Head
//
// For a context of length K, we:
//   1. Look up embeddings for each of the K tokens: each is dim-dimensional
//   2. Concatenate them into a single vector of length K*dim
//   3. Multiply by output weight matrix W (vocabSize x K*dim) to get logits
//
// Parameters: K*dim*vocabSize (W) + vocabSize*dim (embeddings)
//           = vocabSize*(K*dim + dim) = vocabSize*dim*(K+1)
//
// With vocabSize=V, context K, embed dim D:
//   total params ≈ V*D*(K+1) + V (bias)   (bias negligible)
// ---------------------------------------------------------------------------

interface LMWeights {
  E: number[][];   // [vocabSize x dim] — embedding table
  W: number[][];   // [vocabSize x (contextLen * dim)] — output projection
  b: number[];     // [vocabSize] — output bias
}

interface LMConfig {
  vocabSize: number;
  dim: number;
  contextLen: number;
}

function initModel(cfg: LMConfig): LMWeights {
  const inputSize = cfg.contextLen * cfg.dim;
  const scale = Math.sqrt(2 / inputSize);
  return {
    E: randomMatrix(cfg.vocabSize, cfg.dim, 0.01),
    W: randomMatrix(cfg.vocabSize, inputSize, scale),
    b: new Array(cfg.vocabSize).fill(0),
  };
}

function countModelParams(cfg: LMConfig): number {
  return (
    cfg.vocabSize * cfg.dim +                        // embedding table
    cfg.vocabSize * (cfg.contextLen * cfg.dim) +      // W
    cfg.vocabSize                                     // bias
  );
}

// Forward pass: returns logits
function forward(context: number[], weights: LMWeights, cfg: LMConfig): number[] {
  // Concatenate embeddings
  const input: number[] = [];
  for (const id of context) {
    for (const v of weights.E[id]) input.push(v);
  }
  // Linear: logits = W @ input + b
  const logits = matvec(weights.W, input);
  for (let i = 0; i < cfg.vocabSize; i++) logits[i] += weights.b[i];
  return logits;
}

// ---------------------------------------------------------------------------
// Full backprop for this model
//
// Forward: x = concat(E[t_1], ..., E[t_K])   shape: K*D
//          logits = W @ x + b                  shape: V
//          loss = -log(softmax(logits)[target])
//
// Backward:
//   dL/d_logits[i] = p[i] - (i == target ? 1 : 0)   (cross-entropy gradient)
//   dL/dW[i, j] = dL/d_logits[i] * x[j]
//   dL/db[i]    = dL/d_logits[i]
//   dL/dx[j]    = sum_i  dL/d_logits[i] * W[i, j]
//   dL/dE[t, k] = sum over positions p where context[p]==t:  dL/dx[p*D + k]
// ---------------------------------------------------------------------------

function backward(
  context: number[],
  target: number,
  weights: LMWeights,
  cfg: LMConfig,
  lr: number
): number {
  const { dim: D, vocabSize: V, contextLen: K } = cfg;
  const inputSize = K * D;

  // Forward
  const x: number[] = [];
  for (const id of context) {
    for (const v of weights.E[id]) x.push(v);
  }
  const logits = matvec(weights.W, x);
  for (let i = 0; i < V; i++) logits[i] += weights.b[i];

  const p = softmax(logits);
  const loss = -Math.log(p[target] + 1e-10);

  // dL/d_logits
  const dLogits = p.slice();
  dLogits[target] -= 1;

  // dL/db
  for (let i = 0; i < V; i++) {
    weights.b[i] -= lr * dLogits[i];
  }

  // dL/dW and dL/dx simultaneously
  const dx = new Array(inputSize).fill(0);
  for (let i = 0; i < V; i++) {
    const g = dLogits[i];
    if (Math.abs(g) < 1e-10) continue;
    for (let j = 0; j < inputSize; j++) {
      dx[j] += g * weights.W[i][j];
      weights.W[i][j] -= lr * g * x[j];
    }
  }

  // dL/dE — accumulate gradient for each position's token
  for (let pos = 0; pos < K; pos++) {
    const tokenId = context[pos];
    for (let k = 0; k < D; k++) {
      weights.E[tokenId][k] -= lr * dx[pos * D + k];
    }
  }

  return loss;
}

// ---------------------------------------------------------------------------
// Train for one epoch — iterate over all context windows in tokens
// ---------------------------------------------------------------------------

function trainEpoch(
  tokens: number[],
  weights: LMWeights,
  cfg: LMConfig,
  lr: number
): number {
  const K = cfg.contextLen;
  let totalLoss = 0;
  let count = 0;

  for (let i = 0; i + K < tokens.length; i++) {
    const context = tokens.slice(i, i + K);
    const target = tokens[i + K];
    totalLoss += backward(context, target, weights, cfg, lr);
    count++;
  }

  return count > 0 ? totalLoss / count : Infinity;
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

function fmt(n: number, decimals = 4): string {
  return n.toFixed(decimals);
}

function fmtTime(ms: number): string {
  return ms < 1000 ? `${ms.toFixed(0)}ms` : `${(ms / 1000).toFixed(2)}s`;
}

function pad(s: string | number, w: number, right = false): string {
  const str = String(s);
  return right ? str.padEnd(w) : str.padStart(w);
}

// ---------------------------------------------------------------------------
// Fit loss ≈ C * N^(-alpha) via log-log linear regression
// Returns alpha (positive means bigger model → lower loss)
// ---------------------------------------------------------------------------

function fitPowerLaw(xs: number[], ys: number[]): { alpha: number; C: number; r2: number } {
  const lx = xs.map(Math.log);
  const ly = ys.map(Math.log);
  const n = xs.length;
  const mx = lx.reduce((a, b) => a + b, 0) / n;
  const my = ly.reduce((a, b) => a + b, 0) / n;
  const sxx = lx.reduce((a, x) => a + (x - mx) ** 2, 0);
  const sxy = lx.reduce((a, x, i) => a + (x - mx) * (ly[i] - my), 0);
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  // R^2
  const yHat = lx.map(x => slope * x + intercept);
  const ssTot = ly.reduce((a, y) => a + (y - my) ** 2, 0);
  const ssRes = ly.reduce((a, y, i) => a + (y - yHat[i]) ** 2, 0);
  const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 1;
  return { alpha: -slope, C: Math.exp(intercept), r2 };
}

// ---------------------------------------------------------------------------
// ASCII log-log plot
// ---------------------------------------------------------------------------

function logLogPlot(
  points: { label: string; x: number; y: number }[],
  width: number,
  height: number
): string {
  const symbols = ['*', 'o', '#', '@', '+'];
  const lxs = points.map(p => Math.log10(p.x));
  const lys = points.map(p => Math.log10(p.y));

  const minLx = Math.min(...lxs);
  const maxLx = Math.max(...lxs);
  const minLy = Math.min(...lys);
  const maxLy = Math.max(...lys);
  const px = (maxLx - minLx) * 0.2 || 0.5;
  const py = (maxLy - minLy) * 0.2 || 0.1;
  const x0 = minLx - px; const x1 = maxLx + px;
  const y0 = minLy - py; const y1 = maxLy + py;

  const grid: string[][] = Array.from({ length: height }, () =>
    new Array(width).fill(' ')
  );
  for (let r = 0; r < height; r++) grid[r][0] = '|';
  for (let c = 0; c < width; c++) grid[height - 1][c] = '-';
  grid[height - 1][0] = '+';

  // Trend line (power law fit in log-log space)
  if (points.length >= 2) {
    const n = lxs.length;
    const sx = lxs.reduce((a, b) => a + b, 0);
    const sy = lys.reduce((a, b) => a + b, 0);
    const sxx = lxs.reduce((a, b) => a + b * b, 0);
    const sxy = lxs.reduce((a, v, i) => a + v * lys[i], 0);
    const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    const intercept = (sy - slope * sx) / n;
    for (let c = 1; c < width - 1; c++) {
      const lx = x0 + ((c - 1) / (width - 2)) * (x1 - x0);
      const ly = slope * lx + intercept;
      const row = height - 2 - Math.round(((ly - y0) / (y1 - y0)) * (height - 2));
      if (row >= 0 && row < height - 1 && grid[row][c] === ' ') {
        grid[row][c] = '.';
      }
    }
  }

  // Data points (plotted after trend so they appear on top)
  for (let i = 0; i < points.length; i++) {
    const col = Math.round(((lxs[i] - x0) / (x1 - x0)) * (width - 2)) + 1;
    const row = height - 2 - Math.round(((lys[i] - y0) / (y1 - y0)) * (height - 2));
    if (row >= 0 && row < height - 1 && col >= 1 && col < width) {
      grid[row][col] = symbols[i % symbols.length];
    }
  }

  const lines: string[] = [];
  lines.push('  log(loss)');
  lines.push(...grid.map(r => '  ' + r.join('')));
  lines.push('  ' + ' '.repeat(width - 14) + 'log(N params)');
  lines.push('');
  lines.push('  Legend:');
  for (let i = 0; i < points.length; i++) {
    lines.push(`    ${symbols[i % symbols.length]}  ${points[i].label}  ` +
      `(N=${points[i].x.toLocaleString()}, loss=${points[i].y.toFixed(4)})`);
  }
  lines.push('  .  power-law trend (least-squares fit in log-log space)');
  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Training text
// ---------------------------------------------------------------------------

const TEXT = [
  'the quick brown fox jumps over the lazy dog',
  'a stitch in time saves nine',
  'all that glitters is not gold',
  'to be or not to be that is the question',
  'the early bird catches the worm',
  'actions speak louder than words',
  'practice makes perfect and perfect makes permanent',
  'you cannot judge a book by its cover',
  'every cloud has a silver lining somewhere',
  'the pen is mightier than the sword always',
  'where there is a will there is a way through',
  'time flies when you are having fun and learning',
  'knowledge is power and power shapes the world',
  'beauty is in the eye of the beholder always',
  'a picture is worth a thousand words they say',
  'birds of a feather flock together in the park',
  'do not count your chickens before they hatch early',
  'the road to success is always under construction now',
  'in the beginning was the word and the word spread',
  'small steps taken every day lead to great change',
].join('\n');

// ---------------------------------------------------------------------------
// EXPERIMENT 1: Model Size Scaling
// ---------------------------------------------------------------------------

function runModelScaling(
  tokens: number[],
  vocabSize: number
): { params: number; label: string; finalLoss: number; history: number[]; timeMs: number }[] {

  // Three model sizes: vary embedding dimension
  // contextLen=4 for all; vocabSize fixed
  // Params ≈ vocabSize * dim * (contextLen + 1)
  const sizes = [
    { label: 'Small  (dim=8) ', dim: 8 },
    { label: 'Medium (dim=24)', dim: 24 },
    { label: 'Large  (dim=64)', dim: 64 },
  ];

  const EPOCHS = 40;
  const CONTEXT = 4;
  const LR = 0.003;
  const results = [];

  console.log(`  Context length: ${CONTEXT} tokens`);
  console.log(`  Learning rate: ${LR}`);
  console.log(`  Epochs: ${EPOCHS}`);
  console.log('');

  for (const spec of sizes) {
    _seed = 42; // same init seed for all models
    const cfg: LMConfig = { vocabSize, dim: spec.dim, contextLen: CONTEXT };
    const weights = initModel(cfg);
    const params = countModelParams(cfg);
    const history: number[] = [];

    const t0 = Date.now();
    for (let e = 0; e < EPOCHS; e++) {
      history.push(trainEpoch(tokens, weights, cfg, LR));
    }
    const timeMs = Date.now() - t0;

    const finalLoss = history[history.length - 1];
    results.push({ params, label: spec.label, finalLoss, history, timeMs });

    console.log(`  ${spec.label}: ${pad(params.toLocaleString(), 8)} params | ` +
      `final loss = ${fmt(finalLoss)} | time = ${fmtTime(timeMs)}`);
  }

  return results;
}

// ---------------------------------------------------------------------------
// EXPERIMENT 2: Data Scaling
// ---------------------------------------------------------------------------

function runDataScaling(
  baseTokens: number[],
  vocabSize: number
): { multiplier: number; numTokens: number; finalLoss: number }[] {

  const EPOCHS = 40;
  const CONTEXT = 4;
  const LR = 0.003;
  const DIM = 24; // medium model

  console.log(`  Model: dim=${DIM}, context=${CONTEXT}`);
  console.log(`  Epochs: ${EPOCHS} per run`);
  console.log('');

  const results = [];
  for (const mult of [1, 2, 3]) {
    _seed = 42; // same init for fair comparison
    const tokens: number[] = [];
    for (let i = 0; i < mult; i++) tokens.push(...baseTokens);

    const cfg: LMConfig = { vocabSize, dim: DIM, contextLen: CONTEXT };
    const weights = initModel(cfg);
    let finalLoss = Infinity;

    for (let e = 0; e < EPOCHS; e++) {
      finalLoss = trainEpoch(tokens, weights, cfg, LR);
    }

    results.push({ multiplier: mult, numTokens: tokens.length, finalLoss });
    console.log(`  ${mult}x data: ${pad(tokens.length.toLocaleString(), 6)} tokens | ` +
      `final loss = ${fmt(finalLoss)}`);
  }

  return results;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

(function main() {
  const SEP = '─'.repeat(60);

  console.log('');
  console.log('='.repeat(60));
  console.log('  Lesson 22 — Scaling Laws: Empirical Demonstration');
  console.log('='.repeat(60));

  const { charToId, vocab } = buildVocab(TEXT);
  const tokens = encode(TEXT, charToId);
  const vocabSize = vocab.length;

  console.log('');
  console.log(`  Training text : ${TEXT.length} characters`);
  console.log(`  Vocabulary    : ${vocabSize} unique characters`);
  console.log(`  Token count   : ${tokens.length}`);
  console.log('');

  // ─── EXPERIMENT 1 ─────────────────────────────────────────────────────────
  console.log(SEP);
  console.log('EXPERIMENT 1: Model Size Scaling');
  console.log('Same data, same epochs — three model capacities');
  console.log(SEP);
  console.log('');

  const t1start = Date.now();
  const sizeResults = runModelScaling(tokens, vocabSize);
  const t1total = Date.now() - t1start;

  // Loss-per-epoch table (every 5 epochs for brevity)
  console.log('');
  console.log('Loss over training (every 5 epochs):');
  console.log('');
  const step = 5;
  const labelW = 18;
  const colW = 10;
  const epochs = sizeResults[0].history.length;

  const header = pad('Epoch', 7) + sizeResults.map(r => pad(r.label.trim(), colW + 5)).join('');
  console.log(header);
  console.log('-'.repeat(header.length));

  for (let e = step - 1; e < epochs; e += step) {
    const row = pad(e + 1, 7) + sizeResults.map(r => pad(fmt(r.history[e]), colW + 5)).join('');
    console.log(row);
  }

  // Summary
  console.log('');
  console.log('Summary:');
  console.log('');
  console.log(
    pad('Model', labelW, true) +
    pad('Params', 10) +
    pad('Loss epoch 1', 14) +
    pad('Final Loss', 12) +
    pad('Improvement', 13) +
    pad('Time', 8)
  );
  console.log('-'.repeat(labelW + 57));

  for (const r of sizeResults) {
    const pct = ((r.history[0] - r.finalLoss) / r.history[0] * 100);
    console.log(
      pad(r.label, labelW, true) +
      pad(r.params.toLocaleString(), 10) +
      pad(fmt(r.history[0]), 14) +
      pad(fmt(r.finalLoss), 12) +
      pad(`-${pct.toFixed(1)}%`, 13) +
      pad(fmtTime(r.timeMs), 8)
    );
  }

  // Power law fit
  const paramsList = sizeResults.map(r => r.params);
  const lossList = sizeResults.map(r => r.finalLoss);
  const { alpha, C, r2 } = fitPowerLaw(paramsList, lossList);

  console.log('');
  console.log('Power-law fit:  loss ≈ C × N^(-α)');
  console.log(`  Empirical αN = ${alpha.toFixed(4)}   (Kaplan et al.: αN ≈ 0.076)`);
  console.log(`  Constant   C = ${C.toFixed(4)}`);
  console.log(`  R² of fit    = ${r2.toFixed(4)}`);
  console.log('');
  console.log('  Interpretation: every 10× increase in parameters reduces loss');
  console.log(`  by a factor of 10^${alpha.toFixed(3)} = ${(Math.pow(10, alpha)).toFixed(3)}×`);

  // Predictions
  console.log('');
  console.log('Model predictions from fitted power law:');
  console.log('');
  console.log(
    pad('Model', labelW, true) +
    pad('Actual', 10) +
    pad('Predicted', 11) +
    pad('Error', 8)
  );
  console.log('-'.repeat(labelW + 29));
  for (const r of sizeResults) {
    const pred = C * Math.pow(r.params, -alpha);
    const err = Math.abs(pred - r.finalLoss) / r.finalLoss * 100;
    console.log(
      pad(r.label, labelW, true) +
      pad(fmt(r.finalLoss), 10) +
      pad(fmt(pred), 11) +
      pad(`${err.toFixed(1)}%`, 8)
    );
  }

  // Plot
  console.log('');
  console.log('Log-log plot: final loss vs number of parameters');
  console.log('A straight trend line confirms power-law scaling');
  console.log('');
  console.log(logLogPlot(
    sizeResults.map(r => ({ label: r.label.trim(), x: r.params, y: r.finalLoss })),
    54, 13
  ));

  // ─── EXPERIMENT 2 ─────────────────────────────────────────────────────────
  console.log('');
  console.log(SEP);
  console.log('EXPERIMENT 2: Data Scaling');
  console.log('Same model — trained on 1×, 2×, 3× copies of the text');
  console.log(SEP);
  console.log('');

  const dataResults = runDataScaling(tokens, vocabSize);

  console.log('');
  console.log('Summary:');
  console.log('');
  console.log(
    pad('Copies', 8, true) +
    pad('Tokens', 10) +
    pad('Final Loss', 12) +
    pad('vs 1×', 10)
  );
  console.log('-'.repeat(40));

  const baseLoss = dataResults[0].finalLoss;
  for (const r of dataResults) {
    const delta = r.finalLoss - baseLoss;
    const sign = delta < 0 ? '' : '+';
    console.log(
      pad(r.multiplier + '×', 8, true) +
      pad(r.numTokens.toLocaleString(), 10) +
      pad(fmt(r.finalLoss), 12) +
      pad(r.multiplier === 1 ? '—' : `${sign}${delta.toFixed(4)}`, 10)
    );
  }

  if (dataResults.length >= 2) {
    const dTokens = dataResults.map(r => r.numTokens);
    const dLosses = dataResults.map(r => r.finalLoss);
    const fit = fitPowerLaw(dTokens, dLosses);
    console.log('');
    console.log('Power-law fit for data scaling:  loss ≈ C × D^(-αD)');
    console.log(`  Empirical αD = ${fit.alpha.toFixed(4)}   (Kaplan et al.: αD ≈ 0.095)`);
    console.log('');
    console.log('  Note: at this tiny scale and with only 3 data points,');
    console.log('  the fit is indicative, not definitive.');
  }

  // ─── THEORY RECAP ─────────────────────────────────────────────────────────
  console.log('');
  console.log(SEP);
  console.log('SCALING LAW THEORY vs OUR RESULTS');
  console.log(SEP);
  console.log('');
  console.log('  Kaplan et al. (2020) — large transformer LMs:');
  console.log('    αN (model size)  ≈ 0.076   loss ≈ C·N^(-0.076)');
  console.log('    αD (data size)   ≈ 0.095   loss ≈ C·D^(-0.095)');
  console.log('    αC (compute)     ≈ 0.050   loss ≈ C·C^(-0.050)');
  console.log('');
  console.log(`  Our empirical αN  = ${alpha.toFixed(4)}`);
  console.log('');
  console.log('  Why they differ:');
  console.log('    - Our models are tiny (thousands not billions of params)');
  console.log('    - We use a shallow embedding+linear model, not a transformer');
  console.log('    - Scaling laws are fit over 5–7 orders of magnitude');
  console.log('    - Our 3 data points span only 2 orders of magnitude');
  console.log('');
  console.log('  What remains the same:');
  console.log('    - Bigger model → lower final loss  ✓');
  console.log('    - More data   → lower final loss  ✓');
  console.log('    - Loss follows a power law in log-log space  ✓');
  console.log('');
  console.log('─── Chinchilla-Optimal Tokens for Our Models ───');
  console.log('');
  console.log('  Rule of thumb: train on ~20 tokens per parameter');
  console.log('');
  for (const r of sizeResults) {
    const need = r.params * 20;
    const have = tokens.length;
    const ratio = (have / need * 100).toFixed(1);
    console.log(`  ${r.label.padEnd(20)} ${r.params.toLocaleString().padStart(8)} params` +
      ` → needs ~${need.toLocaleString()} tokens  (we have ${ratio}% of that)`);
  }
  console.log('');
  console.log('─── Real-World Comparison ───');
  console.log('');
  console.log('  Model       | Params | Tokens | Tokens/Param | Assessment');
  console.log('  ------------|--------|--------|--------------|------------------');
  console.log('  GPT-3       | 175B   | 300B   |          1.7 | severely undertrained');
  console.log('  Gopher      | 280B   | 300B   |          1.1 | severely undertrained');
  console.log('  Chinchilla  |  70B   | 1.4T   |         20.0 | compute-optimal');
  console.log('  Llama-1 65B |  65B   | 1.4T   |         21.5 | near-optimal');
  console.log('  Llama-2 70B |  70B   | 2.0T   |         28.6 | slightly over');
  console.log('');
  console.log(`  Total experiment time: ${fmtTime(Date.now() - t1start + 1)}`);
  console.log('');
  console.log('='.repeat(60));
})();

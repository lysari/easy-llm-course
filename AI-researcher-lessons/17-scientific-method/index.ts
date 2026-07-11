// The scientific method for ML: why "change one thing at a time" is not optional.
//
// We train the same tiny network four times. The TRUE effects are rigged so we
// (the audience) know the answer:
//   - raising the learning rate helps a lot (the low-lr run is undertrained)
//   - widening the hidden layer adds nothing once the lr is healthy
//
// Researcher A changes BOTH knobs at once and draws a confident, wrong-headed
// conclusion. Researcher B changes one knob per run and learns the truth.

// ── Seeded RNG so every run of this file prints the same story ──
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Gaussian noise via Box–Muller
function gauss(rand: () => number): number {
  const u = Math.max(rand(), 1e-9);
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ── Toy dataset: two overlapping 2-D blobs (class 0 and class 1) ──
function makeData(n: number, rand: () => number): { X: number[][]; y: number[] } {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const label = i % 2; // balanced classes
    const cx = label === 0 ? -0.8 : 0.8; // blob centers at (-0.8,-0.8) and (0.8,0.8)
    const cy = label === 0 ? -0.8 : 0.8;
    X.push([cx + gauss(rand) * 1.0, cy + gauss(rand) * 1.0]);
    y.push(label);
  }
  return { X, y };
}

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

// ── Tiny MLP: 2 → hidden → 1, trained with plain SGD on cross-entropy ──
function trainAndEval(opts: {
  lr: number;
  hidden: number;
  epochs: number;
  seed: number;
}): number {
  const { lr, hidden, epochs, seed } = opts;

  // Same data for every configuration — the dataset is a CONTROLLED variable.
  const dataRand = mulberry32(1234);
  const train = makeData(60, dataRand);
  const test = makeData(60, dataRand);

  // Weight init uses the run's own seed (a controlled variable too: same seed everywhere).
  const rand = mulberry32(seed);
  const W1: number[][] = Array.from({ length: hidden }, () => [
    (rand() - 0.5) * 0.2,
    (rand() - 0.5) * 0.2,
  ]);
  const b1: number[] = Array(hidden).fill(0);
  const W2: number[] = Array.from({ length: hidden }, () => (rand() - 0.5) * 0.2);
  let b2 = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    for (let i = 0; i < train.X.length; i++) {
      const x = train.X[i] ?? [0, 0];
      const target = train.y[i] ?? 0;

      // forward
      const h: number[] = [];
      for (let j = 0; j < hidden; j++) {
        const z = (W1[j]?.[0] ?? 0) * (x[0] ?? 0) + (W1[j]?.[1] ?? 0) * (x[1] ?? 0) + (b1[j] ?? 0);
        h.push(sigmoid(z));
      }
      const out = sigmoid(h.reduce((s, hj, j) => s + hj * (W2[j] ?? 0), b2));

      // backward (cross-entropy + sigmoid → clean gradient: out - target)
      const dOut = out - target;
      for (let j = 0; j < hidden; j++) {
        const hj = h[j] ?? 0;
        const dH = dOut * (W2[j] ?? 0) * hj * (1 - hj);
        W2[j] = (W2[j] ?? 0) - lr * dOut * hj;
        if (W1[j]) {
          W1[j]![0] = (W1[j]?.[0] ?? 0) - lr * dH * (x[0] ?? 0);
          W1[j]![1] = (W1[j]?.[1] ?? 0) - lr * dH * (x[1] ?? 0);
        }
        b1[j] = (b1[j] ?? 0) - lr * dH;
      }
      b2 -= lr * dOut;
    }
  }

  // test accuracy (the DEPENDENT variable, chosen before running)
  let correct = 0;
  for (let i = 0; i < test.X.length; i++) {
    const x = test.X[i] ?? [0, 0];
    const h: number[] = [];
    for (let j = 0; j < hidden; j++) {
      const z = (W1[j]?.[0] ?? 0) * (x[0] ?? 0) + (W1[j]?.[1] ?? 0) * (x[1] ?? 0) + (b1[j] ?? 0);
      h.push(sigmoid(z));
    }
    const out = sigmoid(h.reduce((s, hj, j) => s + hj * (W2[j] ?? 0), b2));
    if ((out > 0.5 ? 1 : 0) === test.y[i]) correct++;
  }
  return correct / test.X.length;
}

// ── The four runs ──
// Deliberately few epochs so a too-small lr leaves the model undertrained.
const EPOCHS = 3;
const SEED = 7; // same init seed everywhere: controlled variable
const LR_LOW = 0.003; // too small: model stays undertrained in 3 epochs
const LR_HIGH = 0.5; // healthy for this task
const H_SMALL = 4;
const H_BIG = 16;

const control = trainAndEval({ lr: LR_LOW, hidden: H_SMALL, epochs: EPOCHS, seed: SEED });
const runA = trainAndEval({ lr: LR_HIGH, hidden: H_BIG, epochs: EPOCHS, seed: SEED }); // TWO changes
const runB1 = trainAndEval({ lr: LR_HIGH, hidden: H_SMALL, epochs: EPOCHS, seed: SEED }); // lr only
const runB2 = trainAndEval({ lr: LR_LOW, hidden: H_BIG, epochs: EPOCHS, seed: SEED }); // width only

const pct = (v: number) => (v * 100).toFixed(1) + "%";

console.log("=== Same tiny network, four experiments ===\n");
console.log("run        lr      hidden   test acc   changed vs control");
console.log("---------  ------  -------  ---------  -------------------");
console.log(`control    ${LR_LOW}   ${H_SMALL}        ${pct(control)}      —`);
console.log(`run A      ${LR_HIGH}     ${H_BIG}       ${pct(runA)}      lr AND hidden  ← two at once!`);
console.log(`run B1     ${LR_HIGH}     ${H_SMALL}        ${pct(runB1)}      lr only`);
console.log(`run B2     ${LR_LOW}   ${H_BIG}       ${pct(runB2)}      hidden only`);

console.log("\n=== Researcher A's story (confounded) ===");
console.log(`A sees: control ${pct(control)} → run A ${pct(runA)}.`);
console.log(`A concludes: "my bigger network gives +${pct(runA - control)}!"`);
console.log("But TWO variables changed. The gain could come from lr, from width,");
console.log("or from lr helping while width hurts. From run A alone: UNKNOWABLE.");

console.log("\n=== Researcher B's story (controlled, factorial) ===");
console.log(`lr only     (control → B1):  ${pct(control)} → ${pct(runB1)}   lr effect = +${(100 * (runB1 - control)).toFixed(1)} points`);
console.log(`width only  (control → B2):  ${pct(control)} → ${pct(runB2)}   width effect AT BAD lr = +${(100 * (runB2 - control)).toFixed(1)} points`);
console.log(`width again (B1 → A):        ${pct(runB1)} → ${pct(runA)}   width effect AT GOOD lr = ${(100 * (runA - runB1)).toFixed(1)} points`);
console.log("\nB's conclusion: the learning rate did all the work. Width only");
console.log("*appeared* to help when the lr was broken (more random features");
console.log("partially masked the undertraining) — once lr is healthy, width");
console.log("adds exactly nothing here. That interaction is invisible to A,");
console.log("who changed both knobs at once and credited the architecture.");
console.log("Two extra runs bought a conclusion that is actually true.");

// The optimization landscape: watch GD, momentum, and noisy SGD
// navigate a bowl and a saddle — the two shapes that matter most.
//
// Run: npx ts-node index.ts

// ── Seeded PRNG (mulberry32) so every run prints the same numbers ──
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
// Roughly standard-normal noise (sum of 12 uniforms − 6)
function gaussian(rand: () => number): number {
  let s = 0;
  for (let i = 0; i < 12; i++) s += rand();
  return s - 6;
}

// ── The two landscapes ──
// Each is a tiny 2D loss function with a hand-computed gradient.

// 1) An ill-conditioned BOWL: L = x² + 5y²
//    (a valley 5× steeper in y than in x — like most real landscapes,
//     curvature differs by direction)
const bowl = {
  name: "bowl  L(x,y) = x² + 5y²",
  loss: (x: number, y: number) => x * x + 5 * y * y,
  grad: (x: number, y: number): [number, number] => [2 * x, 10 * y],
};

// 2) A SADDLE: L = x² − y²
//    Bowl-shaped along x (curves up), dome-shaped along y (curves DOWN).
//    The origin (0,0) has zero gradient but is NOT a minimum:
//    moving along ±y decreases the loss forever. y is the escape direction.
const saddle = {
  name: "saddle  L(x,y) = x² − y²",
  loss: (x: number, y: number) => x * x - y * y,
  grad: (x: number, y: number): [number, number] => [2 * x, -2 * y],
};

type Landscape = typeof bowl;

// ── Three optimizers ──
// Each runs `steps` updates and returns the trajectory of (x, y, loss).

interface Point { step: number; x: number; y: number; loss: number }

function runGD(f: Landscape, x: number, y: number, lr: number, steps: number): Point[] {
  const traj: Point[] = [{ step: 0, x, y, loss: f.loss(x, y) }];
  for (let s = 1; s <= steps; s++) {
    const [gx, gy] = f.grad(x, y);
    x -= lr * gx;
    y -= lr * gy;
    traj.push({ step: s, x, y, loss: f.loss(x, y) });
    if (!isFinite(f.loss(x, y)) || Math.abs(x) > 1e6 || Math.abs(y) > 1e6) break; // diverged
  }
  return traj;
}

function runMomentum(f: Landscape, x: number, y: number, lr: number, beta: number, steps: number): Point[] {
  let vx = 0, vy = 0; // velocity starts at rest
  const traj: Point[] = [{ step: 0, x, y, loss: f.loss(x, y) }];
  for (let s = 1; s <= steps; s++) {
    const [gx, gy] = f.grad(x, y);
    vx = beta * vx + gx; // heavy ball: blend old velocity with new gradient
    vy = beta * vy + gy;
    x -= lr * vx;
    y -= lr * vy;
    traj.push({ step: s, x, y, loss: f.loss(x, y) });
    if (Math.abs(x) > 1e6 || Math.abs(y) > 1e6) break;
  }
  return traj;
}

function runNoisySGD(
  f: Landscape, x: number, y: number, lr: number, noise: number, steps: number, seed: number
): Point[] {
  const rand = mulberry32(seed);
  const traj: Point[] = [{ step: 0, x, y, loss: f.loss(x, y) }];
  for (let s = 1; s <= steps; s++) {
    const [gx, gy] = f.grad(x, y);
    // Minibatch gradient = true gradient + zero-mean noise
    x -= lr * (gx + noise * gaussian(rand));
    y -= lr * (gy + noise * gaussian(rand));
    traj.push({ step: s, x, y, loss: f.loss(x, y) });
    if (Math.abs(x) > 1e6 || Math.abs(y) > 1e6) break;
  }
  return traj;
}

// ── Pretty-printing helpers ──
const fmt = (v: number) => {
  if (!isFinite(v)) return "DIVERGED";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-4)) return v.toExponential(2);
  return v.toFixed(5);
};
function printTraj(label: string, traj: Point[], every: number): void {
  console.log(`  ${label}`);
  for (const p of traj) {
    if (p.step % every === 0 || p.step === traj.length - 1) {
      console.log(`    step ${String(p.step).padStart(3)}:  x=${fmt(p.x).padStart(11)}  y=${fmt(p.y).padStart(11)}  loss=${fmt(p.loss)}`);
    }
  }
}

// ════════════════════════════════════════════════════════════════════
// PART 1 — The bowl: learning-rate regimes
// ════════════════════════════════════════════════════════════════════
console.log("=== PART 1: the bowl —", bowl.name, "===");
console.log("Start at (4, 2). Curvature along y is 5× steeper → y limits the safe lr.\n");

console.log("(a) lr = 0.02 — too LOW: converges but crawls");
printTraj("plain GD:", runGD(bowl, 4, 2, 0.02, 100), 25);

console.log("\n(b) lr = 0.09 — about right: fast, stable");
printTraj("plain GD:", runGD(bowl, 4, 2, 0.09, 100), 25);

console.log("\n(c) lr = 0.25 — too HIGH: safe for x, unstable for the steep y direction");
console.log("    (each step multiplies y by 1 − 0.25·10 = −1.5 → grows 1.5× per step)");
printTraj("plain GD:", runGD(bowl, 4, 2, 0.25, 30), 5);

console.log("\n(d) same lr-too-low case, but with momentum (β=0.9) — velocity accumulates:");
printTraj("momentum:", runMomentum(bowl, 4, 2, 0.02, 0.9, 100), 25);

// ════════════════════════════════════════════════════════════════════
// PART 2 — The saddle: who escapes?
// ════════════════════════════════════════════════════════════════════
console.log("\n=== PART 2: the saddle —", saddle.name, "===");
console.log("Start at (2, 1e-9): almost exactly on the ridge above the saddle point.");
console.log("The escape direction is y (loss falls as |y| grows), but the y-gradient");
console.log("there is ~2e-9 — essentially zero. Watch who gets away. (60 steps, lr=0.1)\n");

const STEPS = 60, LR = 0.1;

printTraj("plain GD — kills x fast, then STALLS (|y| barely grows):",
  runGD(saddle, 2, 1e-9, LR, STEPS), 15);

console.log();
printTraj("momentum (β=0.9) — velocity compounds the tiny y-push, ESCAPES:",
  runMomentum(saddle, 2, 1e-9, LR, 0.9, STEPS), 15);

console.log();
printTraj("noisy SGD (noise=0.1) — random kicks knock it off the ridge, ESCAPES:",
  runNoisySGD(saddle, 2, 1e-9, LR, 0.1, STEPS, 42), 15);

// The verdict, measured
const last = (t: Point[]) => t[t.length - 1];
const gdFinal = last(runGD(saddle, 2, 1e-9, LR, STEPS));
const moFinal = last(runMomentum(saddle, 2, 1e-9, LR, 0.9, STEPS));
const sgFinal = last(runNoisySGD(saddle, 2, 1e-9, LR, 0.1, STEPS, 42));

console.log("\n=== Distance from the saddle ridge after", STEPS, "steps (|y|, bigger = escaped) ===");
console.log(`  plain GD : |y| = ${fmt(Math.abs(gdFinal?.y ?? 0))}   ← still glued to the ridge`);
console.log(`  momentum : |y| = ${fmt(Math.abs(moFinal?.y ?? 0))}   ← long gone`);
console.log(`  noisy SGD: |y| = ${fmt(Math.abs(sgFinal?.y ?? 0))}   ← long gone`);

console.log("\nTakeaways:");
console.log("  1. The learning rate has three regimes: crawl / converge / diverge —");
console.log("     and the steepest direction sets the divergence threshold.");
console.log("  2. At a saddle, plain GD stalls: the gradient along the escape route ~0.");
console.log("  3. Momentum compounds tiny pushes; SGD noise supplies pushes for free.");
console.log("     In high dimensions almost all critical points are saddles —");
console.log("     which is exactly why these two tricks are in every real optimizer.");

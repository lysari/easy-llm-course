// Calculus for research: derivatives as sensitivity, the chain rule,
// and GRADIENT CHECKING — verifying hand-written backprop against
// dumb-but-trustworthy finite differences. The #1 debugging tool for
// anyone who implements gradients by hand.

// ── Part 1: numeric vs analytic derivative of a simple function ──

// f(x) = x³ − 2x + 1     analytic derivative: f'(x) = 3x² − 2
const f = (x: number) => x * x * x - 2 * x + 1;
const fPrime = (x: number) => 3 * x * x - 2;

// Centered finite difference: nudge both ways, divide by 2h.
// This IS the definition of a derivative, executed literally.
function numericDerivative(fn: (x: number) => number, x: number, h: number): number {
  return (fn(x + h) - fn(x - h)) / (2 * h);
}

console.log("=== 1. Derivative = sensitivity to a nudge ===");
console.log("f(x) = x³ − 2x + 1 at x = 2.   Analytic f'(2) = 3·4 − 2 = 10");
for (const h of [0.1, 0.01, 0.001, 0.0001]) {
  const num = numericDerivative(f, 2, h);
  console.log(
    `  h = ${h.toString().padEnd(6)} numeric ≈ ${num.toFixed(8)}   error = ${Math.abs(num - fPrime(2)).toExponential(2)}`
  );
}
console.log("→ shrink the nudge, converge on the true sensitivity (error ~ h²).");

// ── Part 2: the chain rule, verified numerically ─────────────────
console.log("\n=== 2. Chain rule: sensitivities multiply ===");
// y = 3u + 1 where u = x².  dy/dx = (dy/du)(du/dx) = 3 · 2x
const g = (x: number) => 3 * (x * x) + 1;
const x0 = 2;
console.log(`y = 3u+1, u = x², at x = ${x0}:`);
console.log(`  du/dx = 2x = 4,  dy/du = 3,  chain rule → dy/dx = 3×4 = 12`);
console.log(`  numeric check: ${numericDerivative(g, x0, 1e-5).toFixed(6)} ✓`);

// ── Part 3: a tiny 2-layer network with hand-written backprop ────
//
//   x (2) ──W1,b1──▶ pre1 (3) ──tanh──▶ h (3) ──W2,b2──▶ ŷ (1)
//   loss L = (ŷ − y)²
//
// We derive every gradient by hand (chain rule, applied backward),
// then gradient-check EVERY weight numerically.

type Params = { W1: number[][]; b1: number[]; W2: number[][]; b2: number[] };

// Fixed "random" weights so the output is reproducible.
const params: Params = {
  W1: [
    [0.5, -0.3],
    [0.8, 0.2],
    [-0.6, 0.9],
  ],
  b1: [0.1, -0.2, 0.05],
  W2: [[0.7, -0.5, 0.3]],
  b2: [0.2],
};
const x = [1.0, -0.5]; // input
const y = 0.8; // target

// Forward pass. Returns loss plus intermediates needed by backprop.
function forward(p: Params) {
  const pre1 = p.b1.map((bi, i) =>
    bi + (p.W1[i]?.[0] ?? 0) * (x[0] ?? 0) + (p.W1[i]?.[1] ?? 0) * (x[1] ?? 0)
  );
  const h = pre1.map(Math.tanh);
  const yHat =
    (p.b2[0] ?? 0) + h.reduce((s, hj, j) => s + (p.W2[0]?.[j] ?? 0) * hj, 0);
  const loss = (yHat - y) ** 2;
  return { pre1, h, yHat, loss };
}

// Backward pass: chain rule, layer by layer, back to front.
function backward(p: Params) {
  const { pre1, h, yHat } = forward(p);

  // L = (ŷ − y)²             → ∂L/∂ŷ = 2(ŷ − y)
  const dyHat = 2 * (yHat - y);

  // ŷ = b2 + Σⱼ W2[0][j]·hⱼ  → ∂ŷ/∂W2[0][j] = hⱼ,  ∂ŷ/∂b2 = 1,  ∂ŷ/∂hⱼ = W2[0][j]
  const dW2 = [h.map(hj => dyHat * hj)];
  const db2 = [dyHat];
  const dh = h.map((_, j) => dyHat * (p.W2[0]?.[j] ?? 0));

  // hⱼ = tanh(pre1ⱼ)          → ∂hⱼ/∂pre1ⱼ = 1 − tanh²(pre1ⱼ)
  const dpre1 = dh.map((dhj, j) => dhj * (1 - Math.tanh(pre1[j] ?? 0) ** 2));

  // pre1ᵢ = b1ᵢ + Σₖ W1[i][k]·xₖ → ∂pre1ᵢ/∂W1[i][k] = xₖ,  ∂pre1ᵢ/∂b1ᵢ = 1
  const dW1 = dpre1.map(dp => [dp * (x[0] ?? 0), dp * (x[1] ?? 0)]);
  const db1 = dpre1.slice();

  return { dW1, db1, dW2, db2 };
}

console.log("\n=== 3. Tiny 2-layer network: forward + hand-written backprop ===");
const { yHat, loss } = forward(params);
console.log(`prediction ŷ = ${yHat.toFixed(6)}, target y = ${y}, loss = ${loss.toFixed(6)}`);

// ── Part 4: gradient check — the lie detector ────────────────────
// For every single parameter: nudge it ±h, re-run the WHOLE forward
// pass, and compare the finite-difference slope to backprop's answer.

const H = 1e-5;
function numericGradOfParam(get: () => number, set: (v: number) => void): number {
  const orig = get();
  set(orig + H);
  const lossPlus = forward(params).loss;
  set(orig - H);
  const lossMinus = forward(params).loss;
  set(orig); // restore!
  return (lossPlus - lossMinus) / (2 * H);
}

function relError(a: number, b: number): number {
  return Math.abs(a - b) / Math.max(Math.abs(a) + Math.abs(b), 1e-8);
}

const grads = backward(params);
console.log("\n=== 4. Gradient check: analytic (backprop) vs numeric ===");
console.log("param      analytic       numeric        rel.error");

let maxRelError = 0;
const check = (name: string, analytic: number, get: () => number, set: (v: number) => void) => {
  const numeric = numericGradOfParam(get, set);
  const err = relError(analytic, numeric);
  maxRelError = Math.max(maxRelError, err);
  console.log(
    `${name.padEnd(10)} ${analytic.toFixed(8).padStart(12)}  ${numeric.toFixed(8).padStart(12)}  ${err.toExponential(2)}`
  );
};

for (let i = 0; i < 3; i++) {
  for (let k = 0; k < 2; k++) {
    check(`W1[${i}][${k}]`, grads.dW1[i]?.[k] ?? 0,
      () => params.W1[i]?.[k] ?? 0, v => { params.W1[i]![k] = v; });
  }
  check(`b1[${i}]`, grads.db1[i] ?? 0,
    () => params.b1[i] ?? 0, v => { params.b1[i] = v; });
}
for (let j = 0; j < 3; j++) {
  check(`W2[0][${j}]`, grads.dW2[0]?.[j] ?? 0,
    () => params.W2[0]?.[j] ?? 0, v => { params.W2[0]![j] = v; });
}
check("b2[0]", grads.db2[0] ?? 0,
  () => params.b2[0] ?? 0, v => { params.b2[0] = v; });

console.log(`\nMAX relative error: ${maxRelError.toExponential(3)}`);
console.log(maxRelError < 1e-6
  ? "→ < 1e-6: backprop implementation verified. Safe to trust experiments built on it."
  : "→ TOO LARGE: there is a bug in the analytic gradient. Do not train with this!");

// ── Part 5: prove the gradient actually descends ──────────────────
console.log("\n=== 5. One gradient step lowers the loss ===");
const eta = 0.1;
const g2 = backward(params);
for (let i = 0; i < 3; i++) {
  for (let k = 0; k < 2; k++) params.W1[i]![k]! -= eta * (g2.dW1[i]?.[k] ?? 0);
  params.b1[i]! -= eta * (g2.db1[i] ?? 0);
  params.W2[0]![i]! -= eta * (g2.dW2[0]?.[i] ?? 0);
}
params.b2[0]! -= eta * (g2.db2[0] ?? 0);
console.log(`loss before step: ${loss.toFixed(6)}`);
console.log(`loss after step:  ${forward(params).loss.toFixed(6)}   (moved downhill, as ∇ promised)`);

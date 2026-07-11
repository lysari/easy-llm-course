// Numerical computing: where perfect math meets 64 bits and loses.
// Float weirdness, overflow → NaN, catastrophic cancellation,
// naive vs stable softmax, log-sum-exp, simulated float16 training,
// and why memory layout (not math) decides speed.

// ── 1. Float weirdness ───────────────────────────────────────────
console.log("=== 1. Floating point: scientific notation in base 2 ===");
console.log(`0.1 + 0.2 === 0.3   → ${((0.1 + 0.2) === 0.3)}`);
console.log(`0.1 + 0.2           → ${0.1 + 0.2}`);
console.log(`(0.1 is really ${(0.1).toFixed(20)} — binary can't write 1/10 exactly)`);

console.log(`\nmachine epsilon: ${Number.EPSILON}  (smallest x where 1 + x ≠ 1)`);
console.log(`1 + EPSILON === 1     → ${1 + Number.EPSILON === 1}`);
console.log(`1 + EPSILON/2 === 1   → ${1 + Number.EPSILON / 2 === 1}   (below resolution: vanishes)`);
console.log(`1e16 + 1 === 1e16     → ${1e16 + 1 === 1e16}   (big + small = big, small LOST)`);
console.log("→ moral: never (a === b) for computed floats; use |a − b| < tolerance.");

// ── 2. Overflow, underflow, and radioactive NaN ──────────────────
console.log("\n=== 2. Overflow / underflow: falling off the edges ===");
console.log(`biggest float64: ${Number.MAX_VALUE}`);
console.log(`Math.exp(709)  = ${Math.exp(709)}   (just fits)`);
console.log(`Math.exp(710)  = ${Math.exp(710)}   (overflow → Infinity, silently)`);
console.log(`Math.exp(-746) = ${Math.exp(-746)}   (underflow → 0, silently)`);
console.log(`Infinity - Infinity = ${Infinity - Infinity}`);
console.log(`0 / 0               = ${0 / 0}`);
const nan: number = 0 / 0; // NaN, via a route TypeScript can't see through
console.log(`NaN === NaN         → ${nan === nan}   (the one value not equal to itself)`);
console.log(`NaN * 0 + 5         = ${NaN * 0 + 5}   (NaN poisons everything it touches)`);

// ── 3. Catastrophic cancellation ─────────────────────────────────
// One-sided derivative of f(x)=x² at x=1 (true answer: 2).
// Calculus says smaller h → better. Floating point disagrees:
// f(x+h) − f(x) subtracts two nearly-equal numbers, and below
// h ≈ 1e-8 the surviving digits are pure rounding noise.
console.log("\n=== 3. Catastrophic cancellation: subtraction eats digits ===");
const f = (x: number) => x * x;
console.log("estimating f'(1) = 2 with (f(1+h) − f(1)) / h :");
console.log("  h        estimate            error");
for (let e = 2; e <= 15; e += 1) {
  const h = 10 ** -e;
  const est = (f(1 + h) - f(1)) / h;
  const err = Math.abs(est - 2);
  const marker = e === 8 ? "  ← sweet spot" : e >= 12 ? "  ← noise wins" : "";
  console.log(`  1e-${String(e).padEnd(3)}  ${est.toFixed(12)}  ${err.toExponential(1)}${marker}`);
}
console.log("→ error falls (math) then RISES (cancellation). Best h ≈ √ε ≈ 1e-8.");
console.log("  This is why Lesson 02's gradient check used h = 1e-5, not 1e-12.");

// ── 4. Naive vs stable softmax on large logits ───────────────────
console.log("\n=== 4. Softmax: the naive version NaNs on real logits ===");

function naiveSoftmax(logits: number[]): number[] {
  const exps = logits.map(Math.exp); // exp(1000) = Infinity → doomed
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum); // Infinity / Infinity = NaN
}

function stableSoftmax(logits: number[]): number[] {
  const m = Math.max(...logits); // subtract the max: exactly equal math,
  const exps = logits.map(l => Math.exp(l - m)); // biggest exponent now exp(0)=1
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

const small = [1, 2, 3];
const big = [1000, 1001, 1002]; // same GAPS as [1,2,3] → same softmax!
const fmtArr = (v: number[]) => `[${v.map(x => x.toFixed(4)).join(", ")}]`;
console.log(`logits [1,2,3]:          naive ${fmtArr(naiveSoftmax(small))}`);
console.log(`logits [1000,1001,1002]: naive ${fmtArr(naiveSoftmax(big))}   ← overflow → NaN`);
console.log(`logits [1000,1001,1002]: stable ${fmtArr(stableSoftmax(big))}`);
console.log("→ shifting logits by a constant cancels out of the formula —");
console.log("  so shift by max(x) and overflow becomes impossible. Same answer:");
console.log(`  stable([1,2,3]) = ${fmtArr(stableSoftmax(small))}`);

// ── 5. Log-sum-exp: the general-purpose tool ─────────────────────
// LSE(x) = log Σ exp(xᵢ) appears inside every cross-entropy.
console.log("\n=== 5. Log-sum-exp: LSE(x) = m + log Σ exp(xᵢ − m) ===");

const naiveLSE = (x: number[]) =>
  Math.log(x.map(Math.exp).reduce((a, b) => a + b, 0));
const stableLSE = (x: number[]) => {
  const m = Math.max(...x);
  return m + Math.log(x.map(v => Math.exp(v - m)).reduce((a, b) => a + b, 0));
};

console.log(`x = [1000, 1001, 1002]:   naive LSE = ${naiveLSE(big)}   (log(Inf))`);
console.log(`                          stable LSE = ${stableLSE(big).toFixed(6)}`);
const veryNeg = [-1000, -1001, -1002];
console.log(`x = [-1000,-1001,-1002]:  naive LSE = ${naiveLSE(veryNeg)}   (log(0), underflow)`);
console.log(`                          stable LSE = ${stableLSE(veryNeg).toFixed(6)}`);
// log-softmax = x − LSE(x): how real cross-entropy losses are computed.
const logProbs = big.map(l => l - stableLSE(big));
console.log(`stable log-softmax of big logits: ${fmtArr(logProbs)}`);
console.log(`(exp of those = ${fmtArr(logProbs.map(Math.exp))} — matches stable softmax ✓)`);

// ── 6. Simulated float16: why training keeps float32 weights ─────
// float16 has ~10 mantissa bits (~3 decimal digits) and max 65504.
// Simulate f16 rounding by keeping only 10 bits of fraction.
console.log("\n=== 6. float16 vs bf16: why mixed precision exists ===");

function toFloat16ish(x: number): number {
  if (x === 0 || !isFinite(x)) return x;
  if (Math.abs(x) > 65504) return x > 0 ? Infinity : -Infinity; // f16 overflow!
  const exp = Math.floor(Math.log2(Math.abs(x)));
  const scale = 2 ** (exp - 10); // keep 10 fractional bits
  return Math.round(x / scale) * scale;
}

console.log(`float16 max is 65504:  toFloat16(70000) = ${toFloat16ish(70000)}   (bf16 would survive: range ~1e38)`);
console.log(`float16(3.14159265) = ${toFloat16ish(3.14159265)}   (~3 decimal digits survive)`);

// The killer: tiny gradient updates round to NOTHING in 16-bit.
const gradientUpdate = 0.0001;
let w16 = 1.0;
let w32 = 1.0;
for (let step = 0; step < 1000; step++) {
  w16 = toFloat16ish(w16 - gradientUpdate); // update rounds away: 1.0 − 0.0001 → 1.0
  w32 = w32 - gradientUpdate;
}
console.log(`\n1000 updates of −0.0001 on w = 1.0:`);
console.log(`  float32-style weight: ${w32.toFixed(4)}   (learned: moved by 0.1)`);
console.log(`  float16-style weight: ${w16.toFixed(4)}   (update < precision → LEARNING SILENTLY STOPPED)`);
console.log("→ this is why mixed precision keeps float32 'master weights'");
console.log("  and only casts to 16-bit for the fast matmuls.");

// ── 7. Vectorization: memory layout decides speed ────────────────
// Same arithmetic, different memory behavior. (JS lacks SIMD control,
// but allocation & access patterns already show the effect.)
console.log("\n=== 7. Why GPUs exist: same math, different speed ===");
const N = 3_000_000;
const a = new Float64Array(N).fill(1.5);
const b = new Float64Array(N).fill(2.5);

// Version A: allocate a fresh JS array every iteration (hot-loop garbage)
let t0 = performance.now();
let sinkA = 0;
for (let rep = 0; rep < 3; rep++) {
  const out: number[] = [];
  for (let i = 0; i < N; i++) out.push((a[i] ?? 0) * (b[i] ?? 0) + 1);
  sinkA += out[N - 1] ?? 0;
}
const timeA = performance.now() - t0;

// Version B: preallocated typed array, contiguous sequential access
t0 = performance.now();
let sinkB = 0;
const out = new Float64Array(N);
for (let rep = 0; rep < 3; rep++) {
  for (let i = 0; i < N; i++) out[i] = (a[i] ?? 0) * (b[i] ?? 0) + 1;
  sinkB += out[N - 1] ?? 0;
}
const timeB = performance.now() - t0;

console.log(`multiply-add over ${N.toLocaleString()} elements × 3 reps:`);
console.log(`  allocate-per-loop, push():   ${timeA.toFixed(1)} ms`);
console.log(`  preallocated Float64Array:   ${timeB.toFixed(1)} ms   (${(timeA / timeB).toFixed(1)}× faster, same math)`);
console.log(`  (checksums equal: ${sinkA === sinkB})`);
console.log("→ contiguous, preallocated, uniform operations = what CPUs vectorize");
console.log("  and what GPUs run on 17,000 cores at once. Arrange your math as");
console.log("  big regular array ops (matmuls) and the hardware rewards you 100×.");

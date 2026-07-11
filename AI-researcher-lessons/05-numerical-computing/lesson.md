# Lesson 05 — Numerical Computing

---

## The problem this lesson solves

Your math is perfect. Your code implements it faithfully. And your loss is `NaN`.

Welcome to the gap between mathematics and machines. On paper, numbers are infinitely precise; in silicon, every number is squeezed into 16, 32, or 64 bits — and the squeezing *leaks*. Most "my training diverged" mysteries, half of all `NaN` losses, and entire subfields of ML engineering (mixed precision, stable kernels) live in that leak.

Researchers hit this wall constantly, because research code is *new* code — nobody has already added the stability tricks for you. This lesson teaches the failure modes and the standard fixes, ending with the single most important one: **log-sum-exp**, the reason every real softmax subtracts the max.

Start with the famous shocker:

```
0.1 + 0.2 === 0.3     →  false
0.1 + 0.2             →  0.30000000000000004
```

Not a bug. A design decision from 1985 that every computer on Earth shares.

---

## Floating point: scientific notation in base 2

A float stores numbers the way scientists write them — a sign, some significant digits, and an exponent — but in binary:

```
value = (−1)^sign × 1.mantissa × 2^exponent

float64 (JS "number"):  1 sign bit │ 11 exponent bits │ 52 mantissa bits
```
- **sign**: positive or negative
- **mantissa** (fraction): the significant digits — ~15–16 decimal digits' worth in float64
- **exponent**: where the binary point sits — how big or small the number is

The catch: floats are binary, and **0.1 has no finite binary representation** — just like 1/3 = 0.3333… never terminates in decimal, 1/10 = 0.000110011001100…₂ never terminates in binary. So the machine stores the nearest representable number:

```
"0.1" is actually stored as  0.1000000000000000055511151231257827…
"0.2" is actually stored as  0.2000000000000000111022302462515654…
their sum rounds to          0.3000000000000000444089209850062616…
"0.3" is actually stored as  0.2999999999999999888977697537403455…
                              ↑ two DIFFERENT nearest-neighbors → !==
```

Consequences you'll meet in real research code:

```
never:   if (a === b)            for floats computed two different ways
instead: if (Math.abs(a - b) < tolerance)

floats have finite resolution: at magnitude 1, steps of ~2.2e-16 (float64)
"machine epsilon" ε = 2⁻⁵² ≈ 2.22e-16: the smallest x with 1 + x ≠ 1
so: 1e16 + 1 === 1e16   — adding 1 literally does nothing
```

That last line matters: **big + small = big, small lost**. Sum a billion small gradient contributions into one large accumulator and the late ones can vanish entirely.

---

## Overflow and underflow: falling off the edges

The exponent has limits, so representable numbers have edges:

```
float64:  max ~1.8e308,  min normal ~2.2e-308
float32:  max ~3.4e38,   min normal ~1.2e-38
float16:  max  65504(!), min normal ~6.1e-5
```

Fall off the top: **overflow** → `Infinity`. Fall off the bottom: **underflow** → `0`. Both are silent — no crash, just poison that spreads:

```
Math.exp(710)          → Infinity        (e^710 > 1.8e308)
Math.exp(-746)         → 0               (underflow)
Infinity - Infinity    → NaN
0 / 0                  → NaN
Infinity / Infinity    → NaN
```

`NaN` (Not a Number) is radioactive: any arithmetic touching it yields NaN, and `NaN !== NaN` (the one value not equal to itself). One overflowed logit → NaN loss → NaN gradients → **every weight in the model becomes NaN in a single training step**. This is the standard autopsy of a "loss went to NaN at step 40,000" incident.

Note the float16 line: max 65504. That's not astronomy — activations hit 65504 easily. Hold that thought for the mixed-precision section.

---

## Catastrophic cancellation: subtraction eats digits

The subtlest failure. Subtract two nearly-equal numbers and the leading digits — the ones stored *accurately* — annihilate, leaving only the noisy tail:

```
a = 1.23456789012345 6789…   (16 good digits, then junk)
b = 1.23456789012345 1234…

a − b = 0.00000000000000 5555…
        └── all 15 accurate digits cancelled;
            the "answer" is 100% rounding noise ──┘
```

Classic real-world sighting — the one-sided derivative estimate from [Lesson 02](../02-calculus-and-gradients/lesson.md):

```
f'(x) ≈ (f(x+h) − f(x)) / h
```

Make h *smaller* and the math error shrinks but cancellation grows — below h ≈ 1e-8 the estimate gets **worse**. That's why gradient checking uses centered differences with h ≈ 1e-5, not h = 1e-12: a numerical-computing fact, not a calculus fact. The code demonstrates the full U-shaped error curve.

Another classic: computing variance as `E[X²] − E[X]²` (two big nearly-equal numbers) instead of `E[(X−μ)²]` — the textbook formula can return *negative variance*.

---

## The log-sum-exp trick: why softmax subtracts the max

Now the centerpiece. Softmax ([../../lessons/11-softmax/lesson.md](../../lessons/11-softmax/lesson.md)) converts logits to probabilities:

```
softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)
```
- `xᵢ`: logit (raw score) for option i
- `exp(xᵢ)`: makes everything positive and amplifies differences
- dividing by the sum: makes it all add to 1

Perfect on paper. On a machine, feed it logits `[1000, 1001, 1002]` — mild numbers by late-training standards:

```
exp(1000) = Infinity   (overflow: e^1000 ≫ 1.8e308)
softmax   = [Inf/Inf, Inf/Inf, Inf/Inf] = [NaN, NaN, NaN]
```

The fix is one line, justified by one line of algebra. For any constant m:

```
exp(xᵢ − m) / Σⱼ exp(xⱼ − m)
  = (exp(xᵢ)·e⁻ᵐ) / (Σⱼ exp(xⱼ)·e⁻ᵐ)      ← e⁻ᵐ factors out of the sum
  = exp(xᵢ) / Σⱼ exp(xⱼ)                    ← and cancels. EXACTLY equal.
```

Shifting all logits by any constant **changes nothing mathematically**. So choose `m = max(x)`:

```
x = [1000, 1001, 1002] → x − m = [−2, −1, 0]
exp: [0.135, 0.368, 1.0]   ← the largest exp() argument is now 0: never overflows
softmax = [0.090, 0.245, 0.665]   ✓
```

The general tool is **log-sum-exp**, needed because cross-entropy takes the *log* of softmax's denominator:

```
LSE(x) = log Σᵢ exp(xᵢ)  =  m + log Σᵢ exp(xᵢ − m),   m = max(x)
```

- naive `log(sum(exp(x)))` overflows for big logits and underflows to `log(0) = −Infinity` for very negative ones
- the shifted form can do neither: the biggest exponent is exp(0) = 1

And the numerically-stable log-softmax used by every real cross-entropy implementation:

```
log softmax(x)ᵢ = xᵢ − LSE(x)
```

No framework computes `log(softmax(x))` as two steps — the fused, shifted version is why your PyTorch loss doesn't NaN. When you write research code with novel probability math (new attention variants, custom losses, sampling schemes), *you* are the framework. LSE is the tool you'll reach for weekly.

---

## Precision zoo: float32, float16, bf16, and mixed precision

Training is memory- and bandwidth-hungry, so the field uses smaller floats:

```
format   bits  exponent  mantissa   range        precision
float32   32      8         23      ~1e±38       ~7 decimal digits
float16   16      5         10      ~6e-5..65504 ~3 digits   ← tiny RANGE
bfloat16  16      8          7      ~1e±38       ~2-3 digits ← float32 range!
```

The two 16-bit formats make opposite bets:
- **float16** keeps precision, sacrifices range → activations/gradients overflow past 65504 or underflow below 6e-5. Usable, but needs "loss scaling" machinery to keep gradients in the window.
- **bfloat16** keeps float32's *range* (same 8 exponent bits), sacrifices precision → almost nothing overflows; you just get coarser numbers. This trade wins for deep learning — modern LLM training is overwhelmingly bf16.

**Mixed precision**, the standard recipe:

```
matmuls & activations:    16-bit   (fast, small — where 95%+ of compute is)
master weights & optimizer: float32 (accurate accumulation)
loss & softmax/LSE:         float32 (the overflow-prone spots)
```

Why keep float32 weights? A weight update is `w ← w − η·g` where `η·g` can be ~1e-7 of w's size. In a ~3-digit 16-bit format, `w + tiny === w` — the update **rounds to nothing** and learning silently stops. (Same "big + small = big" failure from earlier — recognize it once, see it everywhere.) So updates accumulate in float32, and a 16-bit copy is cast for the fast math.

JS gives us float64 only, but the code simulates float16 rounding and shows a simulated 16-bit weight update dying.

---

## Vectorization: why GPUs exist

The performance half of numerical computing, in three ideas:

**1. The same math has wildly different speeds.** A triple-loop matmul, looping order `i,j,k` vs `i,k,j`, differs by 2–10× — identical arithmetic, different **memory access pattern**. Modern CPUs fetch memory in cache lines; walking a matrix row-wise (contiguous in memory) is fast, column-wise (strided) is slow. Allocating fresh arrays inside hot loops is worse.

**2. SIMD/vectorization: one instruction, many numbers.** CPUs apply one operation to 4–16 numbers simultaneously — *if* the data sits contiguously and the code exposes the pattern. This is why NumPy's `a + b` demolishes a Python for-loop, and why "vectorize your code" is performance advice everywhere.

**3. GPUs are this idea at scale.** A CPU: ~16 sophisticated cores for branching logic. A GPU: ~17,000 simple cores that all do the *same operation on different data* in lockstep. Neural nets are gigantic piles of exactly that (matmuls), which is why GPUs are 100× faster at them:

```
CPU:  [complex core] ×16      "do 16 different complicated things"
GPU:  [simple core] ×17000    "do ONE thing to 17000 numbers at once"
```

Research relevance: compute is the budget everything else trades against (recall the scaling-laws story from [Lesson 00](../00-what-ai-research-is/lesson.md)). An idea that's 2× slower per step must be *more* than 2× better per step. FlashAttention — one of the most-cited papers of the 2020s — changed **zero** math; it reordered attention's memory access (using, yes, an online log-sum-exp). Systems research in one sentence.

---

## The researcher's numerical checklist

```
□ loss NaN?  binary-search WHERE: first NaN activation/gradient, which step
□ any exp()? subtract the max (softmax/LSE) or bound the argument
□ any log()? guarantee argument > 0; prefer fused forms (log-softmax, log1p)
□ x − y with x ≈ y?  cancellation risk — algebraically rearrange if possible
□ comparing floats?  tolerance, never ===
□ float16 anywhere?  check range (65504!); prefer bf16; keep fp32 master weights
□ summing many terms? accumulate in higher precision (or sort small→large)
□ slow loop? preallocate outside, walk memory contiguously, batch into matmuls
```

---

## Code for this lesson

See [index.ts](index.ts) — demonstrates float weirdness (0.1+0.2, epsilon, 1e16+1), overflow/underflow into NaN, catastrophic cancellation's U-shaped error curve, naive-vs-stable softmax on large logits (NaN vs correct), log-sum-exp, a simulated float16 weight update that silently dies, and a timing race between allocate-per-iteration and preallocated loops.

## What's next
[Lesson 06 → The Optimization Landscape](../06-optimization-landscape/lesson.md)

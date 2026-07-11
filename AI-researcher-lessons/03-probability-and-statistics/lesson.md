# Lesson 03 — Probability and Statistics

---

## The problem probability solves

Ask an LLM to continue `"The capital of France is"`. It doesn't *know* the next word. It holds **beliefs**:

```
"Paris"   → 92.1%
"located" →  3.4%
"a"       →  1.1%
"the"     →  0.8%
...50,000 more options, each with a sliver of belief
```

Every LLM output, every training loss, every evaluation metric, every "is my experiment result real or luck?" question is probability. A researcher who can't read distributions can't read their own results. This lesson builds probability from zero — as **the mathematics of quantified uncertainty**.

---

## Probability as belief

A probability is a number between 0 and 1 measuring how strongly you believe something will happen (or is true):

```
P(coin lands heads)      = 0.5    total ignorance between two options
P(sun rises tomorrow)    ≈ 1.0    near certainty
P(next token is "Paris") = 0.921  strong but not total belief
```

Two rules generate everything else:

```
Rule 1 (normalization): beliefs over ALL possible outcomes sum to 1
Rule 2 (joint):         P(A and B) = P(A) · P(B | A)
```
- `P(A)`: probability of event A
- `P(B | A)`: probability of B **given that** A happened — belief *after* learning A. The `|` reads "given".

That's it. Bayes' rule, MLE, cross-entropy — all downstream of these two lines.

---

## Distributions: the shapes beliefs come in

A **distribution** assigns a probability to every possible outcome. Three shapes cover most of ML:

**Uniform** — no idea, all options equal.

```
die roll: P(1) = P(2) = ... = P(6) = 1/6

  1/6 ┤ ██  ██  ██  ██  ██  ██
      └──1───2───3───4───5───6──
```

**Categorical** — a labeled list of options, each with its own probability. **An LLM's output layer is exactly this**: softmax ([../../lessons/11-softmax/lesson.md](../../lessons/11-softmax/lesson.md)) turns raw scores into a categorical distribution over ~50,000 tokens.

```
P("Paris")=0.92, P("located")=0.03, ...   (sums to 1)
```

**Gaussian (normal)** — the bell curve, for continuous values that cluster around a center:

```
p(x) = (1 / √(2πσ²)) · exp(−(x−μ)² / (2σ²))
```
- `μ` (mu): the mean — where the bell's peak sits
- `σ` (sigma): the standard deviation — how wide the bell is
- `σ²`: the variance (spread, squared units)
- `exp(...)`: e to that power; the farther x is from μ, the smaller the belief
- the `1/√(2πσ²)` out front: just makes the total area equal 1 (Rule 1)

```
          ▲
          │        ▄█▄
          │      ▄█████▄
          │    ▄█████████▄
          │ ▄▄█████████████▄▄
          └───────┬───────────▶
                  μ
          ◀──σ──▶     68% of samples land within μ±σ
                      95% within μ±2σ
```

Why Gaussians are everywhere: neural net weights are initialized Gaussian, noise is modeled Gaussian, and — see CLT below — sums of random things *become* Gaussian whether you like it or not.

---

## Expectation and variance: center and spread

**Expectation** = the probability-weighted average — the long-run mean if you sampled forever:

```
E[X] = Σᵢ xᵢ · P(xᵢ)
```
- `X`: a random variable (a quantity whose value is uncertain)
- `xᵢ`: each possible value; `P(xᵢ)`: belief in that value

Die roll: `E[X] = 1·⅙ + 2·⅙ + ... + 6·⅙ = 3.5`. Note you can never roll 3.5 — expectation is a *center of mass*, not a prediction.

**Variance** = expected squared distance from the center — how spread out outcomes are:

```
Var[X] = E[(X − E[X])²]         std dev: σ = √Var[X]
```

Die: `Var = ((1−3.5)² + ... + (6−3.5)²)/6 ≈ 2.917`, so `σ ≈ 1.71`.

Researchers live on these two numbers: "mean accuracy 71.2 ± 1.4 over 5 seeds" is expectation ± a spread estimate. A paper reporting a mean without spread is hiding something (possibly from itself).

---

## Sampling: from belief to action

A distribution is beliefs; **sampling** draws a concrete outcome according to them. When an LLM "picks the next token," it samples from its categorical distribution (that's the temperature/top-p machinery of [../../lessons/23-sampling-strategies/lesson.md](../../lessons/23-sampling-strategies/lesson.md)).

Sampling a categorical with a single uniform random number — walk the cumulative probabilities until you pass r:

```
P = [0.5, 0.3, 0.2]   cumulative: [0.5, 0.8, 1.0]

r = 0.62  →  0.62 > 0.5, ≤ 0.8  →  outcome 1
r = 0.91  →  > 0.8               →  outcome 2
```

Sampling a Gaussian is less obvious — computers only give uniform randoms. The classic **Box–Muller transform** converts two uniforms into two perfect Gaussians:

```
z = √(−2·ln(u₁)) · cos(2π·u₂)
```
- `u₁, u₂`: independent uniform randoms in (0, 1)
- `z`: a standard Gaussian sample (μ=0, σ=1)
- for any other Gaussian: `x = μ + σ·z`

You'll implement this in the code and *verify* it statistically — sample 100,000 times, check that the measured mean ≈ μ, measured σ ≈ σ, and ~68% land within one σ. This habit — *verify your sampler empirically* — is real research hygiene; broken random number handling has silently ruined published experiments.

---

## Bayes' rule: updating beliefs on evidence

The question Bayes answers: *I believed X with some probability; I just saw evidence E; what should I believe now?*

```
P(H | E) = P(E | H) · P(H) / P(E)
```
- `P(H)`: **prior** — belief in hypothesis H before evidence
- `P(E | H)`: **likelihood** — if H were true, how probable is this evidence?
- `P(H | E)`: **posterior** — updated belief after seeing E
- `P(E)`: total probability of the evidence under all hypotheses (a normalizer)

**The medical test example** (famous because human intuition fails hard here):

A disease affects 1% of people. The test is good: 95% sensitive (catches sick people) with a 5% false-positive rate. You test **positive**. How likely are you actually sick?

Intuition screams "~95%." Let's count, using 10,000 people:

```
                      10,000 people
                     /            \
            sick: 100              healthy: 9,900
           /        \              /          \
   test + : 95   test − : 5   test + : 495   test − : 9,405
      ▲                          ▲
      └────── all positives ─────┘
              95 + 495 = 590 people test positive
              ...but only 95 are actually sick

P(sick | positive) = 95 / 590 ≈ 0.161  →  about 16%!
```

Same via the formula: `(0.95 × 0.01) / (0.95×0.01 + 0.05×0.99) = 0.0095/0.0590 ≈ 0.161`.

The false positives from the huge healthy population swamp the true positives from the tiny sick population. **The prior matters enormously.** Research translation: a shiny benchmark gain from a method that was a priori unlikely to work is, more often than not, a false positive (a bug, a leak, a lucky seed). Extraordinary claims need extraordinary evidence — that's Bayes, not a slogan.

---

## Maximum likelihood: cross-entropy loss IS MLE

**Maximum Likelihood Estimation (MLE)**: given data, choose the model parameters under which the observed data was *most probable*.

Tiny example — a bent coin, flipped 10 times: `H H T H H H T H H T` (7 heads, 3 tails). What's your best estimate of `p = P(heads)`?

Likelihood of the data as a function of p (flips independent, so probabilities multiply — Rule 2):

```
L(p) = p⁷ · (1−p)³

p = 0.5 → 0.5⁷·0.5³        ≈ 0.00098
p = 0.7 → 0.7⁷·0.3³        ≈ 0.00222   ← highest
p = 0.9 → 0.9⁷·0.1³        ≈ 0.00048
```

The maximum sits exactly at `p = 7/10 = 0.7` — the observed frequency. MLE formalizes "count and divide."

Two practical moves used everywhere:

1. Products of many small probabilities underflow to zero (Lesson 05!), so we maximize the **log**-likelihood instead — the log turns products into sums and doesn't move the maximum:
   `log L(p) = 7·log p + 3·log(1−p)`
2. Optimizers minimize, so we minimize the **negative** log-likelihood (NLL).

Now the punchline. Training an LLM: at each position the model outputs a categorical distribution; the data says which token actually came next; NLL of the data is

```
loss = −Σₜ log P_model(actual next token at t)
```

That is **exactly cross-entropy loss**. Cross-entropy isn't one clever loss among many — it *is* maximum likelihood applied to next-token prediction. "GPT training" = "MLE for a gigantic categorical model." Every symbol earns its place:
- `P_model(·)`: the probability the model assigned to the token that truly occurred
- `−log`: high assigned probability → small loss (−log 0.9 ≈ 0.105); low assigned probability → huge loss (−log 0.001 ≈ 6.9)
- `Σₜ`: summed over every position in the training text

Lesson 04 re-derives the same quantity from a totally different direction (information theory) — when two unrelated derivations land on the same formula, researchers pay attention.

---

## The Central Limit Theorem: why the bell is everywhere

> **CLT:** add up (or average) many independent random things, and the total's distribution approaches a Gaussian — *almost regardless of the shape you started with*.

One die: flat. Sum of two: a triangle. Average of 20: bell.

```
1 die            avg of 2            avg of 20
██████                █                  █
██████              █████              █████
██████            █████████          ███████
(flat)           (triangle)        (Gaussian!)
```

The code demonstrates this with actual dice. Precisely, for n independent samples with mean μ and standard deviation σ, the *average* is approximately Gaussian with:

```
mean μ,   standard deviation σ/√n
```

That `√n` is the researcher's most-used number:
- Why noisy gradients still work: a minibatch gradient is an *average* of per-example gradients → approximately Gaussian around the true gradient, with noise shrinking as √(batch size). Quadruple the batch to halve the noise.
- Why "run more seeds" has diminishing returns: to halve your error bar you need 4× the runs.
- Where error bars come from at all: `mean ± 2σ/√n` is a ~95% confidence interval, courtesy of the CLT.

---

## Why an LLM is "just" a categorical distribution

Assembling the whole lesson:

```
An LLM is a machine that maps
    (tokens so far)  →  categorical distribution over the next token

trained by  MLE  (= minimizing cross-entropy of the real next tokens),
sampled from at generation time (temperature reshapes the categorical),
evaluated by how much belief it puts on held-out text (perplexity, next lesson).
```

Nothing in the pipeline is anything *other* than probability. The word "just" wears quotes because predicting the next token extremely well turns out to require modeling grammar, facts, style, and arguably reasoning — one of the deepest surprises in the field, and the subject of active research on *what* these distributions internally represent.

---

## Symbol cheat sheet

```
P(A)           probability (belief) of A
P(B | A)       probability of B given A ("|" = "given")
X ~ D          "X is distributed as D" (e.g. x ~ N(0,1): standard Gaussian)
N(μ, σ²)       Gaussian with mean μ, variance σ²
E[X]           expectation (probability-weighted mean)
Var[X], σ²     variance;  σ = std deviation
i.i.d.         independent, identically distributed (papers say this constantly)
argmax_p L(p)  "the p that maximizes L"
log-likelihood log of the data's probability under the model
```

---

## Code for this lesson

See [index.ts](index.ts) — samples uniforms, categoricals, and Gaussians (Box–Muller) and verifies mean/variance empirically; demonstrates the CLT by averaging dice rolls into a text histogram; works the Bayes medical-test numbers; and fits a coin's bias by scanning the log-likelihood to find the MLE.

## What's next
[Lesson 04 → Information Theory](../04-information-theory/lesson.md)

# Lesson 20 — Statistical Significance

---

## The problem: "our method gets 84.2%, the baseline gets 83.5%"

+0.7 points! Ship the paper?

Rerun both with a different random seed:

```
                 seed 0    seed 1
our method       84.2      83.1
baseline         83.5      84.0
```

Now the baseline wins. Nothing changed except the seed — different weight init,
different data shuffling. The +0.7 you were about to publish is smaller than the
random wobble of the measurement itself.

**"One seed beat the baseline" is the single most common fake result in ML.**
This lesson gives you the from-scratch tools to tell wobble from signal.

---

## Intuition: seeds are repeated measurements

A chemist never reports one titration; they repeat the measurement and report
mean ± uncertainty. In ML, rerunning with a new seed IS the repeated measurement:
same method, same data, new roll of the random dice (init, shuffling, dropout masks).

```
method A, 5 seeds:  81.2  82.0  80.8  81.5  81.0
method B, 5 seeds:  82.1  82.9  81.7  82.4  81.9
```

Each method is now a *distribution*, not a number. The question "is B better than A?"
becomes: **are these two distributions genuinely apart, or could their difference be
produced by the wobble alone?**

---

## Step 1: mean and standard deviation

For method A's scores `x = [81.2, 82.0, 80.8, 81.5, 81.0]`, n = 5:

```
mean:      x̄ = (81.2+82.0+80.8+81.5+81.0)/5 = 81.30

deviations from mean:  -0.10, +0.70, -0.50, +0.20, -0.30
squared:                0.01, 0.49, 0.25, 0.04, 0.09     sum = 0.88

sample variance:   s² = 0.88 / (n−1) = 0.88 / 4 = 0.22
sample std dev:    s  = √0.22 ≈ 0.469
```

Every symbol:
- `x̄` — sample mean (our best guess of the "true" score)
- `s²` — sample variance; dividing by `n−1` (not `n`) corrects for the fact that we
  measured deviations from the *sample* mean, which is itself fit to the data
- `s` — standard deviation: the typical seed-to-seed wobble of ONE run

---

## Step 2: standard error — the wobble of the MEAN

`s` tells you how much a single run wobbles. But we report the *mean* of 5 runs, and
means are steadier than single runs. The **standard error** quantifies that:

```
SE = s / √n = 0.469 / √5 ≈ 0.21
```

Averaging n runs shrinks the uncertainty by √n. (Four times the seeds → half the error
bar. This is also why tiny improvements need many seeds to verify.)

Rough 95% confidence interval for the true mean:

```
x̄ ± ~2·SE  →  81.30 ± 0.42  →  [80.88, 81.72]
```

Meaning: if the "true" average score of method A is some fixed number, intervals built
this way capture it about 95% of the time. It is a statement about the *procedure*, not
a 95% probability that the truth is in this particular interval — but as a working
error bar, `±2·SE` is the right reflex.

---

## Step 3: Welch's t-test, by hand

Now compare A and B properly.

```
A: mean 81.30, s²=0.22, n=5
B: mean 82.20, s²=0.22, n=5      (B's numbers: 82.1 82.9 81.7 82.4 81.9)
```

**The t statistic** — how many "standard errors of the difference" apart are the means?

```
        x̄_B − x̄_A                 82.20 − 81.30
t = ───────────────────── = ───────────────────────────
    √(s²_A/n_A + s²_B/n_B)     √(0.22/5 + 0.22/5)

  = 0.90 / √0.088 = 0.90 / 0.297 ≈ 3.03
```

Every symbol:
- numerator — the observed gap between means (0.90 points)
- denominator — the expected wobble *of that gap* if you reran everything
  (each mean contributes s²/n of variance; independent variances add)
- `t` — gap measured in wobble units. |t| ≲ 1: gap is the size of the noise.
  |t| ≳ 3: gap is three times the noise — unlikely to be luck.

**Degrees of freedom** (Welch–Satterthwaite — how much data the wobble estimate rests on):

```
        (s²_A/n_A + s²_B/n_B)²
df = ────────────────────────────────────────── = 8  (here, since s², n equal)
     (s²_A/n_A)²/(n_A−1) + (s²_B/n_B)²/(n_B−1)
```

Welch's version (unequal variances allowed) is the one to use by default; the classic
"Student" t-test assumes both methods have the same variance, which you can't know.

**The p-value**: with t = 3.03 and df = 8, look up (or integrate — see index.ts) the
t-distribution: `p ≈ 0.016`.

---

## What a p-value is — and is NOT

`p = 0.016` means exactly this:

> IF the two methods were truly identical (the "null hypothesis"), THEN the probability
> of seeing a gap at least this large (in either direction) by seed luck alone is 1.6%.

It is **not**:
- ~~"There is a 98.4% chance B is better."~~ (p says nothing about that probability.)
- ~~"The effect is large."~~ (p measures surprise, not size — see below.)
- ~~"The result will replicate."~~ (A p=0.04 result is still wrong quite often.)

Convention calls p < 0.05 "statistically significant". That threshold is arbitrary —
a sociological custom, not a law of nature. Treat p as one input to judgment.

---

## Effect size vs significance

Two different questions, both required:

```
significance:  is the gap real?          (t-test, p-value)
effect size:   is the gap big enough to care?   (the gap itself, vs s)
```

With enough seeds, a +0.01-point improvement becomes "significant" (SE shrinks as
1/√n, so ANY real nonzero gap eventually clears the noise). Significant ≠ important.
Report the gap with its error bar — "+0.90 ± 0.30 points over 5 seeds" — and let the
reader judge whether 0.9 points matters. A useful scale-free companion is
`gap / s` (Cohen's d): here 0.90/0.47 ≈ 1.9 standard deviations — a big effect.

---

## The multiple comparisons trap

The most seductive trap in experimental ML:

> You try 20 variants of your idea. Nineteen do nothing. One shows p = 0.03.
> You write the paper about that one.

If a variant with NO real effect has a 5% chance of hitting p < 0.05 by luck, then with
20 truly-useless variants:

```
P(at least one "significant") = 1 − 0.95²⁰ ≈ 64%
```

Nearly two-thirds of the time, pure noise hands you a "discovery". This is why
pre-registration ([Lesson 17](../17-scientific-method/lesson.md)) exists, and it's the
statistical shadow of hyperparameter search ([Lesson 19](../19-hyperparameter-search/lesson.md)):
every extra thing you try is another lottery ticket against the validation set.

Defenses:
- **Confirm on fresh randomness**: rerun the winning variant with new seeds. A lucky
  winner regresses to nothing; a real one survives. (Cheap and brutal — do this always.)
- **Bonferroni correction**: testing k hypotheses? Demand p < 0.05/k. (For k=20: 0.0025.)
- **Report the denominator**: "we tried 20 variants" belongs in the paper.

---

## The checklist

```
1. ≥ 3 seeds (5 is a good default) for every number you compare.
2. Report mean ± std (or ± SE — say which!) and n. Never a lone number.
3. Welch's t-test for the headline comparison; p < 0.05 is a convention, not proof.
4. Report effect size with error bars, not just p.
5. Tried k variants? Say so, and either correct for it or re-verify the winner
   on fresh seeds before believing it.
```

---

## Code for this lesson

See [index.ts](index.ts) — simulates two methods whose TRUE means differ by a small
margin, runs 5 seeds each, computes mean/std/standard error/Welch's t and the p-value
from scratch (numerically integrating the t-distribution), and prints the verdict.
Then it springs the trap: 20 variants with NO true difference, and at least one
"wins significantly" by pure luck.

Run it:

```bash
npx ts-node index.ts
```

## What's next

Statistics tells you when a result is real. But most of the time the result is: nothing
worked. That's not failure — that's the job.
[Lesson 21 → Negative Results & Research Debugging](../21-negative-results/lesson.md)

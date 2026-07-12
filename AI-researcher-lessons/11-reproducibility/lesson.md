# Lesson 11 — Reproducibility

---

## The problem this lesson solves

A story that happens in every lab, constantly:

> Monday: you tweak the model, run training, loss drops from 2.31 to 2.19.
> "It works! +0.12 improvement!" You tell your advisor.
> Tuesday: you run *the exact same code* to make a plot... loss: 2.29.
> Wednesday: 2.24. Thursday: 2.35 — worse than the baseline.

Nothing changed. So what did Monday's number *mean*?

This is not a hypothetical embarrassment — it's the default state of ML research.
Training is randomized, so **every result is a draw from a distribution**, and a
single run tells you almost nothing about that distribution. The field learned
this the hard way: whole families of published "improvements" later turned out
to be seed luck. Independent re-evaluations have repeatedly found that
once-celebrated gains shrink or vanish when baselines are tuned equally hard and
results are averaged over multiple runs — this is ML's version of science's
**reproducibility crisis**.

The skills in this lesson — controlling randomness, logging everything,
distinguishing "it worked once" from "it works" — are what separate a
researcher from someone who trains models and collects anecdotes.

---

## The analogy: the baking competition

Two bakers claim their new oven bakes better bread.

- Baker A baked **one loaf** in the new oven. It was great. Claim made.
- Baker B knows loaves vary — humidity, yeast mood, flour batch. She bakes
  **five loaves in each oven**, writes down every ingredient and temperature,
  and compares the *averages* (and how much loaves vary around them).

Baker A has a story. Baker B has evidence. Same ovens, same skill — the entire
difference is procedure. This lesson is Baker B's procedure, for ML.

---

## Step 1: know where the randomness hides

"But my code is deterministic!" It isn't. Randomness enters training in at least
four doors:

```
1. WEIGHT INITIALIZATION   every run starts from a different random point
                           on the loss landscape (Lesson 06!) — different
                           basin, different final model

2. DATA SHUFFLING          batch order changes which gradients arrive first;
                           early batches steer where the model goes

3. DROPOUT & SAMPLING      dropout masks (Lesson 07), sampled generations
                           (../../lessons/23-sampling-strategies/lesson.md)

4. THE HARDWARE ITSELF     GPU ops like parallel additions run in
                           nondeterministic ORDER, and floating-point
                           addition is not associative:
                           (a + b) + c ≠ a + (b + c) in float32
                           → bit-level differences even with all seeds fixed!
```

Door 4 deserves a pause: you proved in
[Lesson 05 — Numerical Computing](../05-numerical-computing/lesson.md) that
float addition order changes results. A GPU sums thousands of products in
whatever order its threads finish. Same seed, same code, same machine —
different bits. Those tiny differences then get amplified by training chaos
(a slightly different gradient → slightly different weights → different batch
behavior...). Frameworks offer "deterministic modes" that force fixed reduction
orders, at a speed cost.

---

## Step 2: seeds — making randomness repeatable

Computers don't produce true randomness; they produce **pseudo**-random
sequences: a deterministic formula that scrambles a state so thoroughly the
output *looks* random. The starting state is the **seed**.

```
seed 42 → PRNG → 0.372, 0.891, 0.104, 0.556, ...   ← ALWAYS this sequence
seed 42 → PRNG → 0.372, 0.891, 0.104, 0.556, ...   ← run it again: identical
seed 43 → PRNG → 0.719, 0.028, 0.443, ...          ← different universe
```

The code for this lesson implements a real PRNG — **mulberry32**, small enough
to read in full:

```
state = (state + 0x6D2B79F5) mod 2³²        ← march the state forward
t = scramble(state)                          ← bit-mixing: XORs, shifts,
return t / 2³²                               ← multiplies... then map to [0, 1)
```

Every symbol: `state` is a 32-bit integer (the PRNG's entire memory);
`0x6D2B79F5` is a carefully chosen odd constant; the scramble makes consecutive
outputs statistically uncorrelated; dividing by 2³² maps to `[0, 1)`.
JavaScript's `Math.random()` **cannot be seeded** — which is exactly why a
researcher writing TypeScript needs their own.

Fix the seed → *identical* training run, loss curve, final weights (the code
proves this to every decimal place). Now "same code, same result" is restored —
and one seed IS one loaf of bread. Which brings us to the real discipline:

**Seeds are for repeatability, not for results.** Reporting your best seed is
p-hacking with extra steps. The honest use of seeds:

```
"it worked once"                        "it works"
─────────────────                       ──────────
run with seed 42                        run with seeds 1..5
loss = 2.19                             loss = 2.24 ± 0.06  (mean ± std)
publish 2.19                            baseline = 2.31 ± 0.02
                                        gap (0.07) vs noise (±0.06): marginal!
                                        → run more seeds or temper the claim
```

The mean tells you the typical outcome; the standard deviation tells you how
big seed luck is. **If your improvement is smaller than the seed-to-seed std,
you haven't measured an improvement — you've measured a die roll.**
([Lesson 20 — Statistical Significance](../20-statistical-significance/lesson.md)
turns this instinct into proper tests.)

---

## Step 3: log everything — the experiment tracker

Three weeks from now you will stare at a loss curve named `run_final_v2_REAL`
and remember nothing. The fix is brutally simple: **every run writes down
everything needed to recreate it**, automatically, at launch time:

```
what to log, per run:
  config     : every hyperparameter (lr, batch size, model size, λ, ...)
  seed       : the seed(s)
  code state : git commit hash — WHICH CODE produced this number?
               (+ a dirty-flag if uncommitted changes exist)
  data state : dataset name/version/hash
  environment: library versions, hardware
  results    : metrics over time, wall-clock, final metrics
```

The standard format is **JSON Lines**: one JSON object per line, appended to a
file. Trivial to write, trivial to grep, trivial to load into any analysis
tool — this lesson's code implements a complete tracker in ~20 lines and uses
it for a real 5-seed study:

```
{"runId":"a3f2","gitHash":"7b09e60","seed":1,"config":{"lr":0.05,...},"finalLoss":0.0243}
{"runId":"9c81","gitHash":"7b09e60","seed":2,"config":{"lr":0.05,...},"finalLoss":0.0251}
```

Professional tools (Weights & Biases, MLflow, TensorBoard) are this same idea
with dashboards on top. The tool doesn't matter. The habit does: **a result you
can't trace to code + config + seed is a rumor, not a result.**

### Why researchers keep lab notebooks

The log captures what the machine knew. The notebook captures what *you* knew:

```
2026-07-12  Hypothesis: warmup matters less at small scale.
            Ran lr=0.05 no-warmup, seeds 1-5 → 2.24 ± 0.06 (runs.jsonl #12-16)
            vs warmup baseline 2.25 ± 0.05. No effect at this scale. NOT
            what the blog posts claim. Try 10× model tomorrow?
```

Date, hypothesis, what you ran, what happened, what you concluded, what's next.
Five lines a day. Its value compounds: failed experiments (which you will
otherwise re-run in three months), half-formed hunches, the *reason* you tried
things. Every serious researcher keeps one; papers are basically lab notebooks
distilled ([Lesson 28 — Writing a Paper](../28-writing-a-paper/lesson.md)).

---

## Step 4: the reproducibility ladder

Not all "reproducible" is equal. The ladder, bottom to top:

```
level 0  it ran once on my machine            ← an anecdote
level 1  same seed, same machine → same number    (this lesson's code)
level 2  multiple seeds → stable mean ± std       (the claim survives seed luck)
level 3  someone ELSE reruns your code+data → same conclusion
level 4  someone reimplements FROM YOUR PAPER alone → same conclusion
                                              ← science
```

Level 4 is the gold standard and the acid test of your writing: if a competent
reader can't rebuild your result from the paper's description, the paper is
incomplete no matter how good the result is. (You'll attempt exactly this climb
in [Lesson 15 — Reproducing a Paper](../15-reproducing-a-paper/lesson.md) —
and you'll discover how many published papers fail their own test.)

A checklist you can start using today:

- [ ] seeds controlled and **reported** (all of them, not the best one)
- [ ] results as mean ± std over ≥3 seeds, never a single run
- [ ] every run logged: config + seed + git hash + metrics
- [ ] committed code before big runs (a hash pointing at dirty state is a lie)
- [ ] lab notebook entry: hypothesis → result → conclusion
- [ ] baseline tuned as hard as your method
      ([Lesson 18 — Baselines & Ablations](../18-baselines-and-ablations/lesson.md))

---

## Code for this lesson

See [index.ts](index.ts) — a complete miniature of the discipline:

1. **mulberry32**, a seedable PRNG, implemented and demonstrated,
2. **same seed → bit-identical training runs** of a tiny model (and different
   seeds → different results), shown side by side,
3. a **mini experiment tracker** that appends `{runId, gitHash, seed, config,
   finalLoss}` JSON lines to `runs.jsonl` in this folder, runs the experiment
   with 5 seeds, and reports **mean ± std** — the difference between "it worked
   once" and "it works".

Run it:
```
npx ts-node index.ts
```

(then look at the `runs.jsonl` it wrote — that file is the habit.)

## What's next

Phase B is complete: you know why training works, why models generalize, why
transformers won, how scale is paid for, how to evaluate, and how to make
results real. Time to enter the literature — starting with the researcher's
core skill of reading a paper without drowning.
[Lesson 12 → How to Read a Paper](../12-how-to-read-a-paper/lesson.md)

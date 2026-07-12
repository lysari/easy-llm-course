# Lesson 19 — Hyperparameter Search

---

## The problem: your method has knobs, and the knobs decide everything

You implemented a model. Before training even starts, you must choose:
learning rate, weight decay, batch size, hidden size, warmup steps, dropout…

These choices routinely matter *more* than the architecture ideas people write papers
about. A transformer with a bad learning rate loses to a bag of bigrams with a good one
(you saw a mini version of this in [Lesson 17](../17-scientific-method/lesson.md)).

So: how do you find good knob settings without burning your whole compute budget —
and how do you report the search honestly?

---

## Hyperparameters vs parameters

Two words that sound alike and mean opposite things:

| | Parameters | Hyperparameters |
|---|---|---|
| What | weights, biases | lr, weight decay, hidden size, … |
| How many | millions/billions | a handful (5–20) |
| Set by | gradient descent, automatically | **you**, before training |
| Cost to evaluate one setting | one forward/backward pass | **one entire training run** |

That last row is the whole difficulty. Testing a parameter value is free; testing a
hyperparameter value costs a full training run. With a budget of, say, 25 runs, every
run must count.

(Why can't gradient descent set the learning rate itself? Because the lr is part of the
*rule* for updating, not part of the function being optimized — there's no useful
gradient signal flowing to it. See [../../lessons/17-adam-optimizer/lesson.md](../../lessons/17-adam-optimizer/lesson.md)
for how Adam *adapts* step sizes but still needs a base lr from you.)

---

## Grid search and its curse

The obvious plan: pick a few values per knob, try all combinations.

```
lr   ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1}      (5 values)
wd   ∈ {1e-6, 1e-5, 1e-4, 1e-3, 1e-2}  (5 values)

grid = 5 × 5 = 25 runs
```

Fine for 2 knobs. But the grid grows **exponentially** with the number of knobs:

```
2 knobs, 5 values each:   25 runs
4 knobs:                 625 runs
6 knobs:              15,625 runs
8 knobs:             390,625 runs   ← more runs than you will do in your career
```

This is the **curse of dimensionality**. With any realistic budget you can afford only
2–3 values per knob — a grid so coarse it straddles the good region without landing in it.

---

## Why random search beats grid (Bergstra & Bengio, 2012)

Here is the surprisingly deep argument, in one picture. Suppose only ONE of your two
knobs actually matters (very common — think lr matters, wd barely does):

```
        GRID (9 runs)                    RANDOM (9 runs)

  wd ↑  ●     ●     ●             wd ↑      ●
        |     |     |                  ●          ●
        ●     ●     ●                       ●   ●
        |     |     |                  ●        ●
        ●     ●     ●                     ●       ●
        └──────────────→ lr           └──────────────→ lr

  project onto the lr axis        project onto the lr axis
  (the axis that matters):        (the axis that matters):

        ●     ●     ●             ●  ● ●  ●● ●  ● ● ●
   only 3 DISTINCT lr values!     9 distinct lr values!
```

The grid spends 9 runs but tests only **3 distinct learning rates** — each lr is
wastefully repeated at 3 wd values that don't change anything. Random search tests
**9 distinct learning rates** for the same price.

General statement: if only `k` of your `d` knobs matter, an `n`-point grid explores
about `n^(k/d)` useful settings, while random search explores about `n`. Since you never
know in advance *which* knobs matter, random search is the safer default. It is also
embarrassingly parallel and you can stop it at any time — a half-finished grid is
biased, a half-finished random search is just a smaller random search.

---

## Log-uniform sampling: how to randomize a learning rate

Sampling lr uniformly from [0.0001, 1] is a bug: 99.99% of your samples land above 0.001,
and the region 0.0001–0.001 (often where the answer lives) gets almost nothing.

Learning rates live on a **multiplicative** scale — the interesting difference is between
1e-4 and 1e-3 (a 10× jump), not between 0.90 and 0.91. So sample the *exponent* uniformly:

```
u  ~ uniform(-4, 0)      ← pick the exponent
lr = 10^u                ← 10^-4 … 10^0

Result: each decade [1e-4,1e-3], [1e-3,1e-2], [1e-2,1e-1], [1e-1,1]
        gets exactly 25% of the samples.
```

Same trick for weight decay, and any knob whose plausible range spans orders of
magnitude. Knobs with narrow linear ranges (dropout 0.0–0.5) can stay uniform.

---

## Successive halving: stop wasting compute on losers

A full training run per configuration is honest but wasteful: after 10% of training you
can usually already see that lr=1.0 is diverging. **Successive halving** exploits this:

```
Budget = 100 epochs total.

Round 1: 24 configs × 1 epoch each     = 24 epochs   → keep best 8
Round 2:  8 configs × 3 more epochs    = 24 epochs   → keep best 3
Round 3:  3 configs × 8 more epochs    = 24 epochs   → keep best 1
Round 4:  1 config  × 28 more epochs   = 28 epochs   → final answer
                                        -----------
                                        100 epochs, but 24 configs explored
```

Compare: naive random search with the same 100 epochs at 4 epochs each explores 25
configs *shallowly*; plain full-length runs explore only ~3 configs *deeply*. Successive
halving gets breadth AND depth: many candidates screened cheaply, survivors trained long.

The risk: a "late bloomer" config that starts slow but would win at full length gets cut
in round 1. In practice early performance correlates well enough with final performance
that the trade is worth it (Hyperband is this idea with a hedge across cut schedules).

---

## Budget honesty: search cost is part of your method's cost

The quiet scandal of many comparisons:

```
"Our method:  87.1%     Baseline: 85.0%"

untold story:
  our method   →  200 runs of hyperparameter search, best one reported
  baseline     →  1 run with defaults
```

That's not a method comparison; it's a *budget* comparison — 200 lottery tickets vs 1.
More search also means more chances to overfit the validation set by luck (the multiple
comparisons trap — [Lesson 20](../20-statistical-significance/lesson.md)).

Honest reporting:

```
1. State the search space and the number of trials FOR EVERY method in the table.
2. Give the baseline the same trial budget as your method.
3. Report the search cost: "each result = best of 25 trials of random search
   over lr ∈ 10^[-4,0], wd ∈ 10^[-6,-2]".
4. Never report the best seed × best hyperparameters combo as "the" number.
```

Rule of thumb: **if the tuning budgets differ, the comparison is invalid** — you learned
which method got more compute, not which method is better.

---

## Practical defaults

```
• 2–3 knobs you believe matter → random search, log-uniform, 20–50 trials.
• Expensive runs → successive halving on top of random sampling.
• Grid is fine for ONE knob (a 1-D grid ≡ evenly-spaced search, nothing lost).
• Always log every trial (config → score), not just the winner: the losers
  are your sensitivity analysis for free.
• Tune on validation data, report on test data, exactly once. (Lesson 20.)
```

---

## Code for this lesson

See [index.ts](index.ts) — a synthetic noisy objective `f(lr, wd)` with a known optimum,
where (as usual in real life) lr matters much more than wd. Under an equal budget of 100
"epochs" we compare: grid search (25 points × 4 epochs), random search (25 samples × 4
epochs, log-uniform), and random + successive halving (24 configs, winners trained
longer). The output shows the grid's projection problem numerically: it only ever tries
5 distinct learning rates.

Run it:

```bash
npx ts-node index.ts
```

## What's next

Your search found a winner that beats the baseline by 0.8 points. Is that real, or is it
seed luck? Time for error bars and t-tests.
[Lesson 20 → Statistical Significance](../20-statistical-significance/lesson.md)

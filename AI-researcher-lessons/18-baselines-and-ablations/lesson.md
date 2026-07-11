# Lesson 18 — Baselines & Ablations

---

## The problem: "my model gets 87% accuracy"

Is 87% good?

You genuinely cannot answer. Not "it's hard to answer" — it is *undefined*. Consider:

- If the dataset is 90% one class, a rock that always predicts that class gets 90%.
  Your 87% model is **worse than a rock**.
- If the previous best method gets 70%, your 87% is a breakthrough.
- If a 10-line heuristic gets 86.5%, your six months of work bought half a point.

A number in isolation is not evidence. **Every claim is a comparison**, and the thing
you compare against is called a **baseline**. No baseline, no claim.

---

## Intuition: the "compared to what?" reflex

A friend says: "This stock picker is amazing — it made 8% last year!"

Your reflex should be: *compared to what?* If the whole market went up 20% that year,
the "amazing" picker underperformed a zero-effort index fund by 12 points.

The index fund is the baseline. It's cheap, dumb, and it calibrates every claim.
ML reviewers have the same reflex, and the first question on every paper is:
**"what does the trivial method get?"**

---

## The ladder of baselines

Baselines come in grades, from free to expensive. A strong paper climbs the whole ladder:

```
5. YOUR method minus the new part   ← the ablation (see below)
4. Prior state of the art           ← the published best number
3. Simple heuristic                 ← 10 lines of code, no learning
2. Majority class                   ← always predict the most common label
1. Random guess                     ← the floor
```

With tiny concrete numbers — say a spam classifier on a dataset that is 70% not-spam:

| Baseline | How it works | Accuracy |
|---|---|---|
| Random guess | flip a fair coin | 50% |
| Majority class | always say "not spam" | 70% |
| Heuristic | "contains '$$$' or 'free' → spam" | 78% |
| Prior SOTA | last year's published model | 84% |
| Your full model | the thing you built | 87% |

Now 87% *means something*: +3 over SOTA, +9 over a heuristic. And notice what each rung
tells you:

- **Random (50%)** — the floor. If you're near it, something is broken (Lesson 21).
- **Majority (70%)** — what "no intelligence at all" achieves. Any model below this is
  actively harmful.
- **Heuristic (78%)** — how much of the task is *easy*. The gap between 70 and 78 is
  solvable with an `if` statement; only the gap above 78 needed learning.
- **SOTA (84%)** — the actual bar for a research claim.

---

## The ablation: baseline #5, the heart of a paper

Suppose your method has three components: a fancy attention variant, a data augmentation
trick, and a new loss term. You report 87%. A reader asks:

> "Which of the three parts actually matters?"

An **ablation study** answers this by removing one component at a time and re-measuring
("ablate" = surgically remove):

```
Full model (A + B + C)          87.0%
  − A (fancy attention)         81.0%   ← removing A costs 6.0 → A matters a lot
  − B (augmentation)            86.8%   ← removing B costs 0.2 → B does ~nothing!
  − C (new loss)                84.5%   ← removing C costs 2.5 → C helps
Majority class                  70.0%
```

This table is often the single most informative thing in a paper. It tells the reader:

1. **What to actually use.** If B costs 0.2 points but doubles training time, drop it.
2. **What the contribution really is.** Maybe your paper's *title* is about B — the
   ablation just demoted your headline idea to a footnote. Painful, but that's the job.
3. **That you did the work.** A paper with no ablation is asking readers to take a
   bundle of changes on faith — exactly the confound problem from
   [Lesson 17](../17-scientific-method/lesson.md), at paper scale.

Note the structure: the ablation is the control-everything-but-one rule applied to your
own architecture. Each row changes exactly one thing relative to the full model.

---

## How weak baselines create fake progress

The dark side. A field can "advance" for years on paper while standing still, because of
one systematic error: **the new method is tuned hard, the baseline is not.**

How it happens (usually innocently):

1. You spend 3 weeks tuning your method: lr sweeps, schedules, tricks. It reaches 87%.
2. You need a baseline. You grab an off-the-shelf implementation, run it once with
   default settings. It gets 82%.
3. You publish "+5 points over the baseline."
4. Someone later tunes the baseline with the same 3-week effort. It gets 86.5%.
   Your real contribution was +0.5.

This is so common that **"the baseline was not tuned"** is arguably the most frequent
substantive criticism in peer review. Multiple famous "comparison" studies have found
that once baselines get equal tuning budget, years of claimed improvements collapse —
in metric learning, in recommender systems, in GAN variants, the pattern repeats.

The honest rule:

> **Spend as much tuning effort on the baseline as on your method.**
> Budget for search is part of the comparison (more in
> [Lesson 19](../19-hyperparameter-search/lesson.md)).

If your method only wins when the baseline is handicapped, you don't have a method.
You have a tuning gap.

---

## Reading an ablation table like a reviewer

When you see this in a paper:

```
Full model            91.2
 − component X        91.0
 − component Y        90.9
 − component Z        84.1
```

a reviewer thinks:

- Z is the paper. X and Y are decoration (each worth ~0.2–0.3, likely within noise —
  Lesson 20 gives you the tools to check).
- If the abstract sells X, be suspicious.
- Where is "− X − Y" (remove both)? Components can overlap: each looks useless alone
  if the other covers for it.
- Where are the error bars? A 0.2-point gap with ±0.4 noise is a coin flip.

Train this reflex on your own tables before reviewers do.

---

## Checklist for your own experiments

```
1. Before building anything: compute random + majority-class numbers. (5 minutes.)
2. Write the dumbest heuristic you can. If it nearly matches your model, the task
   is easier than you thought — or your model is weaker.
3. Tune the baseline with the same effort/budget as your method. Log both budgets.
4. One ablation row per component. Change exactly one thing per row.
5. Report the whole ladder in one table. Never a lone number.
```

---

## Code for this lesson

See [index.ts](index.ts) — a tiny 3-component text classifier built from scratch on an
embedded toy dataset (character-bigram features + a length feature + a class-prior bias
correction). It prints a real ablation table: full model, each component removed, and the
majority-class baseline — and the table reveals that one of the three components barely
matters.

Run it:

```bash
npx ts-node index.ts
```

## What's next

Your ablation is only fair if every variant was tuned fairly — which means searching
hyperparameters honestly.
[Lesson 19 → Hyperparameter Search](../19-hyperparameter-search/lesson.md)

# Lesson 21 — Negative Results & Research Debugging

---

## The problem: your idea didn't work. Now what?

You had a hypothesis ([Lesson 17](../17-scientific-method/lesson.md)). You built it,
with baselines ([Lesson 18](../18-baselines-and-ablations/lesson.md)), tuned it fairly
([Lesson 19](../19-hyperparameter-search/lesson.md)), ran 5 seeds
([Lesson 20](../20-statistical-significance/lesson.md)).

Result: no improvement. Flat. Nothing.

First, the fact nobody puts on posters: **this is the normal outcome.** Ask working
researchers and they'll tell you the large majority of their ideas die — the papers you
read are the survivors of a brutal filter (survivorship bias in action). A researcher
who never gets negative results isn't brilliant; they're either not trying new things
or not measuring honestly.

So the real skill isn't avoiding failure. It's what you do in the hours after it:

```
"It didn't work"  is not a conclusion. It's a fork with three branches:

  1. My implementation is wrong.   (a bug — most likely!)
  2. My experiment is wrong.       (bad baseline, confound, noise)
  3. My idea is wrong.             (the actual scientific result)

You may only claim branch 3 after ruling out 1 and 2.
```

This lesson is about telling the branches apart.

---

## "Neural nets want to work" — why bugs hide

In normal software, bugs announce themselves: crash, exception, wrong output.

In ML, they don't. Karpathy's observation: **neural networks want to work.** They are
so flexible that they will learn *around* your bug and produce results that are merely
mediocre instead of obviously broken.

Real examples of this species of bug:

```
bug                                     what you see
------------------------------------   -----------------------------------
forgot to shuffle training data         trains! accuracy just 5 points low
lr 100× too small                       trains! just slowly and plateaus early
normalized by 255 twice                 trains! images are dim, acc a bit low
labels misaligned by one position       trains! loss even goes down a little
                                        (it learns the label PRIOR, ~chance acc)
eval computed on training set           trains! results amazing (too amazing)
softmax over the wrong axis             trains! attention is uniform mush
```

None of these crash. Every one of them silently costs you accuracy — or silently gives
you fake accuracy. That's the danger: a broken experiment *looks exactly like* a
mediocre idea. If you take "it didn't work" at face value, you may bury a good idea
because of a data-loader bug (or publish a bad one because of an eval bug).

---

## The loss-not-going-down checklist

When training fails, resist the urge to "try a different lr" at random. Walk the
pipeline in order — data first, model last, because data bugs are more common and
cheaper to check:

**1. DATA — look at it with your eyes**
```
• Print 5 raw (input, label) pairs. Are the labels actually right?
• Are inputs in a sane range (not 0–255 when you assumed 0–1)?
• Is the class balance what you think? (90/10 explains a "90% accuracy" model.)
• Shuffled? (Sorted-by-class data + SGD = each epoch forgets the previous class.)
```

**2. SHAPES — print them at every stage**
```
• (batch, features) where you expect it? Transposed matrices often still
  multiply "successfully" thanks to broadcasting — silently wrong.
• Off-by-one anywhere labels and inputs are zipped together?
```

**3. LEARNING RATE — the usual suspect**
```
• Loss NaN/exploding      → lr too big (try 10× smaller)
• Loss flat from step 0   → lr too small (try 10× bigger), or gradients
                            aren't flowing (check init, check backprop)
• Loss oscillating hard   → lr slightly too big
Sweep lr in decades {1e-4 … 1} before concluding anything. (Lesson 19.)
```

**4. INIT**
```
• All weights zero → symmetric neurons that stay identical forever (no learning).
• Weights too large → saturated sigmoids/softmax, gradient ~0. (../../lessons/05-activation-functions/lesson.md)
```

**5. LOSS FUNCTION**
```
• Is the loss at step 0 the theoretical "random" value? For 2 balanced classes,
  cross-entropy should start ≈ ln(2) ≈ 0.693; for 10 classes ≈ ln(10) ≈ 2.303.
  If it starts elsewhere, the wiring is wrong before training even begins.
• Softmax + cross-entropy applied to already-softmaxed outputs? (double softmax)
```

**6. EVAL — the bug that flatters you**
```
• Are you evaluating on data the model trained on? (Great numbers, fake.)
• Same preprocessing at train and eval time?
• Is your accuracy function itself correct? Test it on hand-made toy inputs.
```

The order matters: each step is cheaper and more likely to find the bug than the one
below it. Most "my model doesn't learn" mysteries die at steps 1–3.

---

## The overfit-one-batch test

The single most powerful diagnostic in deep learning, and it costs one minute:

> **Take ONE tiny batch (say 4 examples). Train on it, alone, for many steps.
> A correct implementation must reach ~zero loss — it can just memorize 4 examples.**

```
outcome                          meaning
------------------------------  ------------------------------------------
loss → ~0                        pipeline works. Your problem is capacity,
                                 data, or the idea itself — not a bug.
loss plateaus above ~0           IMPLEMENTATION BUG. A working net can
                                 always memorize 4 examples. Go hunting:
                                 data pairing, gradients, loss wiring.
loss explodes / NaN              lr or numerical bug (log(0), exp overflow —
                                 see ../../lessons/05-numerical-computing/... 
                                 lesson 05 of this track).
```

This test cleanly splits branch 1 from branch 3:

```
                    overfits one batch?
                     /              \
                   YES               NO
                    |                 |
        implementation is fine    implementation is broken
        the negative result       the "negative result"
        might be REAL             is MEANINGLESS — fix the
                                  bug and rerun
```

Never write "the idea didn't work" in your log until the one-batch test passes.

---

## When to abandon vs when to persist

No formula exists, but good researchers use rules of thumb:

**Persist when:**
- You haven't passed the one-batch test yet (you don't have a result at all — you have a bug).
- The idea "almost works": clear signal in some settings, unstable in others.
  Instability is often one hyperparameter or one missing normalization away from working.
- The failure taught you something specific to try next ("gradients vanish at layer 3" →
  try a residual connection — that observation is *progress*).

**Abandon (for now) when:**
- The honest, tuned, multi-seed comparison shows nothing, twice, in the setting
  you designed the idea for. (If it can't win on home turf, stop.)
- The idea only wins when the baseline is handicapped (Lesson 18's warning sign).
- Each rescue attempt adds complexity but no signal — you're doing epicycles.
- You notice you *want* it to work so badly that you're metric-shopping (Lesson 17).

Write the abandonment down in your experiment log with the evidence. "Tried X, clean
negative, see exp-041..exp-047" is a gift to your future self — and pruning the search
space is genuine scientific output, even if nobody hands out awards for it.

---

## Why negative results rarely get published — and what researchers do instead

The uncomfortable economics: venues want novelty and wins. "We tried X and it didn't
help" is nearly unpublishable at a main conference, even when X is something half the
field is quietly trying. Consequences:

- **Publication bias**: the literature over-represents successes; readers overestimate
  how well methods work (the same survivorship bias, field-wide).
- **Wasted duplication**: labs everywhere silently re-fail at the same ideas because
  nobody could tell them it doesn't work.
- **The file-drawer problem**: with enough labs trying a useless idea, ONE gets lucky
  (Lesson 20's multiple-comparisons trap at planetary scale) — and that one publishes.

What researchers actually do with negative results:

```
• Workshops        — venues like "I Can't Believe It's Not Better" exist
                     precisely for well-executed negative results.
• Blog posts       — fast, uncensored, often more read than the paper
                     would have been. Great early-career visibility.
• Paper appendices — "we also tried A, B, C; none helped" sections make
                     the main result more credible and save readers time.
• Ablations        — a negative result about a component IS an ablation
                     row (Lesson 18): publishable inside a positive paper.
• Lab notes/log    — minimum viable archive. Future-you always reads it.
```

A negative result that is *well-executed* (controlled, tuned, multi-seed, one-batch
test passed) is real knowledge. Store it somewhere it can compound.

---

## The mindset

```
Bugs first, ideas second:  assume implementation error until proven otherwise.
Mediocre ≠ disproven:      nets learn around bugs; flat results are ambiguous.
One-batch test:            the fastest wrong/right verdict on your pipeline.
Log the failures:          pruned branches are progress.
Abandon without shame:     the goal is true beliefs, not rescued projects.
```

---

## Code for this lesson

No `index.ts` this time — instead a drill. See [exercise.md](exercise.md): you'll run
the debugging checklist, then **intentionally plant three classic bugs** in a copy of a
working network and study the symptoms each one produces. Knowing the symptoms in a
controlled setting is what lets you recognize them in the wild.

## What's next

You can now produce trustworthy evidence. Time for the frontiers — starting with looking
*inside* the models you train.
[Lesson 22 → Interpretability](../22-interpretability/lesson.md)

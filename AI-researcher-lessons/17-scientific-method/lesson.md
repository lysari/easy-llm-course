# Lesson 17 — The Scientific Method for ML

---

## The problem: "I tried something and it got better"

Here is the most common sentence in ML experimentation:

> "I changed the learning rate and made the network bigger, and accuracy went up.
> So bigger networks are better!"

Stop. Two things changed. Accuracy went up. **Which change caused it?**

You cannot know. Maybe the bigger network helped. Maybe the new learning rate helped.
Maybe the bigger network actually *hurt* and the learning rate helped so much it hid the damage.
Maybe neither helped and you got a lucky random initialization.

Research is the craft of making claims you can defend. That requires running experiments
where the conclusion is *forced* by the evidence — not merely suggested by it.

---

## Intuition: the kitchen analogy

You bake a cake and it comes out dry. Next time you:
- lower the oven temperature, AND
- add an extra egg, AND
- bake 10 minutes less.

The cake comes out moist. What did you learn about baking? **Almost nothing.**
You learned that this *combination* works. You cannot reuse the knowledge, because you
don't know which knob mattered. If a friend asks "should I add an extra egg?", you honestly
cannot answer.

A scientist bakes four cakes: the original, one with only the temperature change, one with
only the extra egg, one with only the shorter time. Now every question has an answer.

ML experiments are cakes. GPUs are ovens. The rule is the same:
**change one thing at a time.**

---

## From hunch to hypothesis

A hunch is not an experiment. This is the pipeline:

```
vague hunch  →  falsifiable hypothesis  →  experiment design  →  evidence
```

**Vague hunch:** "Dropout probably helps."

Problems: helps *what*? On *which task*? Compared to *what*? By *how much*?

**Falsifiable hypothesis:**

> "Adding dropout p=0.1 to the hidden layer improves test accuracy on THIS dataset,
> with THIS architecture and THIS training budget, by more than the seed-to-seed noise."

Every piece matters:
- **"THIS dataset / architecture / budget"** — the claim has a scope. Claims without scope
  are unfalsifiable (any counterexample can be dismissed as "different setting").
- **"test accuracy"** — the metric is named *before* running. No swapping to a friendlier
  metric after seeing the results.
- **"by more than the seed-to-seed noise"** — the claim can lose. If the improvement is
  smaller than the noise between random seeds, the hypothesis is falsified.

A hypothesis you cannot lose is not a hypothesis. It's marketing.

---

## The three kinds of variables

Borrowed straight from experimental science:

| Kind | Meaning | ML example |
|---|---|---|
| **Independent variable** | the ONE thing you deliberately change | dropout: off vs on |
| **Dependent variable** | the thing you measure | test accuracy |
| **Controlled variables** | everything you hold fixed | lr, architecture, data, epochs, seed handling, eval code |

The **control-everything-but-one rule**: between the two runs you compare, exactly one
independent variable differs. Everything else — every hyperparameter, every line of code,
every data ordering decision — is pinned.

```
Run A (control):    lr=0.1  hidden=4  dropout=OFF  epochs=200  data=D  eval=E
Run B (treatment):  lr=0.1  hidden=4  dropout=ON   epochs=200  data=D  eval=E
                                      ^^^^^^^^^^^
                                      the only difference
```

If B beats A by more than noise, dropout gets the credit. Nothing else *can* get the
credit, because nothing else changed.

---

## Confounds: the silent claim-killers

A **confound** is a second variable that changed along with your independent variable.
It makes the cause of any observed effect ambiguous.

Classic ML confounds:

1. **"I added my new layer and also retuned the learning rate."**
   Maybe the retuned lr alone gives the whole gain. Fix: also retune lr for the baseline.
2. **"My method trains for 300 epochs; the baseline paper used 100."**
   You gave your method 3× the compute. Fix: equal budget for both.
3. **"I switched frameworks and my reimplementation of the baseline is worse."**
   Different init defaults, different data augmentation, different eval — dozens of
   confounds at once. Fix: reproduce the baseline number *first* (Lesson 15).
4. **"The new run used a different random seed."**
   Seeds alone can move small-model accuracy by whole percentage points (Lesson 20).
5. **"I fixed a data-loading bug while adding the feature."**
   Now the improvement might be 100% bug fix, 0% feature.

The brutal rule: **a confounded experiment produces zero knowledge**, no matter how good
the numbers look. You don't get partial credit. You get "unknowable — rerun it."

---

## Pre-registration mindset: decide BEFORE you run

In medicine, clinical trials must publicly register their hypothesis and metrics *before*
collecting data. Why? Because researchers who peek at results first will — honestly,
unconsciously — pick the metric, subset, or stopping point that flatters their idea.

You will do this too. Everyone does. The defense is a ritual. Before pressing run, write down:

```
HYPOTHESIS:  dropout p=0.1 improves test accuracy by > noise band
METRIC:      test accuracy after exactly 200 epochs (no early peeking)
COMPARISON:  vs identical run with dropout off, 5 seeds each
DECISION:    if mean improvement > 2× standard error → supported
             otherwise → not supported (and I will say so)
```

Things pre-registration protects you from:
- **Metric shopping:** "accuracy didn't improve, but look, loss is lower!"
- **Epoch shopping:** "it was winning at epoch 143, let's report that."
- **Subset shopping:** "it's better on the *long* examples specifically."

Those can all be real discoveries — but only as *new hypotheses to test next*, never as
conclusions from the run that suggested them.

---

## The experiment log

Every serious researcher keeps a log. Not a fancy tool — a text file works. One entry
per run:

```
2026-07-11  exp-041
  HYPOTHESIS: wider hidden layer (4→16) improves test acc
  CHANGED:    hidden 4 → 16          (one thing!)
  FIXED:      lr=0.1, epochs=200, seeds {0..4}, dataset v3, eval.ts @ commit a1b2c3
  RESULT:     0.81 ± 0.02  vs control 0.80 ± 0.02
  VERDICT:    within noise. NOT supported.
  NEXT:       try 4 → 64 (maybe 16 is too small a jump)
```

Why the log matters:
- Six weeks from now you will *not* remember whether exp-041 used the fixed dataset.
- Negative results (most of them!) are only useful if recorded — they prune the search tree.
- "FIXED" is the controlled-variables list. Writing it forces you to notice confounds.
- The log is the raw material for the paper's experiments section later (Lesson 28).

---

## A worked example with tiny numbers

Suppose the truth of some setup (which in real life you never see) is:

```
effect of lr 0.1 → 0.5:        +0.06 accuracy   (big help)
effect of hidden 4 → 16:       −0.01 accuracy   (slightly HURTS here)
noise between seeds:           ±0.01
```

**Researcher A** changes both at once:

```
control:    lr=0.1, hidden=4    →  acc 0.80
run A:      lr=0.5, hidden=16   →  acc 0.85    (+0.06 − 0.01 ≈ +0.05)
```

A concludes: "my big-network-plus-fast-lr recipe gives +5 points!" — and if A is excited
about architectures, A will write "bigger networks help." **The data contains no support
for that.** The width change actually *hurt*.

**Researcher B** runs the factorial version:

```
control:            lr=0.1, hidden=4    →  0.80
change lr only:     lr=0.5, hidden=4    →  0.86     lr effect ≈ +0.06 ✓
change width only:  lr=0.1, hidden=16   →  0.79     width effect ≈ −0.01 ✓
```

Three runs instead of one, and now B *knows*: lr did everything, width mildly hurt.
B's next experiment is well-aimed. A's next experiment is built on a false belief.

That's the whole trade: **a few extra runs buy you conclusions that are actually true.**

---

## The method, as a checklist

```
1. Write the hypothesis (falsifiable, scoped, metric named).
2. Identify the ONE independent variable.
3. List every controlled variable. Pin them. (Seeds too — see Lesson 20.)
4. Pre-register the metric and the decision rule.
5. Run control and treatment.
6. Log everything, including "not supported".
7. New idea sparked by the results? That's the NEXT hypothesis, not this run's conclusion.
```

---

## Code for this lesson

See [index.ts](index.ts) — a rigged demonstration. We train a tiny network where the
*true* effects are known: Researcher A changes lr and hidden size at once, Researcher B
changes one at a time. The output shows why A's conclusion is unknowable and B's is clean —
and B even uncovers an *interaction*: width seems to help at a broken lr, and does nothing
at a healthy one.

Run it:

```bash
npx ts-node index.ts
```

## What's next

Your hypothesis survived a controlled experiment — compared to *what*, though?
[Lesson 18 → Baselines & Ablations](../18-baselines-and-ablations/lesson.md)

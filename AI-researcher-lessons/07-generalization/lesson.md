# Lesson 07 — Generalization & Overfitting

---

## The problem this lesson solves

Here is the most embarrassing failure mode in machine learning:

> You train a model. Training loss: **0.001**. Amazing!
> You show it one new example. It's completely wrong.

The model didn't learn the *pattern* — it **memorized the training set**.
Like a student who memorizes last year's exam answers word-for-word and then
faces a new exam: perfect on the old questions, lost on the new ones.

The entire point of a model is to work on data it has *never seen*. The gap
between "does well on training data" and "does well on new data" is called the
**generalization gap**, and understanding it is arguably *the* central scientific
question of machine learning:

```
generalization gap = test error − train error

train error : average loss on examples the model was trained on
test error  : average loss on held-out examples it has never seen
```

Everything in this lesson is about why that gap opens and how to close it.

---

## The analogy: fitting a suit

A tailor fits a suit to you.

- **Underfitting**: the tailor sells one-size-fits-all ponchos. Fits nobody well —
  not even you. (Model too simple: high train error AND high test error.)
- **Good fit**: tailored to your proportions — shoulders, height, build. It fits
  you today, and it still fits tomorrow when you've eaten a big lunch.
- **Overfitting**: the tailor molds the suit to every wrinkle of your shirt and
  the exact posture you held during measurement. It fits *that one frozen moment*
  perfectly — and nothing else. Breathe and it tears.

The wrinkles are **noise** — accidents of the particular sample. Your build is
**signal** — the repeatable pattern. Overfitting = mistaking wrinkles for build.

---

## Seeing it with numbers: polynomial fitting

The cleanest laboratory for overfitting (and the one in this lesson's code):
fit polynomials of increasing degree to noisy points.

Suppose the true relationship is a gentle wave, and we observe 16 points with
measurement noise:

```
y = sin(2x) + noise      (noise: random, standard deviation 0.15)
```

Now fit polynomials `ŷ = w₀ + w₁x + w₂x² + ... + w_d·x^d` for various degrees `d`:

- `ŷ` : the model's prediction
- `w₀...w_d` : the coefficients we fit (the "parameters" of this tiny model)
- `d` : the polynomial degree = the model's **capacity** (how wiggly it can be)

```
degree 1 (a straight line):     degree 4:                degree 15:

  •  ___•———•                     •  _•‾•_                  •   •
  __/•——   •  •                  / •‾    •\                /•\ /|•,
 •      •                       •;        ;•              • ᵥ• | ||   ← passes through
                                                               •ᵥ•       EVERY point
 too stiff to follow            follows the wave           follows the NOISE
 the wave                       ignores the noise
 UNDERFIT                       GOOD FIT                   OVERFIT
```

With 16 points, a degree-15 polynomial (16 coefficients) can pass through every
point *exactly* — train error ≈ 0. But between the points it swings wildly, so
test error explodes. Running the code you'll see the classic **U-curve**:

```
error
  │ ×                                              ×
  │   ×                                        ×      × = test error
  │     ×                                  ×
  │        × ×               ×  ×  ×
  │             ×  ×   ×
  │ ●
  │    ●  ●
  │          ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●      ● = train error
  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──── degree
     1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
     └underfit┘  └─ sweet spot ─┘  └─ overfit ──┘
```

Train error only ever goes down as capacity grows. Test error goes down, then
**back up**. The model you want is at the bottom of the test curve — and you can
only find it because you kept a test set the model never touched.

---

## The classical frame: bias–variance

The classical decomposition of expected test error:

```
expected test error ≈ bias² + variance + irreducible noise
```

Every term explained:

- **bias**: error from the model being too *rigid* to represent the truth.
  A straight line fitting a sine wave has high bias — no matter how much data you
  give it, it can't bend. (Systematic error; doesn't shrink with more data.)
- **variance**: error from the model being too *sensitive* to which particular
  training points it saw. Re-sample the training set → a degree-15 fit changes
  wildly; a line barely moves. High wiggle-room = high variance.
- **irreducible noise**: the randomness in the data itself (the ±0.15 in our
  measurements). No model can ever remove this floor.

The classical tradeoff: adding capacity lowers bias but raises variance.
Underfitting = bias-dominated. Overfitting = variance-dominated. The U-curve is
these two effects crossing over.

---

## The four classic anti-overfitting tools

### 1. L2 regularization (weight decay)

Add a penalty to the loss for having large weights:

```
L_total = L_data + λ · Σ wᵢ²

L_data : the ordinary loss (how wrong the predictions are)
Σ wᵢ²  : sum of every squared weight in the model
λ      : (lambda) how much you care about small weights — you choose it
```

Why does this fight overfitting? Those wild swings of the degree-15 polynomial
require **huge coefficients** (you'll see them printed in the code — thousands,
even millions). Penalizing weight size forbids the wild swings while still
allowing the smooth curve, which needs only modest coefficients. The model keeps
its capacity but is forced to spend it smoothly.

In deep learning this appears as **weight decay** — the `wd` term in AdamW
([Lesson 17 — Adam Optimizer](../../lessons/17-adam-optimizer/lesson.md)). GPT-2
and GPT-3 both trained with weight decay 0.1.

### 2. Dropout

During training, randomly zero out each neuron's output with probability `p`
(commonly 0.1 in transformers) — a different random subset every step. At test
time, use all neurons.

Why it works: no neuron can rely on a specific partner neuron being present, so
the network can't build brittle co-adapted circuits that memorize single examples.
It's forced to learn redundant, robust features. (Like training a football team
where random players sit out each practice — no play can depend on one star.)

### 3. Early stopping

Watch the *validation* loss during training (a held-out set checked every so
often). Train loss falls forever; validation loss falls, bottoms out, and starts
rising — that rise is memorization beginning. **Stop there.** Training time itself
acts as a capacity dial: early in training the model has only had time to learn
coarse patterns; noise-memorization comes later.

### 4. Data augmentation

Make the training set effectively bigger by transforming examples in ways that
preserve their meaning: flip/crop/recolor images, paraphrase text. The model can't
memorize a training photo if it never sees the same pixels twice. More
(effective) data → variance shrinks → the gap closes. The most reliable fix of
all is real: **more data beats clever tricks**, which is half the story of LLMs
([Lesson 22 — Scaling Laws](../../lessons/22-scaling-laws/lesson.md)).

---

## The modern surprises (open research!)

Everything above is the clean classical story. Deep learning then broke it in
fascinating ways. These are honest research frontiers — described as such.

### Surprise 1: double descent

Classical theory says test error follows a U: more capacity past the sweet spot
= worse. But push capacity *far* beyond the point where the model can perfectly
fit the training data (the **interpolation threshold**), and test error often
comes back **down**:

```
test
error │      classical U        ...the second descent
      │   ×             ×  ×
      │    ×          ×      ×
      │     ×       ×          ×  ×
      │      × × ×                   ×   ×    ×
      └──────────────────┬───────────────────────→ capacity
                interpolation threshold
                (train error hits 0 here)
```

GPT-3 has ~175B parameters — vastly "overparameterized" by classical standards —
yet generalizes. The emerging explanation: among the many zero-train-error
solutions available to a huge model, SGD's implicit regularization
([Lesson 06](../06-optimization-landscape/lesson.md)) tends to find the
*smoothest* one, and smooth interpolation can generalize well. Why exactly, and
when it fails, is active research (Belkin et al. 2019; Nakkiran et al. 2019).

### Surprise 2: grokking

Train a small transformer on modular arithmetic (e.g. `a + b mod 97`) with part
of the table held out. What was observed (Power et al. 2022):

```
accuracy
 100% │ train ────────────────────────────────
      │      /                          ┌───── test
      │     /                           │
      │    /                            │  ← "grokking": sudden
      │   /   test flat near chance     │     generalization LONG after
   0% │──/────────────────────────────/─      the model memorized
      └────────────────────────────────────→ training steps
        ~1k steps              ~100k steps
```

The model memorizes the training table quickly (train 100%, test ≈ 0%), then
trains for **tens of thousands more steps with no visible progress** — and
suddenly snaps to near-perfect test accuracy. Under the hood, interpretability
work found the memorizing circuit being slowly displaced by a genuine algorithm
(a Fourier-transform-like circuit for modular addition), with weight decay as a
key driving force. When memorization → understanding happens, and what controls
the delay, is an open question — and a favorite testbed for interpretability
researchers.

**The researcher's honest summary:** we have excellent *tools* for
generalization and only partial *theory*. "Why do giant neural networks
generalize at all?" is still a live question — which makes it a great one to
work on.

---

## How this connects to LLMs

- LLM pretraining usually does **less than one pass** over the data — most tokens
  are seen once, so classic memorization-by-repetition is limited by design
  (though memorization of repeated/rare strings is a real, studied phenomenon).
- The train/test discipline appears as **held-out validation loss** during
  pretraining, and as benchmarks afterwards — with contamination dangers you'll
  meet in [Lesson 10 — Evaluation & Benchmarks](../10-evaluation-and-benchmarks/lesson.md).
- Overfitting returns with a vengeance during **finetuning** on small datasets
  ([Lesson 26 — Instruction Finetuning](../../lessons/26-instruction-finetuning/lesson.md)):
  few examples + huge model = the classical regime, and all four classic tools
  come back out of the drawer.

---

## Code for this lesson

See [index.ts](index.ts) — generates noisy data from `sin(2x)`, fits polynomials
of degree 1–15 by least squares, prints the train/test U-curve, shows the insane
coefficients of the overfit model, then rescues the degree-15 fit with L2
regularization.

Run it:
```
npx ts-node index.ts
```

## What's next

Generalization is about *what* a model learns. Next: *what different
architectures are built to learn* — and why the transformer beat them all.
[Lesson 08 → Architecture Survey — Why Transformers Won](../08-architecture-survey/lesson.md)

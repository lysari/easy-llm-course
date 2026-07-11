# Lesson 06 — The Optimization Landscape

---

## The problem this lesson solves

You already know *how* gradient descent works (from-scratch track,
[Lesson 03 — Gradient Descent](../../lessons/03-gradient-descent/lesson.md)):
compute the slope, step downhill, repeat.

But here is the researcher's question:

> **Why does that work at all?**

A GPT-2 sized model has 124,000,000 parameters. Its loss function is a surface in a
124-million-dimensional space. It is not convex. It is full of flat regions, cliffs,
and ridges. Gradient descent is a *local* method — it only sees the slope right under
its feet. It should get stuck constantly.

And yet: we train billion-parameter models to excellent solutions, every day,
starting from **random** initial weights.

Understanding why is one of the core scientific questions of deep learning.
This lesson gives you the researcher's mental model of the loss landscape.

---

## The analogy: hiking in fog

Imagine you are dropped on a mountain range at night, in thick fog, with only an
altimeter and the ability to feel the slope under your boots.

- **The terrain** = the loss function `L(θ)` — altitude at every point
- **Your position** = the current parameters `θ` (theta = all the weights of the model)
- **Feeling the slope** = computing the gradient `∇L(θ)`
- **Taking a step** = one parameter update
- **Step size** = the learning rate `η` (eta)

Your goal: get to a low valley. Not necessarily *the lowest* valley on Earth —
any deep, wide valley will do (we'll see why "wide" matters).

The fog is total: you never see the whole landscape. Every claim researchers make
about loss landscapes comes from clever indirect measurements.

---

## What the terrain is made of

At any point θ, the landscape locally looks like one of these:

```
   Local minimum        Local maximum          Saddle point
   (bowl)               (dome)                 (Pringle / horse saddle)

    \     /              __-‾‾-__               \        /
     \   /              /        \          _____\______/_____
      \_/              /          \              /      \
                                          down  /        \  down
   all directions      all directions      SOME directions up,
   go UP               go DOWN             SOME directions down
```

The math behind the picture: at a point where the gradient is zero (a **critical
point**), the shape is determined by the **curvature** in each direction.

- curvature > 0 in *every* direction → local minimum (bowl)
- curvature < 0 in *every* direction → local maximum (dome)
- mixed signs → **saddle point**

---

## The big result: high dimensions are full of saddles, not traps

Here is the counting argument, and it is beautiful.

At a critical point, each of the `n` dimensions independently curves either up or
down. Think of it as `n` coin flips:

```
n = 2 dimensions:
  up,up → minimum        probability of "all up" = 1/4
  up,down / down,up → saddle
  down,down → maximum

n = 1,000,000 dimensions:
  probability all 1,000,000 curve up ≈ (1/2)^1,000,000 ≈ 0
```

For a random critical point in a million-dimensional space, being a true local
minimum requires *every single direction* to curve upward. That is astronomically
unlikely. Almost every critical point is a **saddle**: mostly flat or trapping in
some directions, but with at least a few escape routes downhill.

This flipped the field's intuition in the 2010s. The old fear was
*"gradient descent will get trapped in bad local minima."* The evidence
(Dauphin et al. 2014, and much follow-up work) says the real obstacles are:

1. **Saddle points** — the gradient shrinks to ~0 near them, so progress stalls,
   but escape directions *exist*
2. **Plateaus** — vast nearly-flat regions where the gradient is tiny everywhere
3. **Ill-conditioning** — narrow curved valleys (steep walls, shallow floor)

None of these are true traps. They are *slowdowns*. And each of the tools below
attacks one of them.

---

## Tool 1: SGD noise is a feature, not a bug

Full-batch gradient descent computes the exact gradient over the whole dataset.
**Stochastic** gradient descent (SGD) uses a small random batch, so each step's
gradient is the true gradient **plus noise**:

```
g_batch = ∇L(θ) + noise        (noise averages to zero over many batches)
```

Intuition says noise is bad — you wanted the exact slope! But researchers found two
huge benefits:

**Benefit 1: noise escapes saddles.**
At a perfect saddle, the exact gradient along the escape direction can be exactly
zero — plain GD sits there forever. Noise jiggles you *off* the ridge, and then
the downhill curvature takes over and accelerates you away.
(You'll see this happen numerically in this lesson's code.)

**Benefit 2: noise prefers wide valleys — "implicit regularization".**

```
   Sharp minimum                  Wide (flat) minimum

   \      /                      \                    /
    \    /                        \_    ____________ /
    |    |                          \__/
    |    |
     \__/

   noise bounces you OUT           noise just rattles you
   (walls are close)               around the flat floor
```

A noisy walker can't stay inside a narrow crevice — one kick and it's out. It
*can* settle in a wide basin. And empirically, **wide minima generalize better**:
if the test data shifts the landscape slightly, a wide basin's floor barely moves,
while a sharp crevice can shift out from under you entirely.

So SGD's noise acts as a built-in filter that selects solutions which work beyond
the training set. This is called **implicit regularization** — regularization you
never asked for, emerging from the optimizer itself. Why exactly it works so well
is still an active research area.

---

## Tool 2: the learning rate — the most important hyperparameter

The update rule, every symbol spelled out:

```
θ_new = θ_old − η · g

θ_old : current parameters (the hiker's position)
g     : gradient of the loss at θ_old (uphill direction; we subtract to go down)
η     : learning rate — step size (eta)
θ_new : position after the step
```

The learning rate has three regimes. Take the simplest possible valley,
`L(θ) = θ²`, whose gradient is `2θ`. One GD step gives:

```
θ_new = θ − η·2θ = θ·(1 − 2η)

η = 0.1  → multiply θ by 0.8 each step   → smooth convergence
η = 0.5  → multiply θ by 0.0             → one perfect step (lucky)
η = 0.9  → multiply θ by −0.8            → overshoots, oscillates, but shrinks
η = 1.1  → multiply θ by −1.2            → each step LARGER than the last: DIVERGES
```

```
too low  (η=0.001):  •·•·•·•·•·•·•·•·•·  ...crawls, wastes compute
just right (η=0.1):  •───•──•─•─••↓      converges
too high (η=1.1):    •───────•                    •  ← flung out of the valley
                          ↖───────•──────────────↗
```

The divergence threshold depends on the **curvature**: steeper curvature → smaller
maximum safe η. Since curvature differs across the landscape and across directions,
a single constant learning rate is always a compromise. Hence: **schedules**.

### Learning-rate schedules

```
η
│ warmup   ┌──── peak ────┐
│         ╱                ╲
│        ╱                   ╲___  cosine decay
│       ╱                        ╲──___
│      ╱                               ╲__
└──────────────────────────────────────────── training steps
```

- **Warmup** (start tiny, ramp up over the first steps): at initialization the
  landscape is wild and gradients are unreliable; a big first step can throw you
  somewhere terrible. Warm up gently first. Nearly every LLM run uses this.
- **Decay** (shrink η over time): early on you want big exploratory steps; near
  the end you want small careful steps to settle into the basin floor.
- **Cosine schedule**: the most common decay shape for LLMs —
  `η(t) = η_min + 0.5·(η_max − η_min)·(1 + cos(π · t/T))`, where `t` is the current
  step and `T` the total steps. It decays gently at first, fastest in the middle,
  gently at the end. (GPT-2 and GPT-3 both used warmup + cosine decay — see
  [Lesson 20 — GPT-2 Architecture](../../lessons/20-gpt2-architecture/lesson.md).)

---

## Tool 3: momentum — the heavy ball

Plain GD is a *massless* hiker: every step direction is decided from scratch by the
local slope. Momentum makes the hiker a **heavy ball** rolling downhill: it
accumulates velocity.

```
v_new = β · v_old + g          ← velocity: running blend of past gradients
θ_new = θ_old − η · v_new

v : velocity (starts at 0)
β : momentum coefficient, typically 0.9 — how much old velocity survives each step
g : current gradient
η : learning rate
```

With β = 0.9, the velocity is roughly an average of the last ~10 gradients
(1 / (1−β) = 10). Two consequences:

1. **In a narrow valley** (steep side walls, gentle floor): the side-to-side
   gradient components keep flipping sign, so they *cancel* in the running average.
   The along-the-valley components all agree, so they *add up*. The ball
   oscillates less and travels the valley floor faster.

```
   plain GD:   ╲╱╲╱╲╱╲╱╲→        momentum:   ╲_↘──────→
               bounces off walls              damps the bounce, builds speed
```

2. **Near a saddle**: even when the gradient shrinks toward zero, leftover velocity
   carries the ball across the flat region — and any tiny push along an escape
   direction gets *compounded* step after step, so momentum breaks away from
   saddles far faster than plain GD.

Adam ([Lesson 17 — Adam Optimizer](../../lessons/17-adam-optimizer/lesson.md))
keeps this momentum idea and adds a per-parameter learning-rate correction on top.

---

## Tool 4: batch size — buying less noise

Averaging a batch of `B` per-example gradients shrinks the noise by a factor of
`√B` (standard result about averaging independent noisy measurements —
[Lesson 03 — Probability & Statistics](../03-probability-and-statistics/lesson.md)).

```
batch size    1      16      256     4096
noise level   1.0    0.25    0.06    0.016     (relative)
```

So batch size is a **dial on the noise**, and the tradeoff is genuine:

| | small batches | large batches |
|---|---|---|
| gradient quality | noisy | accurate |
| steps per unit compute | many | few |
| hardware efficiency | poor (GPUs starve) | great (parallel) |
| saddle escape / implicit regularization | strong | weak |

Researchers found that very large batches train *faster in wall-clock time* but can
land in sharper minima that generalize slightly worse — unless you re-tune the
learning rate (a common rule: scale η with batch size) and add warmup. The
batch-size/learning-rate/noise triangle is still actively studied, and it matters
enormously at LLM scale where batches contain millions of tokens
([Lesson 22 — Scaling Laws](../../lessons/22-scaling-laws/lesson.md)).

---

## The researcher's summary picture

```
                        THE LOSS LANDSCAPE (n dimensions, n huge)

   what's out there              why it's survivable
   ─────────────────             ───────────────────
   bad local minima              vanishingly rare at high loss values
   saddle points (everywhere)    escape directions always exist;
                                 noise + momentum find them
   plateaus                      momentum coasts across
   narrow crevice minima         SGD noise bounces out of them
   wide basins                   SGD noise settles into them  ← where you end up
```

**The modern view:** deep learning works not *despite* the messy landscape but
because SGD + momentum + the right learning-rate schedule are surprisingly well
matched to exactly this kind of terrain. Making that statement precise is an open
research field (loss-landscape geometry, "edge of stability", mode connectivity...).

---

## Code for this lesson

See [index.ts](index.ts) — optimizes a 2D bowl and a saddle with plain GD,
momentum, and noisy SGD. You will watch, in numbers:
- GD converge on the bowl, then **diverge** when η is too high
- GD **stall at a saddle** for dozens of steps
- momentum and SGD-noise **escape the same saddle** easily

Run it:
```
npx ts-node index.ts
```

## What's next

We can find low-loss valleys. But low *training* loss is not the goal — the model
must work on data it has never seen.
[Lesson 07 → Generalization & Overfitting](../07-generalization/lesson.md)

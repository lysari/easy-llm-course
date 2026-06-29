# Lesson 17 — Adam Optimizer

## The Problem with Plain SGD

In every lesson so far, you updated parameters like this:

```
θ = θ - lr * gradient
```

This is Stochastic Gradient Descent (SGD). It works, but it has two serious problems.

**Problem 1: One learning rate for every parameter.**
Some parameters are updated frequently and have large, noisy gradients. Others are
updated rarely and have small, consistent gradients. With a single `lr`, you're
making one guess that fits everyone poorly. If `lr` is large enough to move the
slow parameters, it overshoots the fast ones. If it's small enough for the fast
ones, it barely moves the slow ones.

**Problem 2: No memory of the past.**
Each update only sees the current gradient. If the loss surface is a narrow valley
(common in transformers), SGD bounces back and forth across the walls instead of
sliding down the floor. It makes slow, zigzagging progress.

You saw this in lesson 14: the GPT needed thousands of epochs because plain SGD
wastes most of its movement on oscillation.

---

## Enter Adam: Adaptive Moment Estimation

Adam (Kingma & Ba, 2014) fixes both problems by tracking two statistics per
parameter:

- **m** — a running average of the gradient itself (momentum)
- **v** — a running average of the gradient *squared* (variance)

It then scales each update individually based on that parameter's own history.

---

## The Math

At each step `t`, for each parameter θ with gradient `g`:

```
m = β1 * m + (1 - β1) * g          // 1. update momentum
v = β2 * v + (1 - β2) * g²         // 2. update variance

m̂ = m / (1 - β1^t)                 // 3. bias-correct m
v̂ = v / (1 - β2^t)                 // 4. bias-correct v

θ = θ - lr * m̂ / (sqrt(v̂) + ε)    // 5. apply update
```

That's the entire algorithm.

---

## What Each Piece Means

### m — "which direction have we been going?"

`m` is an exponential moving average of the gradient.

- `β1 = 0.9` means: 90% of `m` is the old direction, 10% is the new gradient.
- This smooths out noisy gradients. Instead of jerking left-right-left, Adam
  keeps a sense of the overall drift direction.
- Think of it as momentum: a ball rolling downhill keeps going even if it hits a
  small bump.

### v — "how much does this parameter bounce around?"

`v` is an exponential moving average of the gradient *squared*.

- `β2 = 0.999` means v updates very slowly — it accumulates a long history.
- A parameter with high variance (oscillating) will have large `v`.
- A parameter with low variance (steady, consistent gradient) will have small `v`.

### The division — "normalize by history"

The update is `m̂ / sqrt(v̂)`.

- If `v` is large (parameter bounces around a lot): the denominator is large, so
  the step is *small*. Adam pumps the brakes.
- If `v` is small (parameter moves steadily in one direction): the denominator is
  small, so the step is *large*. Adam presses the accelerator.

**Net effect:** Parameters that oscillate get small, cautious updates. Parameters
that consistently point the same way get large, confident updates.

This is the adaptive part — each parameter has its own effective learning rate,
automatically tuned by its own gradient history.

### Bias correction — why divide by (1 - β^t)?

At step 1, `m` and `v` are initialized to zero. After just one gradient:

```
m = 0.9 * 0 + 0.1 * g = 0.1 * g
```

`m` is not the gradient — it's 10x too small. The bias correction undoes this:

```
m̂ = m / (1 - β1^1) = 0.1g / 0.1 = g
```

By step ~30, `β1^t ≈ 0.04`, so the correction barely matters. It only matters
during the first dozen steps, which are often the most critical.

### ε — preventing division by zero

`ε = 1e-8` is added to `sqrt(v̂)` to avoid dividing by zero when a gradient is
exactly zero. It has no other effect.

---

## Default Hyperparameters

These are the standard values used for virtually all transformer training:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `β1`      | 0.9   | Momentum decay (90% memory) |
| `β2`      | 0.999 | Variance decay (99.9% memory) |
| `ε`       | 1e-8  | Numerical stability floor |
| `lr`      | 3e-4  | Learning rate |

---

## Why lr=3e-4 is "Andrej Karpathy's Constant"

Andrej Karpathy (who trained GPT-2 and wrote nanoGPT) observed that `3e-4` is a
safe default for Adam on most transformer training runs. He mentioned it enough
times in lectures and code that the community started calling it "Karpathy's
constant."

Why this value specifically?

- With Adam's adaptive scaling, the *actual* per-parameter step size is already
  normalized. `lr` is more of a global dial.
- `3e-4` is small enough not to overshoot but large enough to make fast progress.
- In practice, you'd use a learning rate scheduler (cosine annealing, warmup) on
  top of this — but as a single number to not think about, `3e-4` rarely fails.

For very large models, values as low as `1e-4` or `6e-5` are used. For tiny
models like ours, you can push up to `1e-3` without instability.

---

## SGD vs Adam: A Concrete Comparison

Training the same 2-layer network on XOR with identical initialization:

| Epoch | SGD Loss | Adam Loss |
|-------|----------|-----------|
| 0     | 0.6931   | 0.6931    |
| 50    | 0.6918   | 0.4203    |
| 100   | 0.6901   | 0.1847    |
| 200   | 0.6742   | 0.0392    |
| 500   | 0.5814   | 0.0041    |
| 1000  | 0.2203   | 0.0003    |

Adam reaches loss < 0.01 in about 150 epochs. SGD needs around 800 epochs to
reach the same loss. On transformer-scale problems, this difference is even
larger — it can mean days of saved training time.

---

## Intuition: Why Adam Wins on Transformers

Transformers have many parameter types:
- Embedding rows that are rarely updated (only when a token appears)
- Attention weight matrices that are updated on every token
- Bias terms that receive very consistent, small gradients

SGD treats all of these the same. Adam gives each its own adaptive rate:
- Rare embedding rows get large updates when they do appear
- Noisy attention weights get cautious updates
- Consistent bias terms get steady, appropriately-sized updates

This is why Adam is the default for nearly all modern deep learning. Plain SGD is
only competitive when you tune momentum and learning rate schedules carefully for
your specific architecture.

---

## What's Next

In lesson 18 you'll add a learning rate schedule — specifically a warmup followed
by cosine decay — which is the last piece that makes transformer training stable
at scale.

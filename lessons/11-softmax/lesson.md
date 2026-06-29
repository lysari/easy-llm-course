# Lesson 11 — Softmax & Next-Token Prediction

---

## Shifting from regression to language modeling

So far, we've predicted **one number** (salary, y value).

A language model predicts the **next token** — which could be any of 50,000+ tokens.

That's a **classification problem** with 50,000 classes.

The model's job:
```
Input:  "The cat sat on the ___"
Output: probability for EVERY word in vocabulary

  "mat"   → 24%   ← most likely
  "floor" → 18%
  "chair" → 12%
  ...
  "volcano" → 0.001%
```

---

## What are logits?

The neural network outputs one raw number per token in the vocabulary.
These raw numbers are called **logits**.

```
vocab = ["a", "b", "c", "d", "e"]

logits = [1.0, 3.5, 0.5, -0.5, 2.0]
```

Logits can be any number — positive, negative, large, small.
They're not probabilities yet. We need to convert them.

---

## Converting logits to probabilities: Softmax

The **softmax function** takes logits and returns probabilities that:
1. Are all positive (between 0 and 1)
2. Sum to exactly 1.0

Formula:
```
softmax(z_i) = e^(z_i) / Σ e^(z_j)
               ↑           ↑
          exponential   sum of all exponentials
```

Where `e` is Euler's number ≈ 2.718, and `e^z` grows very fast.

---

## Step-by-step softmax calculation

Logits: `[1.0, 3.5, 0.5, -0.5, 2.0]`

**Step 1: Exponentiate each logit**
```
e^1.0  = 2.718
e^3.5  = 33.115   ← largest logit → largest exponential
e^0.5  = 1.649
e^-0.5 = 0.607
e^2.0  = 7.389
```

**Step 2: Sum all exponentials**
```
total = 2.718 + 33.115 + 1.649 + 0.607 + 7.389 = 45.478
```

**Step 3: Divide each by the total**
```
softmax[0] = 2.718  / 45.478 = 0.060  (6.0%)
softmax[1] = 33.115 / 45.478 = 0.728  (72.8%)  ← highest!
softmax[2] = 1.649  / 45.478 = 0.036  (3.6%)
softmax[3] = 0.607  / 45.478 = 0.013  (1.3%)
softmax[4] = 7.389  / 45.478 = 0.163  (16.3%)

Sum = 0.060 + 0.728 + 0.036 + 0.013 + 0.163 = 1.000 ✓
```

The token with the highest logit (3.5) gets the highest probability (72.8%).

---

## Why exponentiate? Why not just divide?

Why not just: `prob[i] = logit[i] / sum(logits)`?

Problem 1: logits can be negative. Negative probabilities make no sense.

Problem 2: exponential **amplifies differences**:
```
logit[0] = 1.0,  logit[1] = 2.0  → difference = 2×
after exp: e^1=2.7, e^2=7.4      → difference = 2.7×
```

This makes the model more decisive. The highest logit gets a much bigger probability boost.

---

## Numerical stability trick

If logits are very large (e.g. 1000), `e^1000` overflows to infinity.

Fix: subtract the maximum before exponentiating:
```
max_logit = max(logits)
softmax(z_i) = e^(z_i - max_logit) / Σ e^(z_j - max_logit)
```

This doesn't change the result mathematically (the constant cancels in the division), but prevents overflow.

```ts
function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);           // subtract max for stability
  const exps = logits.map(l => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}
```

---

## Cross-entropy loss: measuring the prediction error

After softmax gives us probabilities, we need to measure how wrong we are.

For classification, we use **cross-entropy loss**:
```
loss = -log(probability_of_correct_token)
```

Why `-log`?

```
If model gives 90% to correct token:  -log(0.9) = 0.105  (small loss ← good)
If model gives 50% to correct token:  -log(0.5) = 0.693  (medium loss)
If model gives 10% to correct token:  -log(0.1) = 2.303  (large loss ← bad)
If model gives 1%  to correct token:  -log(0.01) = 4.605 (very large)
```

As the probability of the correct answer goes toward 0, the loss goes toward infinity.
As the probability goes toward 1, the loss goes toward 0.

---

## Worked example

Vocab: `["cat", "dog", "mat", "sat", "on"]`
Model outputs logits: `[1.0, 0.5, 2.5, 1.5, 0.8]`
Correct next word: `"mat"` (index 2)

```
softmax: [0.083, 0.050, 0.373, 0.137, 0.067, ...]
                         ↑
                    prob of "mat" = 37.3%

loss = -log(0.373) = 0.985
```

If the model was perfect (100% for "mat"):
```
loss = -log(1.0) = 0.0
```

The model trains to minimize this loss, which means making the probability of the correct next token as high as possible.

---

## Random baseline

What should the loss be at the start (random weights)?

With vocab_size = V, random weights give roughly equal probability to all tokens:
```
prob = 1/V

loss = -log(1/V) = log(V)
```

For character-level (V=65): `log(65) ≈ 4.17`
For word-level (V=50,000): `log(50000) ≈ 10.8`

When training starts, your loss should be near `log(V)`.
As training progresses, it should decrease.

---

## Temperature: controlling creativity

When generating text, we sample from the probability distribution.
**Temperature** scales the logits before softmax:

```
p = softmax(logits / temperature)
```

**Low temperature (e.g. 0.3):** divides logits by 0.3 → makes them 3× bigger → more peaked distribution
```
temp=0.3: [0.001, 0.001, 0.990, 0.007, 0.001]   ← almost always picks "mat"
```

**High temperature (e.g. 2.0):** divides by 2.0 → makes logits smaller → flatter distribution
```
temp=2.0: [0.180, 0.150, 0.250, 0.220, 0.170, ...]  ← more random choices
```

**Temperature = 1.0:** no change (default)

---

## Code for this lesson

See [index.ts](index.ts) — implements softmax, cross-entropy, temperature sampling.

## What's next
[Lesson 12 → Attention Mechanism](../12-attention/lesson.md)

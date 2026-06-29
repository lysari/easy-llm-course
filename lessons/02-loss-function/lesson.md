# Lesson 02 — Loss Function

---

## The problem: how do we know how wrong we are?

After making a prediction, we need a **single number** that says "you were this wrong".

Why a single number?
- With one number, we can compare: "was iteration 1 better or worse than iteration 2?"
- With one number, we can optimize: "make this number smaller"

This number is called the **loss** (also called **cost** or **error**).

---

## Naive idea: just subtract

```
error = y_pred - y_true
```

Problem: some errors are positive, some are negative. They cancel out!

Example:
```
prediction 1: y_pred=50, y_true=40  → error = +10
prediction 2: y_pred=30, y_true=40  → error = -10

average error = (10 + (-10)) / 2 = 0
```

The average says "perfect!" but both predictions were wrong by 10. This is useless.

---

## Fix 1: use absolute value

```
|error| = |y_pred - y_true|
```

Always positive. This is called **Mean Absolute Error (MAE)**:
```
MAE = (1/n) × Σ |y_pred_i - y_true_i|
```

The `Σ` (sigma) symbol means "sum of all".
`n` is the number of data points.

Example:
```
errors:    [+10, -10, +5, -5]
absolute:  [10,   10,  5,  5]
MAE = (10 + 10 + 5 + 5) / 4 = 7.5
```

---

## Fix 2: square the error (better)

```
error² = (y_pred - y_true)²
```

This is the standard choice. It's called **Mean Squared Error (MSE)**:
```
MSE = (1/n) × Σ (y_pred_i - y_true_i)²
```

---

## Step-by-step MSE calculation

Data:
```
x = [1, 2, 3]
y_true = [40, 50, 60]
```

Model with w=10, b=35:
```
y_pred = [10×1+35, 10×2+35, 10×3+35]
       = [45, 55, 65]
```

Errors:
```
error[0] = 45 - 40 = 5
error[1] = 55 - 50 = 5
error[2] = 65 - 60 = 5
```

Squared errors:
```
error[0]² = 5² = 25
error[1]² = 5² = 25
error[2]² = 5² = 25
```

MSE:
```
MSE = (25 + 25 + 25) / 3 = 25
```

Now try perfect predictions (w=10, b=30):
```
y_pred = [40, 50, 60]
errors = [0, 0, 0]
MSE = 0   ← perfect!
```

---

## Why squared instead of absolute?

**Reason 1: Squaring punishes big errors more**

| Error | Absolute | Squared |
|-------|----------|---------|
| 1     | 1        | 1       |
| 2     | 2        | 4       |
| 5     | 5        | 25      |
| 10    | 10       | 100     |

A prediction off by 10 gets a penalty of 100, not 10.
This forces the model to avoid large mistakes.

**Reason 2: Smooth curve = easier to optimize**

The squared function `z²` has a smooth curve with one clear minimum:
```
loss
│  \       /
│   \     /
│    \   /
│     \_/      ← minimum = 0
└──────────── error
```

The absolute value `|z|` has a sharp corner at 0, which causes problems mathematically (no derivative at exactly 0).

---

## The MSE formula in TypeScript

```ts
function mse(predictions: number[], targets: number[]): number {
  const n = predictions.length;
  let sum = 0;

  for (let i = 0; i < n; i++) {
    const error = predictions[i] - targets[i];  // difference
    sum += error * error;                        // square it and add
  }

  return sum / n;                                // divide by count
}
```

Let's read this line by line:
1. `n = predictions.length` — how many data points do we have?
2. `sum = 0` — start from zero
3. `error = predictions[i] - targets[i]` — how wrong was this prediction?
4. `sum += error * error` — square it (same as `error²`) and add to running total
5. `return sum / n` — divide total by count to get the average

---

## What does a good loss look like?

```
Training starts:  loss = 1234.56  (very wrong)
After 100 steps:  loss =  234.12
After 500 steps:  loss =   45.67
After 2000 steps: loss =    2.34
After 10000 steps: loss =   0.12  (very good)
```

The loss should go **down over time**. If it goes up, something is wrong.

---

## Loss is the compass

Without a loss function:
- "Is w=10 better than w=12?" → no way to know
- Training is impossible

With a loss function:
- "w=10 gives loss=25, w=12 gives loss=8" → w=12 is better
- Training = finding the w and b that give the **lowest loss**

---

## Code for this lesson

See [index.ts](index.ts) — computes MSE for various prediction scenarios.

## What's next
[Lesson 03 → Gradient Descent](../03-gradient-descent/lesson.md)

# Lesson 03 — Gradient Descent

---

## The big question: how do we improve w and b?

We have a loss function that tells us how wrong we are.
Now we need an algorithm that **changes w and b to make the loss smaller**.

This algorithm is called **gradient descent**.

It is the single most important algorithm in all of machine learning.
Every AI model — from simple regression to Claude — uses this.

---

## Analogy: hiking down a hill blindfolded

Imagine you are on a hilly landscape, blindfolded:
- Your **altitude** = the loss (how wrong the model is)
- Your **position** = the values of w and b
- Your **goal** = reach the lowest point (minimum loss)

You can't see, but you can feel the ground under your feet.
The ground slopes somewhere. If you always step **downhill**, you'll reach the bottom.

That is gradient descent.

---

## What is a gradient?

A **gradient** is the slope of the loss curve at the current position.

It tells you: "if I increase w by a tiny bit, the loss will go up (positive gradient) or down (negative gradient)?"

```
Loss
│  \         /
│   \       /
│    \     /
│     \   /
│      \_/
└──────────────── w
        ↑
    gradient = 0 here (minimum)
```

At the minimum, the gradient is zero — there's no direction to go.
Left of minimum: gradient is negative (go right = increase w)
Right of minimum: gradient is positive (go left = decrease w)

**Update rule: always subtract the gradient**
```
w = w - gradient_of_w
```
Subtracting means: move opposite to the slope direction = downhill.

---

## Computing the gradient for MSE

Our loss is MSE: `L = (1/n) × Σ (w×x + b - y)²`

Using calculus (you don't need to derive this, just understand it):
```
dL/dw = (2/n) × Σ (y_pred - y_true) × x
dL/db = (2/n) × Σ (y_pred - y_true)
```

Read as:
- `dL/dw` = "how much does the loss change if I increase w slightly?"
- `dL/db` = "how much does the loss change if I increase b slightly?"

**Step by step for dL/dw:**

For each data point i:
1. Compute the prediction: `y_pred = w × x[i] + b`
2. Compute the error: `error = y_pred - y_true[i]`
3. Multiply error by `x[i]`: `error × x[i]`
4. Sum all these up: `Σ (error × x[i])`
5. Multiply by `2/n`

---

## Worked example: one step of gradient descent

Data: `x=[1,2,3], y=[40,50,60]`
Start: `w=0, b=0, learning_rate=0.1`

**Step 1: forward pass** (make predictions)
```
y_pred[0] = 0×1 + 0 = 0
y_pred[1] = 0×2 + 0 = 0
y_pred[2] = 0×3 + 0 = 0
```

**Step 2: compute errors**
```
error[0] = 0 - 40 = -40
error[1] = 0 - 50 = -50
error[2] = 0 - 60 = -60
```

**Step 3: compute gradients**
```
n = 3

dw_sum = (-40)×1 + (-50)×2 + (-60)×3
       = -40 + -100 + -180
       = -320

db_sum = -40 + -50 + -60 = -150

dL/dw = (2/3) × (-320) = -213.3
dL/db = (2/3) × (-150) = -100.0
```

**Step 4: update w and b**
```
w = w - learning_rate × dL/dw
w = 0 - 0.1 × (-213.3)
w = 0 + 21.33
w = 21.33

b = b - learning_rate × dL/db
b = 0 - 0.1 × (-100.0)
b = 0 + 10.0
b = 10.0
```

After 1 step: `w=21.33, b=10.0`.
We went from w=0 to w=21, heading toward the true value of w=10!
(It overshot, but more iterations will fix that.)

---

## The learning rate

The learning rate (`lr`) controls **how big each step is**.

```
w = w - lr × gradient
```

**Too large (e.g. lr=10):**
```
w jumps too far, overshoots the minimum, may never converge
loss: 10 → 500 → 8000 → ∞  (explodes)
```

**Too small (e.g. lr=0.00001):**
```
w barely moves, training takes millions of steps
loss: 1000 → 999.9 → 999.8 → ...  (too slow)
```

**Just right (e.g. lr=0.01):**
```
loss: 1000 → 500 → 200 → 50 → 5 → 0.2  (converges nicely)
```

For our data, `lr=0.01` works well.

---

## The training loop

```
repeat many times:
  1. for every data point: compute (y_pred - y_true) × x  → sum up
  2. multiply by 2/n  → gradient
  3. w = w - lr × gradient_w
  4. b = b - lr × gradient_b
```

Each repetition is called an **iteration** or **epoch**.
After enough iterations, w and b converge to the best values.

---

## Visualizing the loss going down

```
Iteration    w       b       Loss
0            0.000   0.000   5633.0
1000         9.823   26.413   28.4
2000         10.841  24.317    4.2
5000         11.123  24.829    1.1
10000        11.196  24.698    0.8
```

The loss starts huge (5633!) and gets smaller each iteration.
The model is "learning" — adjusting w and b to fit the data better.

---

## Why `2/n` in the gradient?

The `2` comes from differentiating `(error)²`:
```
d/dw (error²) = 2 × error × d(error)/dw
```
The `n` is just the average (same as in MSE). Some textbooks leave out the `2` — it doesn't matter because the learning rate absorbs it.

---

## The code

```ts
function step(): void {
  const n = x.length;
  let dw = 0;
  let db = 0;

  // For each data point, accumulate gradients
  for (let i = 0; i < n; i++) {
    const error = predict(x[i]) - y[i];   // how wrong are we?
    dw += error * x[i];                    // gradient for w
    db += error;                           // gradient for b
  }

  // Update: step opposite to gradient (downhill)
  w -= lr * (2 / n) * dw;
  b -= lr * (2 / n) * db;
}
```

---

## Summary

| Concept | Meaning |
|---------|---------|
| Gradient | slope of the loss — tells you which direction is uphill |
| Gradient descent | always step downhill (opposite to gradient) |
| Learning rate | step size — not too big, not too small |
| Iteration | one round of: compute gradients → update w and b |
| Convergence | when the loss stops decreasing (you reached the bottom) |

---

## Code for this lesson

See [index.ts](index.ts) — runs 10,000 gradient descent steps and prints loss at each 1,000.

## What's next
[Lesson 04 → OOP Model](../04-oop-model/lesson.md)

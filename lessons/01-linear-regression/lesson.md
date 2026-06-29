# Lesson 01 — Linear Regression

---

## What does "linear" mean?

**Linear** means the relationship between x and y forms a **straight line**.

If you plot the data on paper and a straight line goes through (or near) all the points, the relationship is linear.

```
y
│
115 ┤                              ×
100 ┤                         ×
 80 ┤                    ×
 70 ┤               ×
 60 ┤          ×
 50 ┤     ×
 40 ┤×
    └──────────────────────────── x
    1    2    3    4    5    6    7
```

You can see the points roughly go up in a straight line. That means **linear regression** will work well here.

---

## Our data

```ts
const x = [1, 2, 3, 4, 5, 6, 7];
const y = [40, 50, 60, 70, 80, 100, 115];
```

`x` could be: years of experience, hours of study, size of a house...
`y` could be: salary, exam score, house price...

Here the numbers are made up, but the idea is real.

---

## The formula

```
y_pred = w × x + b
```

This is called the **equation of a line**.

- `w` = slope = how steep the line is = how much y changes per 1 unit of x
- `b` = intercept = where the line crosses the y-axis (value of y when x=0)

---

## What does slope mean, really?

If `w = 10`:
- x goes from 1 to 2 (increases by 1)
- y goes from 40 to 50 (increases by 10)
- That means: **for every +1 in x, y goes up by 10**

If `w = 20`:
- y would increase by 20 for each +1 in x
- The line would be **steeper**

```
w = 5 (gentle slope):         w = 20 (steep slope):
   y                             y
   │     /                       │  /
   │    /                        │ /
   │   /                         │/
   └── x                         └── x
```

---

## What does intercept mean, really?

`b` is the value of `y` when `x = 0`.

```
y = w × 0 + b = b
```

If `b = 30`: the line starts at y=30 when x=0.
If `b = 0`:  the line starts at the origin (0,0).

---

## Let's try different lines on our data

**Line A: w=10, b=30**
```
x=1 → 10×1+30 = 40   true=40   error= 0
x=2 → 10×2+30 = 50   true=50   error= 0
x=3 → 10×3+30 = 60   true=60   error= 0
x=4 → 10×4+30 = 70   true=70   error= 0
x=5 → 10×5+30 = 80   true=80   error= 0
x=6 → 10×6+30 = 90   true=100  error=-10  ✗
x=7 → 10×7+30 = 100  true=115  error=-15  ✗
```

Not perfect. The data curves upward a bit at the end.

**Line B: w=13, b=25**
```
x=1 → 13×1+25 = 38   true=40   error=-2
x=2 → 13×2+25 = 51   true=50   error=+1
x=6 → 13×6+25 = 103  true=100  error=+3
x=7 → 13×7+25 = 116  true=115  error=+1
```

Better overall! Smaller errors everywhere.

**The goal of linear regression: find the w and b that minimize total error.**

---

## Why "regression"?

In statistics, "regression" means predicting a continuous number (like salary, price, temperature).
If you were predicting a category (like "cat" vs "dog"), it would be called "classification".

---

## Limitations of linear regression

Linear regression only works when the data **actually follows a straight line**.

Example of data that does NOT fit a line:
```
y
│         ×
│     ×       ×
│   ×           ×
│ ×               ×
│×                 ×
└───────────────────── x
```
This is a curve (parabola). A straight line can't fit it well.

That's why we need neural networks (Lesson 06+) for more complex patterns.

---

## The predict function

```ts
function predict(xValue: number, w: number, b: number): number {
  return w * xValue + b;
}
```

Breaking it down:
- `xValue` — the input (e.g. 5 years of experience)
- `w * xValue` — multiply weight by input
- `+ b` — add the bias
- `return` — send the result back

This one function IS the entire model. Everything else (training, loss, gradients) exists just to find the right `w` and `b` to put into this function.

---

## Making predictions beyond the data

Once trained, we can predict values we've never seen:
```
x=10 → w×10 + b  (extrapolation — predicting outside the training range)
x=3  → w×3  + b  (interpolation — predicting inside the training range)
```

Interpolation is reliable. Extrapolation can be risky — the pattern might change.

---

## Summary

| Concept | Meaning |
|---------|---------|
| Linear regression | find the best straight line through data |
| `y = w×x + b` | the formula for a straight line |
| `w` (slope) | how much y changes per unit of x |
| `b` (intercept) | value of y when x=0 |
| Goal | minimize the difference between predicted and true y |

---

## Code for this lesson

See [index.ts](index.ts) — tries different lines, shows errors for each.

## What's next
[Lesson 02 → Loss Function](../02-loss-function/lesson.md)

# Lesson 00 — What Is a Model?

---

## Before anything else: what is a function?

A function is just a **rule that takes a number in and gives a number out**.

Think of a vending machine:
- You press button `3` → you get a Coke
- You press button `5` → you get chips
- The machine is the function. The button is the input. The snack is the output.

In math we write it like this:
```
f(x) = output
```
- `f` is the name of the function (the machine)
- `x` is the input (the button you press)
- `output` is what comes out

---

## What is a model?

A model is a function that **predicts something useful**.

Example: you want to predict a person's salary from how many years of experience they have.

```
experience (years) → [model] → salary (dollars)
```

The simplest possible model is:
```
salary = w × experience + b
```

Where:
- `w` is called the **weight** (also called the slope)
- `b` is called the **bias** (also called the intercept)
- `×` means multiply

---

## What are w and b?

**w (weight/slope)** — controls how steep the line is.

```
w = 10 means: for every 1 extra year of experience, salary goes up by $10
w = 100 means: for every 1 extra year of experience, salary goes up by $100
```

**b (bias/intercept)** — controls the starting point.

```
b = 30 means: even with 0 years of experience, the salary starts at 30
b = 0  means: someone with 0 experience earns 0
```

---

## Let's see it with real numbers

Say we have this data:

| experience (x) | salary (y) |
|---------------|------------|
| 1             | 40         |
| 2             | 50         |
| 3             | 60         |
| 4             | 70         |
| 5             | 80         |

Now try the model `y = 10 × x + 30`:

```
x=1 → 10×1 + 30 = 40  ✓  (matches!)
x=2 → 10×2 + 30 = 50  ✓  (matches!)
x=3 → 10×3 + 30 = 60  ✓  (matches!)
x=4 → 10×4 + 30 = 70  ✓  (matches!)
x=5 → 10×5 + 30 = 80  ✓  (matches!)
```

The model `y = 10x + 30` perfectly fits this data. We found the right `w=10` and `b=30`.

---

## But what if we guessed wrong?

Try `w=5, b=10`:
```
x=1 → 5×1 + 10 = 15   ✗  (true answer is 40, we're off by 25)
x=2 → 5×2 + 10 = 20   ✗  (true answer is 50, we're off by 30)
```

This model is bad. We need a way to:
1. Measure **how wrong** we are (→ Lesson 02: Loss Function)
2. **Improve** w and b automatically (→ Lesson 03: Gradient Descent)

That process of improving automatically is called **training** or **learning**.

---

## So what is machine learning?

Machine learning = **automatically finding the right w and b** so the model makes good predictions.

Instead of you manually guessing w=10, b=30, the computer tries thousands of values and finds the best ones by itself.

```
start: w=0, b=0        ← wrong
after training: w=10, b=30  ← correct
```

---

## The big picture

```
Step 1: Make a prediction     → y_pred = w × x + b
Step 2: Measure the error     → how far is y_pred from true y?
Step 3: Improve w and b       → move them to reduce the error
Step 4: Repeat many times     → eventually w and b are correct
```

Every machine learning model — including Claude — follows this same 4-step loop.
The math gets more complex, but the idea never changes.

---

## Key words (remember these forever)

| Word | Meaning |
|------|---------|
| model | a function that makes predictions |
| input (x) | the information we give the model |
| output (y_pred) | what the model predicts |
| weight (w) | a number the model learns (controls slope) |
| bias (b) | another number the model learns (controls offset) |
| training | the process of finding the right w and b |
| parameters | all the numbers the model learns (w, b, and later: thousands more) |

---

## Try it yourself (no code needed)

By hand, compute `y = w × x + b` for these:

1. `w=3, b=0, x=4`  → ?
2. `w=0, b=7, x=100` → ?  (notice: w=0 means x doesn't matter at all)
3. `w=2, b=5, x=0`  → ?  (notice: b is the answer when x=0)

Answers: 12, 7, 5

---

## Code for this lesson

See [index.ts](index.ts) — run it to see predictions with different w and b values.

## What's next
[Lesson 01 → Linear Regression](../01-linear-regression/lesson.md)

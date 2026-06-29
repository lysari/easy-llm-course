# Lesson 05 — Activation Functions

---

## Why do we need activation functions?

So far our model is: `y = w × x + b`

This is a line. What if the data is not a line?

```
y                         y
│    ×   ×                │          ×
│  ×       ×              │       ×     ×
│×           ×            │    ×           ×
└──────────── x           └──────────────── x
  curve: can't fit         wave: can't fit
  with a line              with a line
```

What if we stack two layers?
```
layer 1: z = w1×x + b1
layer 2: y = w2×z + b2
```

Substituting:
```
y = w2×(w1×x + b1) + b2
y = (w2×w1)×x + (w2×b1 + b2)
y = W×x + B           ← still a line!
```

**Stacking linear layers gives you another linear layer.** Depth is useless without activation functions.

---

## What is an activation function?

An activation function is applied **after** the weighted sum. It introduces **non-linearity** — the ability to bend and curve.

```
without activation:  output = w×x + b           (straight line)
with activation:     output = σ(w×x + b)        (can be a curve)
```

The `σ` (sigma) is the activation function. Common choices:

---

## Sigmoid

```
σ(z) = 1 / (1 + e^(-z))
```

- Output range: always between **0 and 1**
- `e` is Euler's number ≈ 2.718

```
Input z  │  Output σ(z)
─────────┼─────────────
  -10    │  0.0000   (nearly 0)
   -5    │  0.0067
   -2    │  0.1192
   -1    │  0.2689
    0    │  0.5000   (exactly 0.5)
   +1    │  0.7311
   +2    │  0.8808
   +5    │  0.9933
  +10    │  1.0000   (nearly 1)
```

Shape:
```
output
1 ┤                    ──────
  │              ─────/
  │          ───/
0.5┤         /
  │     ───/
  │  ──/
0 ┤──
  └──────────────────────── z
    -5  -3  -1   0   1   3  5
```

**Good for**: output layer when you want a probability (0 to 1)
**Problem**: gradient disappears for large |z| (called "vanishing gradient" — Lesson 07)

---

## ReLU — Rectified Linear Unit

```
relu(z) = max(0, z)
```

- If z is negative: output is 0 (neuron is "off")
- If z is positive: output equals z (neuron is "on")

```
Input z  │  Output relu(z)
─────────┼────────────────
   -5    │  0
   -2    │  0
   -0.1  │  0
    0    │  0
    0.1  │  0.1
    2    │  2
    5    │  5
   10    │  10
```

Shape:
```
output
5 ┤              /
4 ┤            /
3 ┤          /
2 ┤        /
1 ┤      /
0 ┤─────/
  └──────────────── z
    -5  -2   0   2  5
```

**Good for**: hidden layers in deep networks
**Why better than sigmoid for hidden layers**:
- Gradient is either 0 or 1 (no vanishing for positive values)
- Faster to compute
- Works well in practice

---

## Tanh — Hyperbolic Tangent

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

- Output range: always between **-1 and +1**
- Like sigmoid, but centered at 0

```
output
 1 ┤                ──────
   │           ────/
   │       ───/
 0 ┤      /
   │  ───/
   │──/
-1 ┤
   └──────────────────── z
```

**Good for**: better than sigmoid in many cases because it's centered at 0 (gradients flow better)

---

## The derivative: why it matters for learning

When we use backpropagation (Lesson 07), we need the **derivative** of the activation function.

The derivative tells us: "how much does the output change if the input changes slightly?"

**Sigmoid derivative:**
```
σ'(z) = σ(z) × (1 - σ(z))
```
If we already computed `s = sigmoid(z)`, then the derivative is just `s × (1 - s)`.

**At z=0:** `sigmoid(0) = 0.5`, derivative = `0.5 × 0.5 = 0.25`
**At z=10:** `sigmoid(10) ≈ 1.0`, derivative = `1.0 × 0 = 0.0` ← nearly zero!

This is the vanishing gradient problem — at large values, sigmoid has nearly zero gradient, so learning stops.

**ReLU derivative:**
```
relu'(z) = 1  if z > 0
         = 0  if z ≤ 0
```
Always 0 or 1 — gradient never vanishes for positive inputs.

**Tanh derivative:**
```
tanh'(z) = 1 - tanh(z)²
```

---

## When to use which

| Layer | Activation | Why |
|-------|-----------|-----|
| Hidden layers (deep networks) | **ReLU** | Fast, no vanishing gradient |
| Output: probability (0 to 1) | **Sigmoid** | Output is naturally 0–1 |
| Output: yes/no (classification) | **Sigmoid** | Threshold at 0.5 |
| Output: any number | **None (linear)** | Don't restrict the range |
| Some hidden layers | **Tanh** | Better than sigmoid for hidden |

In your `polynomial.ts`, sigmoid was used for hidden layers. ReLU would train faster there.

---

## Visual comparison

```
z = -3:
  sigmoid(-3) = 0.047
  relu(-3)    = 0       ← dormant neuron
  tanh(-3)    = -0.995

z = 0:
  sigmoid(0) = 0.5
  relu(0)    = 0
  tanh(0)    = 0

z = +3:
  sigmoid(3) = 0.953
  relu(3)    = 3.0      ← passes through unchanged
  tanh(3)    = 0.995
```

---

## Code for this lesson

See [index.ts](index.ts) — implements all three activations, shows the vanishing gradient problem.

## What's next
[Lesson 06 → Neural Network (MLP)](../06-neural-network/lesson.md)

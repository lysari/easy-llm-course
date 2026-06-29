# Lesson 07 — Backpropagation

---

## The problem: many weights, one loss

In linear regression (1 layer), we had 2 parameters: `w` and `b`.
Computing their gradients was straightforward.

In a neural network, we have **hundreds or thousands of parameters** — weights for every layer.
We still need gradients for all of them.

**Backpropagation** is the algorithm that computes all these gradients **efficiently** using the chain rule from calculus.

---

## The chain rule (in plain English)

The chain rule says: if A affects B, and B affects C, then:
```
"how much does A affect C?" = "how much does A affect B?" × "how much does B affect C?"
```

In math:
```
dC/dA = dC/dB × dB/dA
```

Example: you're rolling a snowball down a hill.
- The size of the snowball affects how far it rolls.
- How far it rolls affects how much snow it picks up.
- "How much does size affect snow pickup?" = "size → distance" × "distance → snow"

---

## Our network (3 neurons, 1 input)

```
x  →  [w1,b1] → sigmoid → h1 ──┐
x  →  [w2,b2] → sigmoid → h2 ──┤  [wO1,wO2,wO3,bOut] → sigmoid → output → loss
x  →  [w3,b3] → sigmoid → h3 ──┘
```

We want: `dLoss/dw1`, `dLoss/dw2`, `dLoss/db1`, `dLoss/dwO1`, etc.
All of them. For every weight.

---

## Step 1: Forward pass — save everything

We run the network forward and **save all intermediate values**.
We'll need them during backprop.

```ts
// Forward pass — SAVE all these values
const h1 = sigmoid(w1 * xi + b1);   // save h1
const h2 = sigmoid(w2 * xi + b2);   // save h2
const h3 = sigmoid(w3 * xi + b3);   // save h3
const out = sigmoid(wO1*h1 + wO2*h2 + wO3*h3 + bOut);  // save out

const error = yi - out;
const loss = error ** 2;
```

---

## Step 2: Backward pass — start from the loss and go backwards

**At the output neuron:**

The loss is `L = (yi - out)²`

How does `out` affect the loss?
```
dL/d(out) = -2 × (yi - out)    [derivative of (yi - out)²]
```

How does the pre-activation `z_out = wO1*h1 + wO2*h2 + wO3*h3 + bOut` affect `out`?
```
d(out)/d(z_out) = sigmoid'(out) = out × (1 - out)
```

Combining (chain rule):
```
dL/d(z_out) = dL/d(out) × d(out)/d(z_out)
            = -2 × (yi - out) × out × (1 - out)
```

We'll call this `δ_out` (delta out). In code:
```ts
const error = yi - out;
const δ_out = error * sigmoidDerivative(out);
// Note: the -2 is absorbed into the learning rate direction
```

---

## Step 3: Gradients for output weights

How does `wO1` affect `z_out`?
```
z_out = wO1×h1 + wO2×h2 + ...
d(z_out)/d(wO1) = h1
```

So:
```
dL/d(wO1) = δ_out × h1
dL/d(wO2) = δ_out × h2
dL/d(wO3) = δ_out × h3
dL/d(bOut) = δ_out × 1
```

Update rules:
```ts
wO1 += lr × δ_out × h1;
wO2 += lr × δ_out × h2;
wO3 += lr × δ_out × h3;
bOut += lr × δ_out;
```

(Note: `+=` because error = `yi - out` already flipped the sign)

---

## Step 4: Continue backwards to hidden layer

Now we need `dL/dw1`. The error flows back through `wO1`:

```
dL/d(h1) = δ_out × wO1
           ↑ how much output error cares about h1

d(h1)/d(z1) = sigmoid'(h1) = h1 × (1 - h1)
              ↑ sigmoid squashes the gradient

dL/d(z1) = dL/d(h1) × d(h1)/d(z1)
          = (δ_out × wO1) × (h1 × (1 - h1))
```

Call this `δH1`:
```ts
const δH1 = δ_out * wO1 * sigmoidDerivative(h1);
const δH2 = δ_out * wO2 * sigmoidDerivative(h2);
const δH3 = δ_out * wO3 * sigmoidDerivative(h3);
```

Then update hidden layer weights:
```ts
// z1 = w1 × xi + b1, so d(z1)/dw1 = xi, d(z1)/db1 = 1
w1 += lr * δH1 * xi;
b1 += lr * δH1;

w2 += lr * δH2 * xi;
b2 += lr * δH2;

w3 += lr * δH3 * xi;
b3 += lr * δH3;
```

---

## The full backward pass, all at once

```ts
function backward(xi, yi, h1, h2, h3, out) {
  // Output layer
  const error = yi - out;
  const δOut = error * sigmoidDerivative(out);

  wO1 += lr * δOut * h1;
  wO2 += lr * δOut * h2;
  wO3 += lr * δOut * h3;
  bOut += lr * δOut;

  // Hidden layer (gradient flows back through wO)
  const δH1 = δOut * wO1 * sigmoidDerivative(h1);
  const δH2 = δOut * wO2 * sigmoidDerivative(h2);
  const δH3 = δOut * wO3 * sigmoidDerivative(h3);

  w1 += lr * δH1 * xi;  b1 += lr * δH1;
  w2 += lr * δH2 * xi;  b2 += lr * δH2;
  w3 += lr * δH3 * xi;  b3 += lr * δH3;
}
```

---

## Vanishing gradients

Look at the hidden layer gradient:
```
δH1 = δOut × wO1 × sigmoidDerivative(h1)
```

`sigmoidDerivative(h1)` is at most **0.25** (when h1=0.5).
For large or small values of h1, it's nearly 0.

If you have 5 layers:
```
gradient × 0.25 × 0.25 × 0.25 × 0.25 × 0.25 = gradient × 0.001
```

By the time the gradient reaches the first layer, it's **1000× smaller**. The first layers barely learn.

This is why **ReLU** was invented — its derivative is either 0 or 1, so gradients don't shrink as they flow backward.

---

## What "backpropagation" means

"Back" — gradients flow backwards (from output toward input)
"Propagation" — each layer propagates the gradient to the layer before it

It's essentially: **the chain rule, applied layer by layer, starting from the loss.**

The term was coined in a famous 1986 paper. It's what made deep learning possible.

---

## Summary

```
Forward pass:  x → [layer1] → [layer2] → output → compute loss
                               ↓ save all intermediate values

Backward pass: loss → ∂L/∂out → ∂L/∂layer2_weights
                              ↓ chain rule
                    ∂L/∂layer2_output → ∂L/∂layer1_weights
```

Every weight gets its gradient. Every gradient gets used for an update.

---

## Code for this lesson

See [index.ts](index.ts) — full forward + backward + training loop. Watch the error shrink.

## What's next
[Lesson 08 → Matrix Math](../08-matrix-math/lesson.md)

# Lesson 02 — Calculus and Gradients

---

## The problem calculus solves

Your neural network has a million knobs (weights). The loss is 2.47. You want it lower.

Which knobs do you turn? By how much? In which direction?

Trying knobs at random would take longer than the universe has. Calculus answers the question directly: for **every** knob simultaneously, it tells you *"if you nudge this knob up a tiny bit, the loss will change by this much."* That answer is called the **gradient**, and computing it efficiently (backpropagation) is the engine of all deep learning.

You already *used* backprop in the from-scratch track ([../../lessons/07-backpropagation/lesson.md](../../lessons/07-backpropagation/lesson.md)). This lesson gives you the researcher's view: what derivatives really are, why the chain rule works, what Jacobians and Hessians tell you, and the single most useful debugging tool in empirical ML — **gradient checking**.

---

## Derivative = sensitivity

Forget "slope of tangent line" for a second. The working definition:

> **The derivative measures how sensitive the output is to a tiny nudge of the input.**

Take `f(x) = x²` at `x = 3`, so `f(3) = 9`. Nudge x by a tiny `h = 0.001`:

```
f(3.001) = 9.006001
change in output = 0.006001
change in input  = 0.001
ratio            = 6.001  ≈ 6
```

Shrink the nudge and the ratio settles on exactly 6. That limit is the derivative:

```
f'(x) = lim as h→0 of  [f(x + h) − f(x)] / h
```
- `f'(x)`: the derivative of f at x (also written df/dx)
- `h`: the nudge size
- `f(x+h) − f(x)`: how much the output moved
- dividing by `h`: converts to "output change *per unit* of input change"

For `f(x) = x²`, the rule is `f'(x) = 2x`, so `f'(3) = 6`. Interpretation: *near x=3, the output moves 6× as fast as the input.* If x is a weight and f is the loss, you now know this knob matters, and which way to turn it (loss grows when x grows → turn it down).

The handful of rules you'll actually use:

```
f(x) = c        → f'(x) = 0          (constants don't respond to nudges)
f(x) = x        → f'(x) = 1
f(x) = xⁿ       → f'(x) = n·xⁿ⁻¹
f(x) = eˣ       → f'(x) = eˣ         (its own derivative — why e is special)
f(x) = ln(x)    → f'(x) = 1/x
f(x) = a·g(x)   → f'(x) = a·g'(x)    (scaling passes through)
f = g + h       → f' = g' + h'       (sums pass through)
```

---

## Partial derivatives: one knob at a time

Real losses depend on many inputs. A **partial derivative** nudges one input while freezing the rest:

```
f(x, y) = x²y + y³

∂f/∂x = 2xy        (treat y as a constant, differentiate w.r.t. x)
∂f/∂y = x² + 3y²   (treat x as a constant)
```
- `∂` ("partial"): same idea as d, but signals "other variables are held fixed"
- `∂f/∂x`: sensitivity of f to x alone

At the point `(x=2, y=3)`:

```
f(2,3)   = 4·3 + 27 = 39
∂f/∂x    = 2·2·3    = 12   → nudging x is felt 12×
∂f/∂y    = 4 + 27   = 31   → nudging y is felt 31×  (y matters more here)
```

---

## The gradient: all sensitivities, bundled into an arrow

Stack every partial derivative into one vector:

```
∇f = [∂f/∂x, ∂f/∂y] = [12, 31]   at (2,3)
```
- `∇` ("nabla" or "del"): the gradient operator
- `∇f`: a vector with one entry per input — the full sensitivity report

The gradient has a magical geometric property:

> **∇f points in the direction of steepest ascent** — the single direction, out of all possible directions, in which f increases fastest. Its length says how steep that climb is.

Everyday analogy: standing on a foggy hillside, you can't see the summit, but you *can* feel the slope under your feet. The gradient is that feeling, and `−∇f` (downhill) is the direction gradient descent steps:

```
w ← w − η·∇L(w)
```
- `w`: all the weights, as one giant vector
- `L`: the loss
- `η` (eta): learning rate — how big a step to take
- `−∇L`: steepest descent direction

That one line is [../../lessons/03-gradient-descent/lesson.md](../../lessons/03-gradient-descent/lesson.md), and (with momentum and per-weight scaling bolted on) it's also Adam, which trains every modern LLM.

---

## The chain rule: the engine of backprop

Networks are functions of functions of functions:

```
loss = L( softmax( W₂ · relu( W₁·x ) ) )
```

How does the loss respond to a nudge in W₁, buried three layers deep? The chain rule:

```
If  y = f(u)  and  u = g(x),   then   dy/dx = (dy/du) · (du/dx)
```

**Sensitivities multiply along the chain.** Everyday version: gas pedal → engine RPM → wheel speed. If 1° of pedal adds 100 RPM, and 100 RPM adds 2 km/h, then 1° of pedal adds 2 km/h. You never needed to know how the engine works internally — just the two local sensitivities.

Concrete with numbers:

```
u = x²,  y = 3u + 1.   At x = 2:  u = 4, y = 13.

du/dx = 2x  = 4     (x nudge → u feels it 4×)
dy/du = 3           (u nudge → y feels it 3×)
dy/dx = 3 · 4 = 12  (x nudge → y feels it 12×)

check: y = 3x² + 1 → dy/dx = 6x = 12 ✓
```

Backpropagation is *just* this, organized: start at the loss, walk backward through the network, multiplying local sensitivities layer by layer. One backward pass gets the gradient for **every** weight at once — a million knobs, priced in roughly the cost of two forward passes. This efficiency is the entire reason deep learning is computationally possible.

---

## Jacobian and Hessian: the shape of the response

**Jacobian** — when the *output* is also a vector. A layer maps m inputs to n outputs; every output has a sensitivity to every input, giving an n×m grid:

```
Jᵢⱼ = ∂(output i)/∂(input j)
```

The Jacobian is the best *linear approximation* of the layer near the current point — the matrix-machine (Lesson 01) it locally behaves like. Backprop through a layer = multiplying by its Jacobian transposed. And the chain rule for vectors is literally **matmul of Jacobians** — which is why vanishing/exploding gradients are an eigenvalue story: multiply many Jacobians whose stretch factors are < 1 and the gradient dies; > 1 and it blows up.

**Hessian** — the derivative *of the gradient*: an n×n matrix of second derivatives.

```
Hᵢⱼ = ∂²L / ∂wᵢ∂wⱼ
```

The gradient gives slope; the Hessian gives **curvature** — is the landscape a sharp ravine or a wide gentle bowl?

```
sharp ravine (big eigenvalues)      flat bowl (small eigenvalues)
   \        /                        _
    \      /                     ___/ \___
     \    /                   __/         \__
      \__/
  small steps only!            big steps fine
```

Why researchers care: the Hessian's eigenvalues bound the largest safe learning rate; "flat minima generalize better" is an ongoing research debate framed entirely in Hessian language; and second-order optimizers (and their cheap cousins like Adam's per-weight scaling) are attempts to see curvature without paying for the full n×n matrix — impossibly big when n is billions.

---

## Gradient checking: the researcher's lie detector

Here's the tool this lesson exists to give you.

There are two independent ways to compute a gradient:

**1. Analytic** (backprop): fast, exact, used for training — but it's code *you wrote*, and hand-derived gradients are the classic source of silent bugs. The model still trains (badly). Loss still falls (slowly). Nothing crashes. You just quietly get wrong research results.

**2. Numeric** (finite differences): dumb, slow, nearly impossible to get wrong. Just nudge each weight and re-measure the loss — the definition of a derivative, executed literally. The **centered** version is much more accurate:

```
∂L/∂wᵢ ≈ [L(w + h·eᵢ) − L(w − h·eᵢ)] / (2h)
```
- `eᵢ`: a vector that's all zeros except a 1 at position i ("nudge only knob i")
- `h`: small nudge, typically 1e-4 or 1e-5
- centered (±h, divide by 2h) has error ~h², vs ~h for the one-sided version

**Gradient check** = compute both, compare with *relative* error (absolute error is meaningless when gradients themselves are tiny or huge):

```
rel_error = |analytic − numeric| / max(|analytic| + |numeric|, ε)
```
- `ε` (e.g. 1e-8): floor to avoid dividing by zero when both are ~0

Reading the result:

```
rel_error < 1e-6   your backprop is almost certainly correct
1e-6 … 1e-4        suspicious — okay for kinks (relu at 0), else look closer
> 1e-2             you have a bug. Full stop.
```

Numeric gradients cost one forward pass *per weight per side* — hopeless for training, perfect for verifying a handful of weights. The professional habit: **every time you hand-implement a gradient, gradient-check it before trusting a single experiment built on it.** Frameworks like PyTorch ship this as `torch.autograd.gradcheck`; today you build your own.

---

## Symbol cheat sheet

```
f'(x), df/dx     derivative — sensitivity of f to x
∂f/∂x            partial derivative — sensitivity, others frozen
∇f               gradient — vector of all partials; steepest ascent
∇²f, H           Hessian — curvature matrix
J                Jacobian — per-output-per-input sensitivity grid
η                learning rate
argmin L(w)      "the w that makes L smallest" — what training seeks
w ← w − η∇L      one step of gradient descent
```

---

## Code for this lesson

See [index.ts](index.ts) — computes numeric vs analytic derivatives of a small function, then hand-implements backprop for a tiny 2-layer network and gradient-checks every weight against centered finite differences, printing the max relative error.

## What's next
[Lesson 03 → Probability and Statistics](../03-probability-and-statistics/lesson.md)

# Lesson 06 — Neural Network (MLP)

---

## Why one layer isn't enough

Linear regression (`y = wx + b`) can only draw a straight line.
One layer with a sigmoid can draw an S-curve.

But real data is complicated: language, images, sounds. A single layer can never learn those patterns.

**A multi-layer network (MLP) stacks multiple layers.** Each layer transforms the data, and deeper layers learn more complex patterns.

---

## What is a neuron?

A single **neuron** does three things:
1. Takes multiple inputs
2. Multiplies each input by its own weight, adds a bias
3. Passes the result through an activation function

```
inputs → [weighted sum + bias] → [activation] → output
```

Example: a neuron with 3 inputs:
```
inputs:   x1=1.0, x2=0.5, x3=2.0
weights:  w1=0.3, w2=-0.1, w3=0.5
bias:     b=0.2

z = w1×x1 + w2×x2 + w3×x3 + b
  = 0.3×1.0 + (-0.1)×0.5 + 0.5×2.0 + 0.2
  = 0.3 + (-0.05) + 1.0 + 0.2
  = 1.45

output = relu(1.45) = 1.45  (relu passes positive values through)
```

---

## What is a layer?

A **layer** is a group of neurons that all receive the same inputs but have different weights.

```
           ┌─ neuron 1 (w11,w12,b1) → h1
input [x1, x2] ─┤─ neuron 2 (w21,w22,b2) → h2
           └─ neuron 3 (w31,w32,b3) → h3
```

This is called a **hidden layer** with 3 neurons.
`h1, h2, h3` are called **hidden activations** or **hidden states**.

Why "hidden"? Because they're in the middle — not the input, not the output.

---

## What does each neuron learn?

Each neuron learns to detect a **different feature** of the input.

Imagine recognizing if a number is "large":
- Neuron 1 might fire when x > 5
- Neuron 2 might fire when x is between 2 and 4
- Neuron 3 might fire when x is exactly 1

The output layer combines these signals to make the final prediction.

---

## The full architecture

```
input layer    hidden layer (3 neurons)   output layer (1 neuron)

               ┌──── h1 = relu(w1*x + b1)
               │                           output = sigmoid(wO1*h1 + wO2*h2 + wO3*h3 + bOut)
x ─────────────┤──── h2 = relu(w2*x + b2) ──────────────────────────────────────────────────► y
               │
               └──── h3 = relu(w3*x + b3)
```

This is exactly what `polynomial.ts` does — but with `sigmoid` instead of `relu` for hidden neurons.

---

## From one input to vector input

So far our input `x` was a single number. Real problems have **many features**:

```
House price prediction:
  x = [size_sqm, num_bedrooms, distance_to_city, age_years]
  x is a vector (list of numbers), not one number
```

A neuron with vector input:
```
inputs:  x = [x1, x2, x3, x4]
weights: w = [w1, w2, w3, w4]  (one weight per input)

z = w1×x1 + w2×x2 + w3×x3 + w4×x4 + b
  = Σ(w_i × x_i) + b    ← sum over all inputs
```

This sum `Σ(w_i × x_i)` is called a **dot product** (Lesson 08 will formalize this with matrices).

---

## The forward pass

"Forward pass" = running the input through the network, layer by layer, to get the output.

```
Step 1: Input
  x = [1.0, 0.5]   (2 features)

Step 2: Hidden layer (4 neurons, each with 2 weights)
  h1 = relu( w11×x1 + w12×x2 + b1 )
  h2 = relu( w21×x1 + w22×x2 + b2 )
  h3 = relu( w31×x1 + w32×x2 + b3 )
  h4 = relu( w41×x1 + w42×x2 + b4 )

Step 3: Output layer (1 neuron, 4 inputs from hidden layer)
  y = w_out1×h1 + w_out2×h2 + w_out3×h3 + w_out4×h4 + b_out
```

---

## Why more neurons = more expressive?

Imagine approximating a curve with neurons:

```
1 neuron:  can make one "bend"
3 neurons: can make three bends, fit more complex curves
10 neurons: can fit even more complex shapes
100 neurons: can approximate almost any function
```

The famous **Universal Approximation Theorem** says: a neural network with enough neurons in one hidden layer can approximate any continuous function to any desired accuracy.

This is why neural networks are so powerful.

---

## Counting parameters

Every weight and bias is a **parameter** — a number the model learns.

Example: 1 input → 3 hidden neurons → 1 output:
```
Hidden layer: 3 neurons × (1 weight + 1 bias) = 6 parameters
Output layer: 1 neuron × (3 weights + 1 bias) = 4 parameters
Total: 10 parameters
```

For vector input of size 4 → 3 hidden neurons → 1 output:
```
Hidden layer: 3 neurons × (4 weights + 1 bias) = 15 parameters
Output layer: 1 neuron  × (3 weights + 1 bias) = 4 parameters
Total: 19 parameters
```

Claude has **hundreds of billions** of parameters. Same idea, massive scale.

---

## One neuron vs one layer (code comparison)

```ts
// One neuron (one set of weights):
function neuron(inputs: number[], weights: number[], bias: number): number {
  const z = inputs.reduce((sum, xi, i) => sum + xi * weights[i], bias);
  return relu(z);
}

// One layer (multiple neurons in parallel):
function layer(inputs: number[], allWeights: number[][], biases: number[]): number[] {
  return allWeights.map((weights, i) => neuron(inputs, weights, biases[i]));
  //     ↑ returns an array — one output per neuron
}
```

A layer is just: "run each neuron with the same inputs, collect all outputs".

---

## Code for this lesson

See [index.ts](index.ts) — MLP with vector input (2 features), 4 hidden neurons, 1 output.

## What's next
[Lesson 07 → Backpropagation](../07-backpropagation/lesson.md)

# Lesson 15 — Feed-Forward Network (the other half of a transformer)

---

## What you built in Lesson 14 — and what's missing

Lesson 14's tiny GPT has a working transformer block, but it is only half complete.

Here is what it actually does:

```
X (token embeddings)
  │
  ▼
[Causal Self-Attention]   ← tokens communicate with each other
  │
  ▼
[Residual + LayerNorm]    ← X2 = LayerNorm(X + attnOut)
  │
  ▼
[Linear projection → logits]   ← Wproj maps X2 to vocab
```

What a real transformer block looks like:

```
X (token embeddings)
  │
  ▼
[Causal Self-Attention]
  │
  ▼
[Residual + LayerNorm]    ← X2 = LayerNorm(X + attnOut)
  │
  ▼
[Feed-Forward Network]    ← MISSING in lesson 14
  │
  ▼
[Residual + LayerNorm]    ← X3 = LayerNorm(X2 + ffnOut)
  │
  ▼
[Linear projection → logits]
```

The Feed-Forward Network (FFN) is the piece you are adding in this lesson.

---

## What the FFN does

After attention, all tokens have had the chance to "talk to each other" — each position now has a representation that mixes in context from other positions.

But that mixed representation is still a linear combination of what was there before. The FFN is where the model does its **nonlinear thinking**.

Think of it this way:

- **Attention** = communication. Tokens share information with each other.
- **FFN** = computation. Each token independently processes its own updated representation.

The FFN is applied to each token **independently** — there is no cross-token interaction here. It is just a small, two-layer MLP applied once per position.

---

## The math

```
FFN(x) = W2 · GELU(W1 · x + b1) + b2
```

Where:
- `x` is a single token's embedding vector, shape `[embedDim]`
- `W1` is shape `[4*embedDim × embedDim]` — expands into a wider hidden layer
- `b1` is shape `[4*embedDim]`
- `W2` is shape `[embedDim × 4*embedDim]` — compresses back down
- `b2` is shape `[embedDim]`
- `GELU` is applied elementwise to the hidden layer

Step by step for a single token `x`:

```
Step 1 — expand:   pre = W1 · x + b1        shape: [4*embedDim]
Step 2 — activate: h   = GELU(pre)           shape: [4*embedDim]
Step 3 — compress: out = W2 · h + b2         shape: [embedDim]
```

The output has the same shape as the input (`[embedDim]`), so it slots cleanly into the residual stream.

---

## The 4x expansion ratio

Why does W1 expand to `4 * embedDim` before W2 compresses back?

If you used the same dimension throughout:
```
W1: [embedDim × embedDim]  →  W2: [embedDim × embedDim]
```
That would be equivalent (with a nonlinearity) to just one matrix. The model would have limited capacity to represent complex functions.

The 4x expansion gives the FFN a large "working space" where it can compute more complex functions before projecting back:

```
embedDim = 16:
  W1 expands  16 → 64   (the model has 64 numbers to work with)
  W2 compresses 64 → 16 (distills those 64 numbers back into the residual)
```

This expansion-compression pattern is called a **bottleneck MLP** or **projection MLP**.

**Parameter count for FFN:**
```
W1: embedDim × 4*embedDim  = 16 × 64 = 1,024
b1: 4*embedDim             = 64
W2: 4*embedDim × embedDim  = 64 × 16 = 1,024
b2: embedDim               = 16
Total per layer: 2,128 parameters
```

In GPT-2 (embedDim = 768):
```
W1: 768 × 3072 = 2,359,296
W2: 3072 × 768 = 2,359,296
Total FFN per layer: ~4.7 million parameters
```

Across 12 layers, the FFN alone accounts for more than half of GPT-2's 117 million parameters.

---

## GELU: the activation function

GPT-2 and all modern transformers use **GELU** (Gaussian Error Linear Unit) instead of ReLU.

### ReLU (what you know)

```
ReLU(x) = max(0, x)
```

Simple, fast, but has a hard cutoff at zero. Neurons with input < 0 produce exactly 0 — they are "dead" for that input and pass no gradient back.

### GELU

```
GELU(x) = x · 0.5 · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

GELU is a smooth approximation to ReLU. Instead of a hard zero-cutoff, it gently gates the input by the probability that a Gaussian random variable is less than x.

Visually:

```
       GELU vs ReLU
  
  1.5 │              /
      │             /
  1.0 │            /
      │           /   ← both functions agree here
  0.5 │          /
      │         /
  0.0 │────────/────────
      │   (ReLU: hard zero)
 -0.2 │  ↘
      │   ↘__ (GELU: small negative bump)
 -0.5 │
       ─────────────────
      -2   -1    0    1    2
```

The key difference: GELU allows **small negative values** to pass through instead of clamping to zero. This means:
1. Gradients flow more smoothly during training
2. Neurons are never completely "dead"
3. The model can learn to use negative activations as a signal

### Comparison: GPT-1 vs GPT-2+

| Property | GPT-1 (2018) | GPT-2+ (2019–present) |
|---|---|---|
| Activation | ReLU | GELU |
| Gradient flow | Hard cutoff at 0 | Smooth near 0 |
| Dead neurons | Yes (for x < 0) | Rare |
| Compute cost | Fast | ~same (tanh is cheap) |
| Used in Claude | No | Yes |
| Training stability | Good | Better |

GELU became the standard in GPT-2 and has remained the default ever since. Most modern architectures (BERT, RoBERTa, GPT-3, GPT-4, Claude) use GELU or a variant called SwiGLU.

---

## The complete transformer block

Here is what a full transformer block looks like with both pieces:

```
Input: X  (shape: [T × embedDim], T = sequence length)
  │
  ├── (save X for residual)
  │
  ▼
[LayerNorm]       (pre-norm style, used in GPT-2)
  │
  ▼
[Causal Self-Attention]
  Q = X · Wq
  K = X · Wk
  V = X · Wv
  scores = Q · K^T / sqrt(embedDim)   + causal mask
  attnWeights = softmax(scores)
  attnOut = attnWeights · V
  │
  ▼
[Residual]        X1 = X + attnOut
  │
  ▼
[LayerNorm]
  │
  ▼
[FFN]             for each token t independently:
  pre = W1 · X1[t] + b1         expand to 4*embedDim
  h   = GELU(pre)
  out = W2 · h + b2              compress back to embedDim
  │
  ▼
[Residual]        X2 = X1 + ffnOut
  │
Output: X2  (same shape as input)
```

This block can be stacked. Each layer has its own Wq, Wk, Wv, W1, W2, etc. The output of one block is the input to the next.

Note: lesson 14 uses **post-norm** (LayerNorm after residual). Real GPT-2 uses **pre-norm** (LayerNorm before attention/FFN). The code in this lesson uses post-norm for simplicity — the structure is the same, the placement of LayerNorm differs.

---

## What the FFN actually "stores"

Research suggests the FFN layers act as a **key-value memory store**:

- The rows of W1 are "keys" — patterns the model learned to detect
- The columns of W2 are "values" — what to output when a key fires
- GELU selects which keys are active for a given input

This is why large models can "recall" facts: the FFN weights memorize associations during training, and the attention layers route the right information to the right FFN neurons at inference time.

The larger the FFN (larger `embedDim`, more layers), the more facts the model can store.

---

## Difference vs Lesson 14

| | Lesson 14 | Lesson 15 |
|---|---|---|
| Attention | Yes | Yes |
| FFN | No | Yes |
| Residual connections | 1 (after attention) | 2 (after attention, after FFN) |
| Parameters trained | Wproj + embTable | Wproj + embTable + W1 + W2 + b1 + b2 |
| Transformer completeness | Half | Full |
| GELU | No | Yes |

Adding the FFN does not change the overall flow — the model still takes tokens, embeds them, passes through a transformer block, and projects to logits. But now the transformer block is complete, matching the architecture of GPT-1 and GPT-2.

---

## Backpropagation through the FFN

The FFN is a two-layer MLP. You derived backprop for MLPs in Lesson 07.

Given:
```
pre  = W1 · x + b1          (linear)
h    = GELU(pre)             (elementwise activation)
out  = W2 · h + b2           (linear)
```

Given `dout` (gradient flowing back from the residual + LayerNorm above):

```
dL/dW2    = dout^T · h       (outer product, shape: [embedDim × 4*embedDim])
dL/db2    = dout

dL/dh     = dout · W2^T      (shape: [4*embedDim])
dL/dpre   = dL/dh ⊙ GELU'(pre)   (elementwise, ⊙ = Hadamard product)

dL/dW1    = dL/dpre^T · x    (outer product, shape: [4*embedDim × embedDim])
dL/db1    = dL/dpre

dL/dx     = W1^T · dL/dpre   (gradient for the input, passed back to attention)
```

GELU gradient:

```
GELU'(x) ≈ 0.5 · tanh(c·(x + 0.044715·x³))
           + 0.5 · x · (1 - tanh²(c·(x + 0.044715·x³))) · c · (1 + 3·0.044715·x²)

where c = sqrt(2/π) ≈ 0.7979
```

This is the full chain rule applied to the GELU formula.

---

## What you built

In this lesson you extended the tiny GPT to include a complete Feed-Forward Network:

```
Lesson 00: f(x) = wx + b                   ← one parameter
Lesson 01: linear regression               ← find the best line
Lesson 02: loss function (MSE)             ← measure wrongness
Lesson 03: gradient descent                ← move downhill
Lesson 04: OOP model                       ← clean code
Lesson 05: activation functions            ← sigmoid, relu, tanh
Lesson 06: MLP                             ← stacked layers
Lesson 07: backpropagation                 ← chain rule
Lesson 08: matrix math                     ← matmul is everything
Lesson 09: tokenization                    ← text → integers
Lesson 10: embeddings                      ← integers → vectors
Lesson 11: softmax                         ← logits → probabilities
Lesson 12: attention                       ← tokens communicate
Lesson 13: transformer block               ← attention + FFN structure
Lesson 14: tiny GPT (attention only)       ← half a transformer
Lesson 15: tiny GPT + FFN                  ← a COMPLETE transformer block ✓
```

You now have the full transformer: embedding + positional encoding + causal self-attention + feed-forward network + residual connections + layer normalization + projection to logits.

Every modern LLM — GPT-2, GPT-3, GPT-4, Claude, LLaMA — is this structure repeated many times at a much larger scale.

---

## Code for this lesson

See [index.ts](index.ts) — lesson 14's tiny GPT extended with the complete FFN.

# Lesson 13 — Transformer Block

---

## What is a transformer block?

A **transformer block** is the repeating building unit of every LLM.

GPT-2 (small): 12 blocks stacked
GPT-3: 96 blocks stacked
Claude: ~100+ blocks stacked

Each block takes a matrix of shape `(T × d)` and outputs a matrix of the same shape `(T × d)`.
The shape never changes — blocks transform, not reshape.

---

## The two sublayers inside every block

Every transformer block has exactly two sublayers:

```
Input X  (T × d)
  │
  ├── [Sublayer 1] LayerNorm → Multi-Head Attention → + residual
  │
  └── [Sublayer 2] LayerNorm → Feed-Forward Network → + residual
  │
Output X  (T × d)
```

---

## Component 1: Layer Normalization

**Why do we need it?**

As a signal passes through many layers, the values can become very large or very small.
This makes training unstable — gradients explode or vanish.

Layer normalization **standardizes** each token's representation to have mean=0 and std=1.

**Formula:**
```
LayerNorm(x) = (x - mean(x)) / sqrt(variance(x) + ε) × γ + β
```

Where:
- `x` — one token's embedding vector (length d)
- `mean(x)` — average of all d values
- `variance(x)` — how spread out the values are
- `ε` — tiny number (1e-5) to prevent division by zero
- `γ, β` — learnable scale and shift (initialized to 1 and 0)

**Step by step:**
```
x = [2.0, 4.0, 6.0, 8.0]

mean(x) = (2+4+6+8)/4 = 5.0

variance(x) = ((2-5)² + (4-5)² + (6-5)² + (8-5)²) / 4
            = (9 + 1 + 1 + 9) / 4 = 5.0

normalized = (x - 5.0) / sqrt(5.0 + 1e-5)
           = [-3, -1, 1, 3] / 2.236
           = [-1.342, -0.447, 0.447, 1.342]

(mean is now ~0, std is now ~1)
```

**Note**: LayerNorm is applied per token (each row of the matrix independently), not across tokens.

---

## Component 2: Multi-Head Attention

You learned single-head attention in Lesson 12.

**Multi-head attention** runs the attention operation **H times in parallel**, each with different weight matrices.

```
Input X (T × d)
  │
  ├── Head 1: attention(X@Wq1, X@Wk1, X@Wv1)  → out1 (T × d/H)
  ├── Head 2: attention(X@Wq2, X@Wk2, X@Wv2)  → out2 (T × d/H)
  │   ...
  └── Head H: attention(X@WqH, X@WkH, X@WvH)  → outH (T × d/H)
  │
  concat([out1, out2, ..., outH])   → (T × d)
  │
  @ Wo (output projection)           → (T × d)
```

Each head operates in a smaller dimension `d/H`, so total computation stays the same.

**Why multiple heads?**
Each head can specialize:
- Head 1: tracks pronouns
- Head 2: tracks syntax
- Head 3: tracks topic
- Head 4: tracks distance

A single head would have to do all of these at once, which is harder.

---

## Component 3: Feed-Forward Network (FFN)

The FFN is applied **independently to each token** after attention.

```
ffn(x) = relu(x @ W1 + b1) @ W2 + b2
```

Dimensions:
- `x`: (d,) — one token
- `W1`: (d × 4d) — expands to 4× wider
- `W2`: (4d × d) — compresses back
- Output: (d,) — same size as input

Why 4× expansion? Empirically found to work well. It gives the FFN capacity to store patterns.

The FFN acts like the model's **memory** — it stores facts and patterns learned during training.
Attention decides what to look at; FFN processes it.

---

## Component 4: Residual Connections

After each sublayer, the input is **added back** to the output:
```
X = X + attention(X)
X = X + ffn(X)
```

This is called a **residual connection** (or skip connection).

**Why?**

Without residual:
```
layer 1 → layer 2 → layer 3 → ... → layer 100
gradient must travel backward through all 100 layers → vanishes
```

With residual:
```
X → layer → X + delta

Gradient can flow directly backward through the + sign (gradient of x+f(x) w.r.t. x = 1 + grad_of_f)
```

The `1` in that expression means there's always a direct gradient path to every layer.
This is what makes training 100+ layers possible.

**Intuition**: each layer adds a small "correction" to the input, rather than transforming it completely. The residual path carries the original signal unmodified alongside the correction.

---

## Putting it all together: one block

```ts
function transformerBlock(X: number[][]): number[][] {
  // --- Sublayer 1: Self-Attention ---

  // 1a. Layer normalize each token
  const normed1 = X.map(row => layerNorm(row));

  // 1b. Project to Q, K, V and compute attention
  const Q = matmul(normed1, Wq);
  const K = matmul(normed1, Wk);
  const V = matmul(normed1, Wv);
  const attnOut = attention(Q, K, V, causal=true);

  // 1c. Residual connection: X = X + attention_output
  const X2 = X.map((row, i) =>
    row.map((v, j) => v + attnOut[i][j])
  );

  // --- Sublayer 2: Feed-Forward Network ---

  // 2a. Layer normalize each token
  const normed2 = X2.map(row => layerNorm(row));

  // 2b. FFN: expand → relu → compress
  const ffnOut = normed2.map(row => ffn(row, W1, b1, W2, b2));

  // 2c. Residual connection: X = X + ffn_output
  const X3 = X2.map((row, i) =>
    row.map((v, j) => v + ffnOut[i][j])
  );

  return X3;
}
```

---

## The full transformer (stacking N blocks)

```
text
  → tokenize
  → embed (vocab lookup)
  → + positional encoding
  → block 1
  → block 2
  → ...
  → block N
  → layer norm
  → project to vocab logits (T × vocab_size)
  → softmax
  → sample next token
```

Each block refines the representation. Early blocks handle simple patterns (punctuation, common words). Later blocks handle complex reasoning (coreference, long-range dependencies).

---

## Parameter count for one block

For `d=512`, `num_heads=8`, `ffn_dim=2048`:

```
LayerNorm 1:          γ + β = 2d = 1,024
Attention Q,K,V,Wo:   4 × (d × d) = 4 × 262,144 = 1,048,576
LayerNorm 2:          2d = 1,024
FFN W1:               d × 4d = 1,048,576
FFN W2:               4d × d = 1,048,576
FFN biases:           4d + d = 2,560
                      ─────────────────────
Total per block:      ≈ 3.15 million parameters
```

GPT-3 has 96 such blocks → ~300 million parameters just in blocks.
(Plus embeddings, total is 175 billion parameters.)

---

## Code for this lesson

See [index.ts](index.ts) — full transformer block: LayerNorm + attention + FFN + residuals.

## What's next
[Lesson 14 → Tiny GPT](../14-tiny-gpt/lesson.md)

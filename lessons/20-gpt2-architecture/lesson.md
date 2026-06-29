# Lesson 20 — GPT-2 Architecture: Pre-LN and Weight Tying

GPT-2 (Radford et al., 2019) is GPT-1 with two structural changes and a lot more scale.
The two changes are small on paper but large in effect: **Pre-LayerNorm** and **weight tying**.

---

## What stayed the same from GPT-1

- Causal (decoder-only) transformer
- Token embeddings + positional embeddings
- Multi-head self-attention with causal masking
- Feed-forward network (FFN) with 4× expansion
- Residual connections throughout

---

## Change 1: Pre-LayerNorm (Pre-LN)

### GPT-1 used Post-LN (LayerNorm after the residual)

```
x ──► Attention ──► (+) ──► LayerNorm ──► FFN ──► (+) ──► LayerNorm ──► out
       ▲               │                   ▲               │
       └───────────────┘                   └───────────────┘
            residual                            residual
```

Post-LN: the residual stream is normalized *after* being added. The raw residual
bypasses the norm and feeds directly into the next attention layer.

### GPT-2 uses Pre-LN (LayerNorm before the sublayer)

```
x ──► LayerNorm ──► Attention ──► (+) ──► LayerNorm ──► FFN ──► (+) ──► out
                                   ▲                              ▲
                                   └──────────────────────────────┘
                                              residual (x)
```

Then, after all transformer blocks, a **final LayerNorm** is applied:

```
out ──► Final LayerNorm ──► Output projection ──► logits
```

### Block comparison side-by-side

```
GPT-1 Transformer Block            GPT-2 Transformer Block
─────────────────────────          ─────────────────────────
1. attn(x)           = a           1. ln1(x)            = n1
2. x + a             = r1          2. attn(n1)          = a
3. layerNorm(r1)     = h           3. x + a             = r1   ← residual
4. ffn(h)            = f           4. ln2(r1)           = n2
5. h + f             = r2          5. ffn(n2)           = f
6. layerNorm(r2)     = out         6. r1 + f            = out  ← residual
```

### Why Pre-LN wins at scale

**Gradient flow**: In Post-LN, gradients from the loss must pass through LayerNorm
to reach early layers. LayerNorm rescales by 1/std, which can squish or amplify
gradients unpredictably. With Pre-LN, there is always a clean residual path from
output to input — the gradient can flow backward through the additions unimpeded.

**Training stability**: Post-LN requires careful learning rate warmup and tuning
to avoid divergence. Pre-LN is stable with a constant learning rate out of the box.
This matters enormously at GPT-2 scale (1.5B params) where instability is catastrophic.

**The tradeoff**: Pre-LN slightly underperforms Post-LN at convergence on some
benchmarks, but it is far more trainable in practice. Every major LLM after GPT-2
(GPT-3, LLaMA, Mistral, etc.) uses Pre-LN or a variant of it.

---

## Change 2: Weight Tying

### The two matrices that share weights

A language model has two large matrices that both map between *token space* and
*embedding space*:

| Matrix      | Shape               | Purpose                                |
|-------------|---------------------|----------------------------------------|
| `embTable`  | vocabSize × embedDim | Converts token id → embedding vector  |
| `Wout`      | embedDim × vocabSize | Converts final hidden state → logits  |

Notice: `Wout` is the transpose of `embTable`.

Weight tying means we literally use the same matrix for both:

```
logits = X_final · embTable^T
```

Instead of allocating a separate `Wout`, we reuse `embTable` transposed.

### Why this works mathematically

Both matrices are doing the same conceptual job in opposite directions:

- **embTable**: "given token id t, what vector represents this token?"
- **Wout row t**: "how similar is the hidden state to token t?"

If `embTable[t]` is a good representation of token `t`, then the dot product
`hidden · embTable[t]` is a natural similarity score — which is exactly what we
want for the logit of token `t`. The matrices are naturally aligned.

This was first shown to work well in Press & Wolf (2016) "Using the Output Embedding
to Improve Language Models".

### Parameter savings

```
savings = vocabSize × embedDim

GPT-2 small:  50,257 × 768  =  38,597,376  ≈ 38M params saved
GPT-2 XL:     50,257 × 1600 =  80,411,200  ≈ 80M params saved
```

For GPT-2 small (117M total), weight tying saves 33% of the parameter count.
The model is smaller *and* trains better because the embedding and output projection
are jointly optimized through every forward pass.

---

## Full GPT-2 Architecture Diagram

```
Input token ids: [t0, t1, t2, ..., tT-1]
        │
        ▼
┌───────────────────────────────────────┐
│  Token Embedding Table                │
│  embTable[t]  →  embedDim vector      │   vocabSize × embedDim  (SHARED)
└───────────────────────────────────────┘
        │
        + (add)
        │
┌───────────────────────────────────────┐
│  Positional Embedding Table           │
│  posTable[pos]  →  embedDim vector    │   blockSize × embedDim
└───────────────────────────────────────┘
        │
        ▼
   X  [T × embedDim]
        │
┌───────────────────────────────────────┐
│  Transformer Block × numLayers        │
│                                       │
│  ┌─────────────────────────────────┐  │
│  │  LayerNorm (pre-LN)             │  │
│  │  Multi-Head Self-Attention      │  │
│  │  + Residual                     │  │
│  │  LayerNorm (pre-LN)             │  │
│  │  Feed-Forward Network (GELU)    │  │
│  │  + Residual                     │  │
│  └─────────────────────────────────┘  │
│          × numLayers                  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Final LayerNorm                      │  ← GPT-2 adds this
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Output Projection                    │
│  logits = X_final · embTable^T        │  ← REUSES embTable (weight tying)
└───────────────────────────────────────┘
        │
        ▼
  logits  [T × vocabSize]
```

---

## GPT-2 Size Variants

| Variant    | Layers | Heads | Embed Dim | Params  |
|------------|--------|-------|-----------|---------|
| Small      | 12     | 12    | 768       | 117M    |
| Medium     | 24     | 16    | 1024      | 345M    |
| Large      | 36     | 20    | 1280      | 762M    |
| XL         | 48     | 25    | 1600      | 1.5B    |

All variants use:
- Context window (blockSize): 1024 tokens
- Vocabulary size: 50,257 (BPE from lesson 18)
- Activation: GELU (not ReLU)
- Weight tying on token embedding and output projection
- Pre-LayerNorm in every transformer block
- Final LayerNorm before the output projection

---

## What we implement in index.ts

- `TransformerBlock`: Pre-LN architecture (ln → attn → residual → ln → FFN → residual)
- `GPT2Model`: full model class with weight tying (`logits = X · embTable^T`)
- Tiny demo: 2 layers, 4 heads, 64 dim — same training loop as lesson 17 (Adam)
- Parameter count comparison: with vs. without weight tying

---

## Key Takeaways

1. **Pre-LN** moves LayerNorm inside the residual branch, before the sublayer.
   This gives cleaner gradient paths and stable training at scale.

2. **Weight tying** reuses the token embedding matrix as the output projection.
   `logits = X · embTable^T`. Saves tens of millions of parameters and improves
   alignment between embedding and prediction.

3. **Final LayerNorm** is added after all blocks to normalize before the projection.

4. These two changes, combined with much more data and scale, are what separates
   GPT-2 from GPT-1 architecturally. The rest is just more layers and parameters.

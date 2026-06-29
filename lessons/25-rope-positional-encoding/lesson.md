# Lesson 25 — RoPE: Rotary Position Embeddings

## What We Have So Far

In Lesson 14, we built Tiny GPT and gave each token a sense of position using **absolute position embeddings**: a learned embedding table of shape `(blockSize, n_embd)`. Before feeding tokens into the transformer, we simply added their position embedding to their token embedding.

```
x = token_emb[token_id] + pos_emb[position]
```

This works, but it carries two fundamental limitations.

### Problem 1: Fixed Maximum Sequence Length

The position embedding table has a fixed number of rows — one per position up to `blockSize`. If your model was trained with `blockSize = 512`, it cannot process a sequence of 513 tokens. The table has no entry for position 512 (0-indexed). You would need to retrain from scratch to extend the context.

### Problem 2: No Relative Position Awareness

Absolute position embeddings encode each position independently. When computing attention, token `i` attends to token `j` based on their **individual** position embeddings — not on how far apart they are.

Consider two pairs of tokens:
- Token at position 2 attending to token at position 0 (distance = 2)
- Token at position 102 attending to token at position 100 (distance = 2)

These two pairs have the same relative distance, but the model sees completely different absolute position vectors. It must **learn** from data that "distance 2 means something" across all positions. This is wasteful and does not generalize well.

What we want: the model should naturally understand that two tokens are `k` positions apart, regardless of where in the sequence they are.

---

## Introducing RoPE

**RoPE** (Rotary Position Embedding) was proposed by Su et al. in 2021 and has since become the dominant positional encoding scheme in modern LLMs:

- GPT-NeoX (EleutherAI)
- Llama, Llama 2, Llama 3 (Meta)
- Mistral, Mixtral
- Falcon
- Gemma (Google)

The key idea is elegant: instead of **adding** a position vector to the token embedding, RoPE **rotates** the Query and Key vectors by an angle proportional to their position. Rotation is applied inside the attention mechanism, not at the input.

### The Core Insight

If token at position `m` has its Q vector rotated by angle `m * θ`, and token at position `n` has its K vector rotated by angle `n * θ`, then the dot product:

```
Q_m · K_n = f(m - n)
```

The dot product **only depends on the relative distance** `m - n`, not on the absolute values of `m` and `n`. This is the mathematical guarantee that gives RoPE its relative position awareness — it falls out of the rotation arithmetic automatically.

---

## The Math

### Step 1: Pair Up the Dimensions

Each attention head operates on vectors of size `head_dim`. RoPE splits these dimensions into pairs:

```
[x₀, x₁, x₂, x₃, x₄, x₅, ...]
→ pairs: (x₀, x₁), (x₂, x₃), (x₄, x₅), ...
```

There are `head_dim / 2` pairs total, indexed `i = 0, 1, 2, ..., head_dim/2 - 1`.

### Step 2: Assign a Frequency to Each Pair

Each pair `i` gets a frequency:

```
θ_i = 1 / 10000^(2i / head_dim)
```

This is the same frequency formula as the original sinusoidal positional encoding (Vaswani et al. 2017). Low-indexed pairs rotate quickly (high frequency), high-indexed pairs rotate slowly (low frequency). This creates a range of rotation speeds, analogous to how a clock has a fast second hand and a slow hour hand.

### Step 3: Apply the Rotation

For a vector at position `m`, pair index `i`, with values `(x₁, x₂)`:

```
new_x₁ = x₁ * cos(m * θ_i) - x₂ * sin(m * θ_i)
new_x₂ = x₁ * sin(m * θ_i) + x₂ * cos(m * θ_i)
```

This is exactly the 2D rotation matrix:

```
[cos(α)  -sin(α)] [x₁]
[sin(α)   cos(α)] [x₂]
```

where `α = m * θ_i`.

The full rotation of the Q (or K) vector at position `m` applies this 2D rotation independently to each pair of dimensions.

### Step 4: Why Dot Products Encode Relative Position

Let `q` be a query vector and `k` a key vector, both of dimension `head_dim`. After applying RoPE:

```
q_rotated(m) · k_rotated(n)
= Σᵢ [q₂ᵢ * cos(m*θᵢ) - q₂ᵢ₊₁ * sin(m*θᵢ)] * [k₂ᵢ * cos(n*θᵢ) - k₂ᵢ₊₁ * sin(n*θᵢ)]
  + [q₂ᵢ * sin(m*θᵢ) + q₂ᵢ₊₁ * cos(m*θᵢ)] * [k₂ᵢ * sin(n*θᵢ) + k₂ᵢ₊₁ * cos(n*θᵢ)]
```

Expanding and using the trig identity `cos(A - B) = cos(A)cos(B) + sin(A)sin(B)`:

```
= Σᵢ (q₂ᵢ * k₂ᵢ + q₂ᵢ₊₁ * k₂ᵢ₊₁) * cos((m - n) * θᵢ)
    + (q₂ᵢ * k₂ᵢ₊₁ - q₂ᵢ₊₁ * k₂ᵢ) * sin((m - n) * θᵢ)
```

Every term involves `(m - n)`, never `m` or `n` independently. The dot product is a function of `m - n` only.

---

## Implementation Details

### Pre-computation

The cos and sin tables are pre-computed for efficiency:

```
cosTable[pos][i] = cos(pos / 10000^(2i/dim))
sinTable[pos][i] = sin(pos / 10000^(2i/dim))
```

These tables have shape `(maxSeqLen, head_dim/2)`. Computing them once at initialization saves work during the forward pass.

### Where to Apply RoPE

RoPE is applied **inside the attention computation**, after projecting to Q and K but before computing `QK^T`:

```
Q = input @ W_Q          # (T, head_dim)
K = input @ W_K          # (T, head_dim)
Q_rotated = applyRoPE(Q, cosTable, sinTable)
K_rotated = applyRoPE(K, cosTable, sinTable)
attn = softmax(Q_rotated @ K_rotated.T / sqrt(head_dim))
```

Values (V) are **not** rotated — only Q and K need position information because they are the ones involved in the similarity computation.

---

## Why RoPE Is Better

### Relative Positions Are Naturally Encoded

As shown above, the dot product between any Q at position `m` and K at position `n` is a function of `m - n`. This is not learned — it is a mathematical property of rotations. The model gets relative position awareness for free.

### No Extra Parameters

Absolute learned embeddings add `blockSize * n_embd` trainable parameters. RoPE adds zero parameters — the cos/sin tables are computed deterministically from fixed frequencies.

### Extrapolation Beyond Training Length

Because RoPE uses continuous angles (not a discrete lookup table), it can in principle apply rotations for positions it was never trained on. If trained on sequences of length 512, you can compute `cos(513 * θ_i)` without any issue. In practice, naive extrapolation degrades; techniques like:

- **RoPE scaling** (divide all positions by a constant > 1 to "compress" long contexts)
- **YaRN** (Yet Another RoPE extensioN) — interpolates between frequencies
- **LongRoPE** — uses non-uniform scaling per frequency

...allow models to generalize to much longer sequences than their training length.

---

## Positional Encoding Methods: A Comparison

| Method | Model | Relative Positions | Extra Params | Extrapolates |
|--------|-------|-------------------|--------------|--------------|
| Absolute learned | GPT-1, GPT-2 | No | Yes (blockSize * d) | No |
| Sinusoidal | Original Transformer | Partial (via trig) | No | Theoretically yes, poorly in practice |
| RoPE | Llama, Mistral, GPT-NeoX | Yes (exact) | No | Yes (with scaling) |
| ALiBi | MPT, BLOOM | Yes (via bias) | No | Yes |

### ALiBi (Attention with Linear Biases)

ALiBi takes a different approach: instead of modifying Q and K, it adds a negative linear bias to attention scores based on the distance between tokens:

```
attention_score[m][n] = (Q_m · K_n) / sqrt(d) - slope * |m - n|
```

Each head gets a different `slope`. Tokens far apart are penalized more, encoding a preference for local context. ALiBi is simple and also extrapolates well, but does not encode the full richness of relative rotation that RoPE provides.

---

## Summary

RoPE replaces the fixed position embedding table with a rotation applied to Q and K vectors inside each attention layer. The rotation angle depends on position, using the same frequency schedule as sinusoidal embeddings. The key mathematical consequence is that attention dot products naturally encode relative — not absolute — distance. This is why virtually every major LLM released after 2022 uses RoPE or a variant of it.

In the code (`index.ts`), you will implement `computeRoPE` to pre-compute the tables, `applyRoPE` to rotate a Q or K matrix, and run three demonstrations: visualizing the rotation vectors, verifying the relative-distance property empirically, and comparing training loss between RoPE and absolute PE on the same task.

# Lesson 19 — Multi-Head Attention

## The problem with single-head attention

In lessons 12 and 16, every token produces one query, one key, and one value — all
living in the full embedding space of dimension D. The attention matrix is a single
T×T table that says how much each token attends to every other.

The limitation is **capacity**. One attention pattern must simultaneously encode
every relationship that matters:

- Which noun is the subject of this verb?
- Which pronoun refers to which earlier noun?
- How far apart are two tokens positionally?
- Which modifier belongs to which head noun?

These are structurally different questions. Trying to answer all of them with one
scalar per (query, key) pair forces the model to average and compress. Information
is lost.

---

## Intuition: parallel attention streams

Multi-head attention runs H independent attention operations **in parallel**, each
on a smaller subspace of dimension `head_dim = D / H`.

Think of the heads as specialised lenses:

| Head | What it might learn to track |
|------|------------------------------|
| Head 1 | Syntactic structure — subject → verb, verb → object |
| Head 2 | Coreference — "it" → "the cat", "they" → "the dogs" |
| Head 3 | Positional proximity — tokens nearby in sequence |
| Head 4 | Semantic similarity — synonyms, related concepts |

None of this is hardcoded. The heads discover their own specialisations through
training. But the key insight is that H smaller attention patterns can together
represent far richer structure than one large pattern of the same total cost.

---

## The math

### Step 1: split the embedding dimension

Given input `X` of shape `(T × D)`, split D into H equal slices of size
`head_dim = D / H`.

This is done via learned projection matrices, one set per head:

```
For each head h = 1 … H:
  Qh = X · Wqh       Wqh  is  (D × head_dim)
  Kh = X · Wkh       Wkh  is  (D × head_dim)
  Vh = X · Wvh       Wvh  is  (D × head_dim)
```

Each head sees the full sequence but projects it into a lower-dimensional space
where it can ask its own kind of question.

### Step 2: attention per head

Each head runs standard scaled dot-product attention (with causal mask for
language modelling):

```
Scores_h  = Qh · Kh^T  /  sqrt(head_dim)       (T × T)
Attn_h    = softmax(Scores_h)                    (T × T, after causal mask)
Output_h  = Attn_h · Vh                          (T × head_dim)
```

Note the scaling factor is `sqrt(head_dim)`, not `sqrt(D)`. Each head works in
its own reduced space, so the scale adapts to that space.

### Step 3: concatenate and project

All H head outputs are concatenated along the feature dimension:

```
concat = [ Output_1 | Output_2 | … | Output_H ]    (T × D)
```

This restores the original embedding dimension D (since H × head_dim = D).

A final learned projection mixes information across heads:

```
output = concat · Wo       Wo  is  (D × D)       → (T × D)
```

`Wo` is the output projection. It lets the model recombine signals from all heads
into a coherent representation before the residual add + layernorm.

---

## Parameter count: same as single-head

A natural worry: H separate projection matrices must cost more. It does not.

**Single-head attention:**

| Matrix | Shape | Parameters |
|--------|-------|------------|
| Wq     | D × D | D² |
| Wk     | D × D | D² |
| Wv     | D × D | D² |
| Wo     | D × D | D² |
| **Total** | — | **4D²** |

**Multi-head attention:**

| Per head | Shape | Parameters |
|----------|-------|------------|
| Wqh      | D × head_dim | D × (D/H) |
| Wkh      | D × head_dim | D × (D/H) |
| Wvh      | D × head_dim | D × (D/H) |

Summed across H heads: `3 × H × D × (D/H) = 3D²`.  
Plus Wo: `D × D = D²`.  
**Total: 4D²** — exactly the same.

Multi-head attention does not add parameters. It **reorganises** the same
parameter budget to run H parallel attention streams, each of lower rank.
This is strictly more expressive for the same cost.

---

## Visual: slicing the embedding dimension

```
Embedding dim D = 12, H = 3 heads, head_dim = 4

Input X  (T × 12):
┌────────────────────────────────────────┐
│  pos 0  │ d0  d1  d2  d3  d4  d5  d6  d7  d8  d9  d10  d11 │
│  pos 1  │ ...                                                  │
│  pos 2  │ ...                                                  │
└────────────────────────────────────────┘
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
        Head 1         Head 2         Head 3
      Wq1(12×4)      Wq2(12×4)      Wq3(12×4)
      Wk1(12×4)      Wk2(12×4)      Wk3(12×4)
      Wv1(12×4)      Wv2(12×4)      Wv3(12×4)
           │             │             │
    Attn1 (T×4)   Attn2 (T×4)   Attn3 (T×4)
           │             │             │
           └─────────────┴─────────────┘
                         │  concat → (T × 12)
                         │
                      · Wo (12×12)
                         │
                    output (T × 12)
```

Each head operates on the same T tokens but projects into a different 4-dimensional
subspace. Concatenation and Wo restore the full dimension.

---

## Real model scale

| Model | D (embed dim) | H (heads) | head_dim |
|-------|--------------|-----------|----------|
| GPT-2 small | 768 | 12 | 64 |
| GPT-2 medium | 1024 | 16 | 64 |
| GPT-2 large | 1280 | 20 | 64 |
| GPT-3 | 12,288 | 96 | 128 |
| GPT-4 (est.) | ~12,288 | ~96 | ~128 |

GPT-2 small: 12 heads × 64 dim each = 768. Each head has full visibility over
the sequence but reasons in a 64-dimensional subspace. 12 different questions
are asked simultaneously; their answers are concatenated and recombined by Wo.

`head_dim = 64` is a widely used default. Going below 32 tends to hurt quality
(too little per-head capacity); going above 128 does not consistently help.

---

## What changes relative to lessons 12 and 16

- **Lesson 12** introduced single-head attention: Q, K, V each shape (T × D).
- **Lesson 16** added full backprop through the softmax.
- **Lesson 19** runs H parallel attention heads, each of dimension `D/H`, then
  concatenates and projects with `Wo`.

The core per-head computation is identical to what you built in lesson 12. The
new mechanics are:
1. Splitting the linear projection into H smaller ones.
2. Concatenating outputs.
3. Applying Wo.
4. Backprop through Wo and through each head independently.

---

## Key takeaways

1. Single-head attention is a bottleneck: one T×T matrix must encode all
   relationships simultaneously.
2. Multi-head attention runs H independent attention operations in parallel,
   each in a `D/H` subspace, allowing specialisation.
3. The parameter count is identical to single-head: 4D² (just reorganised).
4. Heads can learn structurally different patterns — syntax, coreference,
   position, semantics.
5. GPT-2 small uses 12 heads × 64 dim = 768; this ratio (~64 per head) is a
   stable default across many architectures.
6. The output projection Wo mixes signals across heads — it is not optional.
   Without it, heads cannot share information, and the concat is just H
   separate single-head models running in parallel with no cross-talk.

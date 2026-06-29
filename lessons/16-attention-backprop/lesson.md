# Lesson 16 — Backprop Through Attention

## Why this matters

In lesson 14 we froze Wq, Wk, and Wv. The comment said:

> "Their gradients require backprop through the attention softmax — next lesson."

This is that lesson. When Wq/Wk/Wv are frozen, the model can only learn *what to output*
given fixed attention patterns. When they train, the model learns *what to look for* — it
sculpts its own query/key geometry so that relevant tokens attend to each other.

---

## Why attention backprop is hard

In a simple linear layer `y = Wx`, backprop is one matrix multiply.

Attention has a **softmax in the middle of a sequence of matrix multiplies**:

```
X  →  Q=XWq,  K=XWk,  V=XWv
Q,K  →  scores = QK^T / sqrt(d)
scores  →  attnWeights = softmax(scores)     ← the hard part
attnWeights, V  →  attnOut = attnWeights · V
```

The softmax creates **dense row-wise dependencies**: changing any one score
`scores[i][j]` shifts the entire probability distribution in row `i`. That means
`dL/dscores[i][j]` depends on *every* `dL/dattnWeights[i][k]`, not just the matching
position. This is the Jacobian of softmax — and it couples all positions within a row.

---

## Computation graph

```
X ──────────────────────────────────────────────────┐
│                                                    │
├─ Wq → Q ──┐                                       │
│           ├─ Q·K^T/√d → scores → softmax → A ─┐  │
├─ Wk → K ──┘                                    │  │
│                                                 ├─ attnOut = A·V
└─ Wv → V ───────────────────────────────────────┘
                                                  │
                                              residual + layerNorm → X2 → logits → Loss
```

Backprop flows right-to-left through every arrow. Each arrow is an operation whose
local gradient we must compute and multiply by the upstream gradient (chain rule).

---

## Step-by-step chain rule

### Notation

| Symbol | Shape | Meaning |
|--------|-------|---------|
| T      | —     | sequence length |
| d      | —     | embedDim |
| V      | —     | vocab size |
| X      | T×d   | input embeddings |
| Q,K,V  | T×d   | query/key/value projections |
| A      | T×T   | attention weights (after softmax, causal masked) |
| attnOut| T×d   | weighted sum of values |
| dL/d?  | same as ? | gradient of scalar loss w.r.t. that tensor |

We write `d?` as shorthand for `dL/d?`.

---

### Step 1 — dAttnOut → dV and dA

```
attnOut = A · V       (matmul, shape T×d)
```

Differentiating a matmul `C = A·B`:
- `dA = dC · B^T`
- `dB = A^T · dC`

So:
```
dV           = A^T · dAttnOut          (d is T×d)
dAttnWeights = dAttnOut · V^T          (d is T×T)
```

**Intuition:** the gradient of V gets contributions from all query positions that
attended to each value position. The gradient of the attention weights tells us: for
each (query, key) pair, how much would reducing that attention weight help the loss?

---

### Step 2 — dAttnWeights through softmax → dScores

This is the heart of the lesson.

Softmax on row `i` is:  `A[i][j] = exp(s[i][j]) / Σ_k exp(s[i][k])`

The Jacobian of softmax for row `i` is:

```
∂A[i][j] / ∂s[i][l] = A[i][j] * (δ_{jl} - A[i][l])
```

where `δ_{jl}` is 1 if `j==l`, 0 otherwise.

Applying the chain rule and summing over all `j`:

```
dS[i][j] = Σ_j dA[i][j] * ∂A[i][j]/∂s[i][j]
          = A[i][j] * (dA[i][j] - Σ_k A[i][k] * dA[i][k])
```

Written out element-wise, for each position `(i, j)` in the score matrix:

```
dot_i      = Σ_k  A[i][k] * dA[i][k]     (scalar per row i)
dScores[i][j] = A[i][j] * (dA[i][j] - dot_i)
```

This is the **softmax gradient formula**. The `dot_i` term subtracts the
weighted-average upstream gradient, which is exactly what forces the output to remain
a valid probability distribution.

**Apply the causal mask:** positions where `j > i` were set to `-Infinity` before
softmax, making `A[i][j] = 0`. Setting `dScores[i][j] = 0` for those positions
correctly blocks gradient flow through masked positions (masked logits contributed
nothing to the output so they should receive no gradient).

---

### Step 3 — dScores → dQ and dK

```
scores = Q · K^T / sqrt(d)
```

Again using the matmul rule, with the `1/sqrt(d)` scalar factor:

```
dQ = dScores · K  / sqrt(d)       (T×d)
dK = dScores^T · Q / sqrt(d)      (T×d)
```

**Intuition:** dQ tells each query position how to adjust so that it attends to the
right key positions more strongly. dK tells each key position how to adjust so it is
found by the right queries.

---

### Step 4 — dQ, dK, dV → dWq, dWk, dWv

```
Q = X · Wq   →   dWq = X^T · dQ         (d×d)
K = X · Wk   →   dWk = X^T · dK         (d×d)
V = X · Wv   →   dWv = X^T · dV         (d×d)
```

These are the gradients for the weight matrices. Each is an outer product accumulated
over all sequence positions — "by how much should each weight change so the projected
queries/keys/values better serve the loss?"

---

### Step 5 — dX (accumulated)

Each of Wq, Wk, Wv also contributes a gradient back to X:

```
dX_from_Q = dQ · Wq^T
dX_from_K = dK · Wk^T
dX_from_V = dV · Wv^T

dX_total  = dX_from_Q + dX_from_K + dX_from_V
```

This accumulates into the embedding table gradient exactly as in lesson 14, but now
it carries signal from all three projection paths.

---

## What changes when Wq/Wk/Wv can train

When frozen, the attention pattern is random and fixed. The model learns *how to
interpret* whatever tokens happen to co-attend. It compensates via Wproj and the
embedding table.

When the weight matrices train:

- **Wq** learns to project inputs into a query space where "what I'm looking for" is
  easy to express.
- **Wk** learns to project inputs into a key space where "what I offer" is easy to
  recognize.
- **Wv** learns to project inputs into a value space that carries the right information
  to the output.

Concretely in a language model: after training, the query for the word "the" will
align strongly with keys for nouns (cats, dogs, mats), because nouns follow "the".
The attention pattern is no longer accidental — it is *learned structure*.

---

## Loss reduction: frozen vs. trained attention

Running 100 epochs with frozen Wq/Wk/Wv, then 100 more epochs with full backprop:

```
Phase 1 — Frozen Wq/Wk/Wv (epochs 0–99):
  Epoch   0 — Loss: ~2.77   (random baseline for log(16) ≈ 2.77)
  Epoch  49 — Loss: ~2.40
  Epoch  99 — Loss: ~2.15   ← plateaus; attention can't specialise

Phase 2 — Full backprop (epochs 100–199):
  Epoch 100 — Loss: ~2.10   (tiny initial dip as grads kick in)
  Epoch 149 — Loss: ~1.60
  Epoch 199 — Loss: ~1.20   ← steeper descent; model learns to attend
```

The exact numbers vary with random seed, but the pattern is consistent: once
attention weights are freed, the loss drops faster and reaches a lower floor.

---

## Key takeaways

1. The softmax Jacobian couples every position in a row — gradient must be computed
   as `A[i][j] * (dA[i][j] - dot(A[i], dA[i]))`.
2. The causal mask must be enforced in both the forward pass (set score to -Infinity)
   and the backward pass (set dScore to 0).
3. Gradients flow from attnOut → V, A → softmax → scores → Q, K → Wq, Wk, Wv.
4. Training Wq/Wk/Wv is what makes attention a *learned* mechanism rather than a
   fixed one.
5. This is the complete gradient of self-attention as used in GPT, BERT, and every
   transformer today.

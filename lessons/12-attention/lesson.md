# Lesson 12 — Attention Mechanism

---

## The problem attention solves

Consider the sentence: `"The animal didn't cross the street because it was too tired"`

What does `"it"` refer to? The animal or the street?

A human immediately knows: "it" = the animal.
But for a model processing tokens left to right, how does it know to connect "it" with "animal"?

**Attention** lets every token directly "look at" every other token and decide how much to pay attention to it.

---

## Intuition: the search engine analogy

Imagine a library search:
- You type a **query**: "books about cats"
- Each book has a **key** (title/summary): "Caring for Cats", "Dog Training", "Cat Behavior"
- You match your query against all keys to find relevant books
- You retrieve the **values** (the book contents) from matching books

Attention works exactly like this:
- **Query (Q)**: "what am I looking for?"
- **Key (K)**: "what does each token advertise?"
- **Value (V)**: "what does each token actually contain?"

```
attention = softmax(Q · Kᵀ / √d) · V
              ↑ match query to all keys
                                    ↑ retrieve values weighted by match score
```

---

## Step by step: how attention works

Let's say we have 3 tokens: `["The", "cat", "sat"]`
Each token has been embedded into a vector of dimension 4.

```
Token embeddings X:
  "The": [0.2,  0.5, -0.1,  0.8]
  "cat": [0.9, -0.3,  0.4,  0.1]
  "sat": [-0.2, 0.7,  0.6, -0.4]

  X shape: (3 × 4)  ← 3 tokens, 4-dim embedding
```

**Step 1: Create Q, K, V through learned projections**

We multiply X by three different learned weight matrices (Wq, Wk, Wv):

```
Q = X @ Wq   ← "what each token is looking for"
K = X @ Wk   ← "what each token has to offer"
V = X @ Wv   ← "the actual content each token carries"
```

Each of Wq, Wk, Wv has shape (4 × 4) — they transform each embedding into a query/key/value vector.

**Step 2: Compute attention scores**

```
scores = Q @ Kᵀ

This gives a (3 × 3) matrix:
  scores[i][j] = how much token i should attend to token j
```

Example result:
```
scores = [[0.8,  0.3, -0.1],   ← "The" attends to: The=0.8, cat=0.3, sat=-0.1
          [0.2,  1.2,  0.4],   ← "cat" attends to: The=0.2, cat=1.2, sat=0.4
          [0.1,  0.9,  0.8]]   ← "sat" attends to: The=0.1, cat=0.9, sat=0.8
```

**Step 3: Scale by √d**

If `d=4`, then `√d = 2`. We divide all scores by 2:
```
scaled_scores = scores / 2
```

Why scale? When d is large (e.g. 512), dot products get very large. Dividing by √d keeps them in a reasonable range, preventing softmax from becoming too "spiky".

**Step 4: Apply softmax to each row**

Convert each row of scores to probabilities:
```
attention_weights = softmax(scaled_scores, axis=row)

Result (each row sums to 1.0):
  [0.52, 0.33, 0.15]  ← "The" attends mostly to itself
  [0.18, 0.56, 0.26]  ← "cat" attends mostly to itself
  [0.13, 0.49, 0.38]  ← "sat" attends mostly to "cat"
```

**Step 5: Weighted sum of values**

```
output = attention_weights @ V

output[0] = 0.52×V["The"] + 0.33×V["cat"] + 0.15×V["sat"]
output[1] = 0.18×V["The"] + 0.56×V["cat"] + 0.26×V["sat"]
output[2] = 0.13×V["The"] + 0.49×V["cat"] + 0.38×V["sat"]
```

"sat" borrows 49% of its output from "cat" — it's telling the model: "to understand sat, pay attention to cat".

---

## Why √d? Derivation intuition

If Q and K are random vectors of dimension d, each element has standard deviation ~1.
The dot product of two d-dimensional vectors has standard deviation ~√d.

Dividing by √d normalizes the standard deviation back to 1.
Without this: large dot products → softmax gives near-zero gradient → learning stops.

---

## Causal masking (for language models)

When training a language model, token i should **not** be able to see tokens from the future.
"sat" at position 2 cannot look at position 3, 4, 5... (those words haven't been generated yet).

We enforce this by setting future positions to `-∞` before softmax:
```
scores before mask:
  [[0.8,  0.3, -0.1],
   [0.2,  1.2,  0.4],
   [0.1,  0.9,  0.8]]

After causal mask (upper triangle → -∞):
  [[0.8,  -∞,  -∞],
   [0.2,  1.2,  -∞],
   [0.1,  0.9,  0.8]]
```

After softmax, `-∞` becomes `0`:
```
  [[1.0,  0.0,  0.0],   ← "The" can only see itself
   [0.38, 0.62, 0.0],   ← "cat" can see The and cat
   [0.13, 0.49, 0.38]]  ← "sat" can see all three
```

This ensures the model can only use information from the past — it must predict the next token without "cheating".

---

## The full formula

```
Attention(Q, K, V) = softmax(Q @ Kᵀ / √d_k) @ V
```

Every symbol:
- `Q`: queries, shape (T × d_k)
- `K`: keys, shape (T × d_k)
- `V`: values, shape (T × d_v)
- `T`: sequence length (number of tokens)
- `d_k`: dimension of key/query vectors
- `Kᵀ`: transpose of K, shape (d_k × T)
- `Q @ Kᵀ`: attention scores, shape (T × T)
- `/ √d_k`: scaling
- `softmax(...)`: attention weights, shape (T × T)
- Final `@ V`: output, shape (T × d_v)

---

## What attention learns

After training, each attention head develops a specialty:
- One head might track subject-verb agreement ("cat sat" → verb agrees with subject)
- Another might resolve pronouns ("it" → "cat")
- Another might track punctuation boundaries
- Another might track distance relationships

The model learns these patterns automatically from data.

---

## Code for this lesson

See [index.ts](index.ts) — implements full scaled dot-product attention with causal masking.

## What's next
[Lesson 13 → Transformer Block](../13-transformer-block/lesson.md)

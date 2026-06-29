# Lesson 10 — Embeddings

---

## The problem with integer IDs

After tokenization, "cat" might be token ID `42`.
But `42` is just a number. Numbers have relationships:
- `43` is very close to `42`
- `1000` is far from `42`

Are "cat" (42) and "kitten" (43) more related than "cat" (42) and "philosophy" (1000)?
Not necessarily! Token IDs are arbitrary.

We need a representation where **similar things have similar numbers**.

---

## The problem with one-hot encoding

The obvious fix: represent each token as a vector with a single `1` and the rest `0`.

For vocabulary of size 5:
```
token 0: [1, 0, 0, 0, 0]
token 1: [0, 1, 0, 0, 0]
token 2: [0, 0, 1, 0, 0]
token 3: [0, 0, 0, 1, 0]
token 4: [0, 0, 0, 0, 1]
```

Problems:
1. **Huge**: vocab_size=100,000 → each token is a vector of 100,000 numbers, 99,999 of which are zero.
2. **No similarity**: every pair of tokens is equally distant. "cat" and "kitten" are just as different as "cat" and "volcano".
3. **Not learnable**: the representation is fixed, not learned.

---

## Embeddings: the solution

An **embedding** replaces each token ID with a small, dense vector of real numbers.

```
token ID 42 → [0.12, -0.34, 0.56, 0.89, -0.11, 0.23, 0.45, -0.67]
              ↑ this is an embedding vector of dimension 8
```

These numbers are **learned** during training. The model adjusts them so that:
- "cat" and "kitten" end up with similar vectors
- "cat" and "volcano" end up with very different vectors

---

## The embedding table

An embedding table is just a **matrix** where each row is one token's embedding vector.

```
vocab_size = 5
embed_dim  = 4

embedding_table:
  row 0: [ 0.12, -0.34,  0.56,  0.89]  ← embedding for token 0
  row 1: [ 0.91,  0.03, -0.77,  0.22]  ← embedding for token 1
  row 2: [-0.45,  0.67,  0.12, -0.33]  ← embedding for token 2
  row 3: [ 0.78, -0.12,  0.44,  0.91]  ← embedding for token 3
  row 4: [-0.23,  0.55, -0.88,  0.14]  ← embedding for token 4

  shape: (5 × 4)  = (vocab_size × embed_dim)
```

**To embed token 2**: look up row 2: `[-0.45, 0.67, 0.12, -0.33]`

That's it. A lookup operation.

---

## Why is this a "lookup" (not a multiply)?

You might expect: token 2 → one-hot [0,0,1,0,0] → multiply by table → row 2

That's mathematically correct! Multiplying a one-hot vector by a matrix **selects one row**.

```
[0, 0, 1, 0, 0] @ embedding_table
= 0×row0 + 0×row1 + 1×row2 + 0×row3 + 0×row4
= row2
```

But since the result is always just "pick one row", we skip the multiplication and just index directly. It's faster.

---

## Embedding a sequence of tokens

For a sequence like "cat" (tokens [2, 0, 19]):

```
token 2  → embedding_table[2]  → [0.12, -0.34, 0.56, 0.89]
token 0  → embedding_table[0]  → [-0.45, 0.67, 0.12, -0.33]
token 19 → embedding_table[19] → [0.78, -0.12, 0.44, 0.91]

Result: matrix of shape (3 × 4)  ← 3 tokens, each with 4-dim embedding
```

This sequence embedding is the **input to the neural network**.

---

## What does "learned" mean here?

Initially, all embedding values are random.

```
Before training:
  embedding["cat"]    = [0.03, -0.12, 0.44, -0.08]   (random)
  embedding["kitten"] = [-0.91, 0.33, 0.12, 0.56]    (random)
  similarity = very low (random vectors)
```

After training on text (the model sees "cat" and "kitten" used in similar sentences):
```
After training:
  embedding["cat"]    = [0.78, 0.82, -0.12, 0.65]    (learned)
  embedding["kitten"] = [0.75, 0.79, -0.10, 0.63]    (learned)
  similarity = very high (similar vectors)
```

The embeddings encode meaning through usage patterns in the training data.

---

## Cosine similarity: measuring how similar two vectors are

```
similarity = (a · b) / (|a| × |b|)
```

Where:
- `a · b` = dot product (sum of element-wise products)
- `|a|` = length of vector a (square root of sum of squares)

Returns a value from -1 to 1:
- `1.0` = identical direction (very similar)
- `0.0` = perpendicular (unrelated)
- `-1.0` = opposite direction (antonyms, perhaps)

```
sim("cat", "kitten") → 0.92   (very similar)
sim("cat", "dog")    → 0.74   (somewhat similar — both are animals)
sim("cat", "volcano")→ 0.08   (unrelated)
```

---

## Embedding dimension: how big should the vector be?

```
embed_dim = 4       ← too small: can't capture complex meaning
embed_dim = 64      ← ok for small models (your Tiny GPT)
embed_dim = 512     ← typical for medium models
embed_dim = 8192    ← Claude uses this (approximately)
```

Larger embeddings can encode more nuance but need more memory and training data.

---

## The code

```ts
// Initialize: random small values
function initTable(vocabSize: number, embedDim: number): number[][] {
  return Array.from({ length: vocabSize }, () =>
    Array.from({ length: embedDim }, () => (Math.random() - 0.5) * 0.1)
  );
}
// Note: (Math.random() - 0.5) gives values in (-0.5, 0.5)
// × 0.1 makes them small — big initial values can cause problems

// Lookup: just get the row
function lookup(tokenId: number, table: number[][]): number[] {
  return table[tokenId];
}

// Embed a whole sequence
function embedSequence(tokens: number[], table: number[][]): number[][] {
  return tokens.map(t => lookup(t, table));
  // returns shape (sequence_length × embed_dim)
}
```

---

## Where embeddings fit in the full model

```
text: "the cat"
  ↓
tokenize: [20, 3, 1, 2, 0, 20]
  ↓
embed: table lookup → matrix of shape (6 × embed_dim)
  ↓
transformer layers: process the matrix → output matrix same shape
  ↓
project to vocabulary: each position → logits over all tokens
  ↓
softmax → probabilities → predict next token
```

The embedding table is the very first thing in every language model. Without it, you can't feed text in.

---

## Code for this lesson

See [index.ts](index.ts) — creates an embedding table, looks up tokens, computes cosine similarity.

## What's next
[Lesson 11 → Softmax & Next-Token Prediction](../11-softmax/lesson.md)

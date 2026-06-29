# Lesson 08 — Matrix Math

---

## Why do we need matrices?

In Lesson 06, our neural network had hard-coded variables:
```ts
let w1 = ..., w2 = ..., w3 = ...;
let b1 = ..., b2 = ..., b3 = ...;
```

This only works for **exactly 3 neurons**. If you want 100 neurons, you'd need 100 variables.
If you want to process 32 samples at once, you'd need to call `forward()` 32 times separately.

**Matrices solve both problems at once:**
- Weights become a 2D grid of numbers (a matrix)
- You can process hundreds of samples in one operation
- Every layer in every real neural network uses matrix multiplication

---

## What is a vector?

A **vector** is a list of numbers. You've been using these already:

```
x = [1, 2, 3]        ← a vector of length 3
y = [40, 50, 60]     ← a vector of length 3
```

We write vectors as a single row (row vector) or a single column (column vector).
Length is also called **dimension**.

---

## What is a matrix?

A **matrix** is a 2D grid of numbers — a list of lists.

```
A = [[1, 2, 3],
     [4, 5, 6]]
```

This is a **2×3 matrix** (2 rows, 3 columns).

Always say "rows × columns". Never "columns × rows".

```
shape of A = (2, 3)
             ↑  ↑
          rows  columns
```

---

## Dot product (two vectors)

The **dot product** of two vectors multiplies matching elements and sums the results.

```
a = [1, 2, 3]
b = [4, 5, 6]

a · b = (1×4) + (2×5) + (3×6)
      = 4 + 10 + 18
      = 32
```

This is exactly what a neuron computes:
```
z = w1×x1 + w2×x2 + w3×x3
  = w · x
```

One neuron = one dot product. 10 neurons = 10 dot products.

---

## Matrix multiplication

Matrix multiply (written `A @ B` or `A × B`) applies the dot product systematically.

**Rule: (m×k) @ (k×n) → (m×n)**
The inner dimensions must match. The result shape is outer dimensions.

Example: (2×3) @ (3×2) → (2×2)

```
A = [[1, 2, 3],        B = [[7, 8],
     [4, 5, 6]]             [9, 10],
                            [11, 12]]

Result[0][0] = row 0 of A · col 0 of B = 1×7 + 2×9 + 3×11 = 7+18+33 = 58
Result[0][1] = row 0 of A · col 1 of B = 1×8 + 2×10 + 3×12 = 8+20+36 = 64
Result[1][0] = row 1 of A · col 0 of B = 4×7 + 5×9 + 6×11 = 28+45+66 = 139
Result[1][1] = row 1 of A · col 1 of B = 4×8 + 5×10 + 6×12 = 32+50+72 = 154

C = [[58,  64],
     [139, 154]]
```

---

## How a layer becomes matrix multiply

**Before (neuron by neuron, 3 neurons, 2 inputs each):**
```ts
h1 = relu(w11*x1 + w12*x2 + b1);
h2 = relu(w21*x1 + w22*x2 + b2);
h3 = relu(w31*x1 + w32*x2 + b3);
```

**After (matrix multiply, entire layer at once):**
```
X = [x1, x2]                    ← input vector, shape (1 × 2)

W = [[w11, w12],                 ← weights, shape (3 × 2)
     [w21, w22],                    one row per neuron
     [w31, w32]]

b = [b1, b2, b3]                ← biases, shape (3,)

Z = X @ Wᵀ + b                  ← shape (1 × 3)
H = relu(Z)                     ← apply relu to every element
```

One matrix multiply replaces ALL 3 neurons simultaneously.

---

## Processing a batch (multiple samples at once)

Instead of one input vector, use a **matrix** of inputs:

```
X = [[x1_sample1, x2_sample1],   ← sample 1
     [x1_sample2, x2_sample2],   ← sample 2
     [x1_sample3, x2_sample3]]   ← sample 3

     shape: (3 × 2)  = 3 samples, 2 features each
```

Matrix multiply:
```
Z = X @ Wᵀ    ← shape (3 × 3) = 3 samples, 3 neurons each
H = relu(Z)   ← apply relu element-wise
```

All 3 samples processed in one operation. In practice, batch sizes are 32, 64, 256...

---

## Transpose

**Transpose** flips a matrix: rows become columns, columns become rows.

```
A = [[1, 2, 3],        Aᵀ = [[1, 4],
     [4, 5, 6]]              [2, 5],
                             [3, 6]]

shape: (2 × 3)          shape: (3 × 2)
```

We need transpose because of shape rules in matrix multiply:
```
X: (3 × 2)  ← 3 samples, 2 features
W: (3 × 2)  ← 3 neurons, 2 weights each
Wᵀ:(2 × 3)  ← transposed for multiplication to work

X @ Wᵀ → (3 × 2) @ (2 × 3) = (3 × 3) ✓
X @ W  → (3 × 2) @ (3 × 2) = ✗ inner dims 2 ≠ 3
```

---

## Implementing from scratch

```ts
// Dot product: one neuron, one sample
function dotProduct(a: number[], b: number[]): number {
  return a.reduce((sum, ai, i) => sum + ai * b[i], 0);
}

// Matrix multiply: all neurons, all samples at once
function matmul(A: number[][], B: number[][]): number[][] {
  const rows = A.length;
  const cols = B[0].length;
  const inner = B.length;
  const C = Array.from({ length: rows }, () => Array(cols).fill(0));

  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++)
      for (let k = 0; k < inner; k++)
        C[i][j] += A[i][k] * B[k][j];   // the core triple loop

  return C;
}

// Transpose: flip rows and columns
function transpose(A: number[][]): number[][] {
  return A[0].map((_, j) => A.map(row => row[j]));
}
```

The triple `for` loop in `matmul` is the heart of all deep learning.
Libraries like PyTorch run this on a GPU — millions of times faster — but the math is identical.

---

## Why this matters for LLMs

A transformer model (Lesson 12+) is essentially a sequence of matrix multiplications:
```
embedding_lookup  → (T × d)
attention Q,K,V   → (T × d) @ (d × d)  ← 3 matrix multiplies
attention scores  → (T × d) @ (d × T)
ffn layer 1       → (T × d) @ (d × 4d)
ffn layer 2       → (T × 4d) @ (4d × d)
output projection → (T × d) @ (d × vocab_size)
```

Every single one of these is a matrix multiply. Mastering this is mastering the skeleton of LLMs.

---

## Code for this lesson

See [index.ts](index.ts) — implements `dotProduct`, `matmul`, `transpose`, and a full dense layer using them.

## What's next
[Lesson 09 → Tokenization](../09-tokenization/lesson.md)

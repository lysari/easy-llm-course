# Lesson 01 — Linear Algebra for Research

---

## The problem linear algebra solves

Open any ML paper. Within the first two pages you'll hit something like:

```
h = Wx + b        "we project the input..."
A = softmax(QKᵀ/√d)   "attention scores..."
W ≈ UΣVᵀ          "we approximate the weight matrix with rank r..."
```

If those lines are noise to you, every paper is locked. If they're obvious, papers become surprisingly readable — because **90% of deep learning is one operation: multiply a matrix by a vector**. Everything else is decoration.

Linear algebra is the study of that one operation. This lesson makes it concrete.

---

## Vectors: two mental pictures, both true

A vector is just a list of numbers:

```
v = [3, 4]
```

**Picture 1 — an arrow.** Start at the origin, go 3 right and 4 up. The vector has a *direction* and a *length*. This picture powers geometric intuition: angles, similarity, projections.

```
        ▲
      4 │      ● v = [3,4]
        │     ╱
        │    ╱
        │   ╱
        │  ╱
        │ ╱
        └──────────▶
              3
```

**Picture 2 — a row of data.** `[3, 4]` might be one apartment: 3 rooms, 4th floor. In an LLM, each token becomes a vector of hundreds of numbers (an *embedding* — you built these in [../../lessons/10-embeddings/lesson.md](../../lessons/10-embeddings/lesson.md)):

```
"cat" → [0.2, -1.3, 0.7, ...]   768 numbers describing "cat-ness"
```

Research constantly switches between pictures. "The embedding of 'king' minus 'man' plus 'woman' is near 'queen'" — that's arrow arithmetic applied to data.

---

## Dot product: the similarity meter

The dot product of two vectors multiplies matching entries and adds them up:

```
a = [1, 2, 3]
b = [4, 5, 6]

a · b = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
```

Formula and every symbol:

```
a · b = Σᵢ aᵢ bᵢ
```
- `a`, `b`: vectors of the same length
- `aᵢ`: the i-th entry of a
- `Σᵢ`: "sum over all i" — just a loop that adds things up

Why care? Because geometrically:

```
a · b = ‖a‖ ‖b‖ cos(θ)
```
- `‖a‖`: the length (norm) of a — defined below
- `θ`: the angle between the two arrows
- `cos(θ)`: +1 when pointing the same way, 0 at 90°, -1 when opposite

So the dot product is a **similarity meter**:

```
a·b > 0   → vectors point roughly the same way   (similar)
a·b = 0   → perpendicular                        (unrelated)
a·b < 0   → roughly opposite                     (anti-similar)
```

Attention scores? `Q · Kᵀ` is dot products — "how similar is what I'm looking for to what you're offering." The similarity meter, run a few billion times, is what an LLM *is*.

---

## Norms: how long is a vector?

The (Euclidean, or L2) norm is the arrow's length — Pythagoras in n dimensions:

```
‖v‖ = √(v₁² + v₂² + ... + vₙ²)

v = [3, 4]
‖v‖ = √(9 + 16) = √25 = 5
```

Researchers watch norms obsessively: gradient norms (exploding? vanishing?), weight norms (regularization), activation norms (LayerNorm literally divides by one).

**Cosine similarity** removes length and keeps only direction:

```
cos_sim(a, b) = (a · b) / (‖a‖ ‖b‖)
```

Result is always in [-1, 1]. This is *the* standard way to compare embeddings: "cat" and "kitten" have cosine similarity ~0.8; "cat" and "carburetor" ~0.1. Length doesn't matter — a whispered "cat" and a shouted "CAT" should be the same concept.

---

## Orthogonality: the geometry of "unrelated"

Two vectors are **orthogonal** when their dot product is 0 — perpendicular arrows, zero similarity.

```
[1, 0] · [0, 1] = 0    ✓ orthogonal
```

Why researchers care: in high dimensions there's exponentially more "room" for near-orthogonal directions. In 768 dimensions, you can pack *tens of thousands* of vectors that are all almost-orthogonal to each other. This is why one embedding space can represent a vast vocabulary of nearly-independent concepts — and it's the geometric heart of the "superposition" hypothesis in interpretability research.

---

## Matrices: machines that transform vectors

A matrix is a grid of numbers. But the useful mental model:

> **A matrix is a machine. A vector goes in, a transformed vector comes out.**

```
M = [2  0]      input:  v = [1, 1]
    [0  3]

M·v = [2×1 + 0×1] = [2]
      [0×1 + 3×1]   [3]
```

This machine stretched x by 2 and y by 3. Other matrices rotate, reflect, shear, flatten, or mix dimensions together. A neural network layer `h = Wx + b` is exactly this: `W` transforms, `b` shifts. Training = *learning which transformation to apply*.

Matrix-vector multiply, entry by entry:

```
(M·v)ᵢ = Σⱼ Mᵢⱼ vⱼ
```
- `Mᵢⱼ`: the number in row i, column j
- `(M·v)ᵢ`: entry i of the output = dot product of **row i of M** with v

So a matrix-vector product is just: one dot product per output entry. Each output dimension is a "similarity meter" between the input and one row of the matrix.

---

## Matrix multiply: chaining machines

Multiplying matrices `A @ B` composes transformations: first apply B's machine, then A's.

Tiny concrete example:

```
A = [1  2]     B = [5  6]
    [3  4]         [7  8]

C = A @ B

C[0][0] = row 0 of A · col 0 of B = 1×5 + 2×7 = 19
C[0][1] = row 0 of A · col 1 of B = 1×6 + 2×8 = 22
C[1][0] = row 1 of A · col 0 of B = 3×5 + 4×7 = 43
C[1][1] = row 1 of A · col 1 of B = 3×6 + 4×8 = 50

C = [19  22]
    [43  50]
```

Rule and symbols:

```
Cᵢⱼ = Σₖ Aᵢₖ Bₖⱼ
```
- `A`: shape (n × m), `B`: shape (m × p) — inner dimensions must match
- `C`: shape (n × p)
- `Cᵢⱼ`: dot product of A's row i with B's column j

Shape bookkeeping — `(n×m) @ (m×p) → (n×p)` — is the researcher's first sanity check when reading any paper. If the shapes don't line up, you've misread the equation. (You met this in the from-scratch track: [../../lessons/08-matrix-math/lesson.md](../../lessons/08-matrix-math/lesson.md).)

Also: matmul order matters. `A @ B ≠ B @ A` in general — "rotate then stretch" is not "stretch then rotate."

---

## Transpose: flipping rows and columns

```
A = [1  2  3]        Aᵀ = [1  4]
    [4  5  6]             [2  5]
                          [3  6]
```

`(Aᵀ)ᵢⱼ = Aⱼᵢ`. In attention, `Q @ Kᵀ` works because transposing K turns "one key per row" into "one key per column" — exactly what matmul needs to compute every query·key dot product at once.

---

## Eigenvectors: what stays pointing the same way

Feed lots of vectors through a matrix-machine. Most come out **rotated** — pointing somewhere new. But a special few come out pointing **exactly the same direction**, merely stretched or shrunk:

```
M·v = λ·v
```
- `v`: an **eigenvector** — a direction the machine doesn't rotate
- `λ` (lambda): the **eigenvalue** — the stretch factor along that direction
  (λ = 2: doubled. λ = 0.5: halved. λ < 0: flipped but same axis.)

Concrete:

```
M = [2  1]     v = [1, 1]
    [1  2]

M·v = [2+1, 1+2] = [3, 3] = 3·[1, 1]   ✓ eigenvector, eigenvalue 3

try u = [1, 0]:
M·u = [2, 1]   — not a multiple of [1,0], so NOT an eigenvector
```

Intuition: eigenvectors are the machine's **natural axes** — the skeleton of the transformation. The eigenvalue tells you what happens along each axis.

Why researchers care:
- Apply a matrix repeatedly (deep network, recurrence): directions with |λ| > 1 **explode**, |λ| < 1 **vanish**. That's the exploding/vanishing gradient problem in one sentence.
- The dominant eigenvector (largest |λ|) is where repeated application converges — the basis of **power iteration**, which you'll implement in this lesson's code. Google's original PageRank was exactly this: the web's dominant eigenvector.
- The Hessian's eigenvalues (next lesson) describe the loss landscape's curvature — sharp vs flat minima, a live research debate about generalization.

---

## SVD: every matrix is rotate · stretch · rotate

The **Singular Value Decomposition** says any matrix — any shape, no exceptions — factors as:

```
M = U Σ Vᵀ
```
- `Vᵀ`: a rotation (pick a good set of input axes)
- `Σ` (Sigma): a diagonal matrix — pure stretching, by amounts σ₁ ≥ σ₂ ≥ ... ≥ 0, the **singular values**
- `U`: another rotation (into good output axes)

So every matrix-machine, however messy, is secretly: *rotate, stretch along the axes, rotate again*. Nothing else exists.

The singular values rank the machine's "modes" by importance. If σ₁=10, σ₂=3, σ₃=0.01, the matrix mostly does two things; the third mode is almost noise. Keeping only the top-r modes gives the **best rank-r approximation** of the matrix — provably.

Why researchers care, concretely:

- **LoRA** (how most open-model fine-tuning works): instead of updating a huge weight matrix W (say 4096×4096 ≈ 16.8M numbers), learn a correction `ΔW = B @ A` where B is 4096×8 and A is 8×4096 (≈ 65K numbers — 250× smaller). This bets that the *change* needed for a new task is low-rank — a few new modes, not a full rewrite. The bet works remarkably well, and *why* it works is genuine open research.
- **PCA** (how researchers make those 2-D plots of embedding spaces): SVD of your data matrix; keep the top 2 stretch directions; project. The picture shows the two directions where the data varies most.
- **Compression & analysis**: measuring the singular value spectrum of trained weight matrices is a standard interpretability probe — trained networks turn out far more low-rank than random ones, and nobody fully knows why. Open problem. Maybe yours.

---

## Cheat sheet: reading papers after this lesson

```
x ∈ ℝᵈ          "x is a vector with d real-number entries"
W ∈ ℝⁿˣᵐ        "W is an n-by-m matrix"
Wx              matrix-vector product: transform x
xᵀy  or  ⟨x,y⟩  dot product
‖x‖ or ‖x‖₂     L2 norm (length)
‖x‖₁            L1 norm: sum of absolute values
QKᵀ             all query-key dot products at once
rank(W)         how many independent "modes" W really has
```

---

## Code for this lesson

See [index.ts](index.ts) — implements dot product, matmul, transpose, norms, cosine similarity, and power iteration that finds the dominant eigenvector of a small matrix step by step, printing how the estimate converges.

## What's next
[Lesson 02 → Calculus and Gradients](../02-calculus-and-gradients/lesson.md)

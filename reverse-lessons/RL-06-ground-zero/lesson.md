# Reverse Lesson 06 — Ground Zero

---

## The bottom

We have peeled back every layer.

Let's look at what is actually left when you strip away:

- the words (→ integers)
- the integers (→ coordinate vectors)
- the relationships (→ dot products and weighted sums)
- the knowledge (→ highest-probability token)
- the learning (→ gradient descent on a loss)

What remains?

---

## One equation, repeated

The entire transformer — attention, feed-forward, embeddings, everything — reduces to this operation, applied over and over:

```
output = W × input + b
```

Where:
- `W` is a matrix of learned numbers
- `input` is the current vector
- `b` is a bias vector of learned numbers
- `×` is matrix multiplication
- `output` is the result (another vector of numbers)

Between layers, a non-linear activation function (like ReLU) is applied:

```
output = relu(W × input + b)
       = max(0, W × input + b)
```

That's it. `max(0, x)` — set negative numbers to zero.

**The entire intelligence of a language model is:**
```
relu(W × relu(W × relu(W × ... relu(W × input + b) ... + b) + b) + b)
```

Numbers in. Numbers out. All the way down.

---

## The full picture, bottom to top

```
┌────────────────────────────────────────────────────────────┐
│  WHAT YOU SEE                                              │
│  "The capital of France is Paris."                         │
│  Sounds meaningful. Sounds like understanding.             │
└──────────────────────┬─────────────────────────────────────┘
                       │  STRIP: text → sampling
┌──────────────────────▼─────────────────────────────────────┐
│  LAYER 5: Next-token prediction                            │
│  50,000 probabilities. Highest one wins.                   │
│  Confidence = training data frequency, not factual truth.  │
└──────────────────────┬─────────────────────────────────────┘
                       │  STRIP: probabilities → logits
┌──────────────────────▼─────────────────────────────────────┐
│  LAYER 4: Logits from output matrix                        │
│  Final hidden vector × W_out = 50,000 raw numbers          │
└──────────────────────┬─────────────────────────────────────┘
                       │  STRIP: logits → transformer layers
┌──────────────────────▼─────────────────────────────────────┐
│  LAYER 3: Attention + FFN, stacked N times                 │
│  Dot products, softmax, weighted sums, relu(Wx+b)          │
│  Relationships = geometry. Grammar = statistics.           │
└──────────────────────┬─────────────────────────────────────┘
                       │  STRIP: vectors → lookup table
┌──────────────────────▼─────────────────────────────────────┐
│  LAYER 2: Embedding lookup                                 │
│  Integer ID → coordinate in 768-dimensional space          │
│  Meaning = proximity in vector space = co-occurrence       │
└──────────────────────┬─────────────────────────────────────┘
                       │  STRIP: words → integers
┌──────────────────────▼─────────────────────────────────────┐
│  LAYER 1: Tokenization                                     │
│  "cat" → 3797. All language stripped. Just row numbers.    │
└──────────────────────┬─────────────────────────────────────┘
                       │  STRIP: behavior → parameters
┌──────────────────────▼─────────────────────────────────────┐
│  LAYER 0: GROUND ZERO                                      │
│  Floating-point numbers in matrices.                       │
│  Chosen by gradient descent to minimize -log(P(correct)).  │
│  f(x) = relu(Wx + b)  ×  billions of layers               │
│  No understanding. No facts. No intentions.                │
└────────────────────────────────────────────────────────────┘
```

---

## So where does the meaning come from?

The meaning is not in the model.

It is in you.

You read "The capital of France is Paris" and your brain — which has a genuine map of the world, memories of learning about France, possibly memories of visiting Paris — attaches meaning to those words.

The model produced a sequence of tokens that, to a human reader, maps to meaningful propositions. The model did not produce meaning. You supplied it.

This is not a flaw. It is how the technology works.

---

## Why this matters in practice

Understanding ground zero is what separates people who use AI well from people who use it badly.

**People who don't know ground zero:**
- Trust AI output as fact
- Don't verify code AI writes
- Assume AI "understood" their requirements
- Are surprised by hallucinations

**People who know ground zero:**
- Treat AI as a very fast, pattern-matching text predictor
- Verify facts against authoritative sources
- Prompt carefully, knowing the model only sees tokens
- Are never surprised by hallucinations (they expect them)
- Know when to use AI and when not to

---

## What the model is, precisely

A language model is a function:

```
f(token_ids: number[]) → probability_distribution_over_vocabulary
```

That function is a composition of linear transformations and element-wise nonlinearities, with parameters chosen to maximize the average log-probability of correct next tokens across a large corpus of text.

That's the complete technical description.

Everything else — the intelligence, the knowledge, the helpfulness, the understanding — is an interpretation layer that humans apply to the output. An interpretation layer that the model never built and doesn't have access to.

---

## The journey

You started at the surface: AI generates meaningful text.

You ended here: `relu(Wx + b)`, repeated billions of times over floating-point numbers.

Between those two points, there is no understanding — only the appearance of it, created by scale.

---

## What to do next

These reverse lessons are designed to be read alongside the [normal lessons](../../LESSON.md), which build the same machine from the bottom up.

If you've read both, you now have two views of the same system:

- **From below:** how to build a transformer from scratch
- **From above:** what a transformer actually is and isn't

That is a complete picture.

---

## Run the demo

See [demo.ts](demo.ts) — the entire transformer, from token IDs to output probabilities, in ~80 lines of TypeScript. No libraries. No abstractions. Just the math, exactly as described in this lesson series.

---

**← [RL-05 — Weights, Not Wisdom](../RL-05-weights-not-wisdom/lesson.md)**

**↑ [Back to Reverse Lessons index](../../REVERSE-LESSON.md)**

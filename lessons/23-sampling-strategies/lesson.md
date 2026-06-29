# Lesson 23 — Sampling Strategies: Temperature, Top-K, Top-P

## What You Have Built

You have a working GPT. Given a sequence of tokens it computes logits — one raw score per token in the vocabulary. To generate text you need to convert those logits into an actual next token. How you do that matters enormously.

---

## The Problem With Greedy Decoding

The simplest strategy is **greedy decoding**: always pick the token with the highest logit (argmax).

```
logits = [2.3, 1.1, 0.8, 0.2, ...]
greedy → pick index 0 (highest score)
```

This sounds reasonable. In practice it produces terrible text.

**Why?**

1. **Repetitive, boring output.** The model learns that common tokens like "the", "of", "is" are frequently correct. Once it predicts one of them, the context shifts just slightly — but the next-highest token is still "the". You get: `the the the the the`.

2. **Loop traps.** The model can fall into a cycle: `... it is important to note that it is important to note that it is important ...`. Each prediction is locally "correct" but globally catastrophic.

3. **No diversity.** The same prompt always produces the exact same output. Useful for debugging, terrible for a language model meant to be useful.

The logits contain a whole distribution of reasonable next tokens. Greedy decoding throws almost all of that away.

---

## Temperature

Temperature is the first and most fundamental fix.

### The Math

Before converting logits to probabilities via softmax, divide all logits by a temperature value T:

```
logits_scaled = logits / T
probs = softmax(logits_scaled)
```

### What This Does

Recall that softmax is:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

When you divide by T before softmax:

- **T < 1** (e.g. 0.5): dividing makes the differences between logits *larger*. The highest-scoring token gets an even larger relative advantage. The distribution becomes **peaky** — concentrated on a few tokens. More predictable, more coherent, less creative.

- **T = 1**: no change. Standard softmax.

- **T > 1** (e.g. 2.0): dividing by a large number makes the differences between logits *smaller*. All tokens get more similar scores. The distribution becomes **flat**. More random, more surprising, potentially more creative — but also more likely to produce nonsense.

- **T → 0**: approaches greedy. The highest logit dominates so completely that it always wins.

- **T → ∞**: approaches uniform random. Every token is equally likely regardless of logits.

### Concrete Example

Suppose your model produces these logits for three tokens:

```
tokens:  ["dog",   "cat",   "fish"]
logits:  [2.0,     1.0,     0.0  ]
```

**At T = 1.0 (no change):**
```
exp values: [7.389, 2.718, 1.000]
sum = 11.107
probs:       [0.665, 0.245, 0.090]
```
"dog" is most likely but "cat" and "fish" have real chances.

**At T = 0.5 (sharper / more focused):**
```
scaled logits: [4.0, 2.0, 0.0]
exp values:    [54.6, 7.389, 1.000]
sum = 62.989
probs:          [0.867, 0.117, 0.016]
```
"dog" dominates. "fish" is nearly impossible.

**At T = 2.0 (flatter / more random):**
```
scaled logits: [1.0, 0.5, 0.0]
exp values:    [2.718, 1.649, 1.000]
sum = 5.367
probs:          [0.506, 0.307, 0.186]
```
All three tokens have a meaningful shot. "fish" now has an 18.6% chance.

**Summary:**

| T   | p("dog") | p("cat") | p("fish") | Character         |
|-----|----------|----------|-----------|-------------------|
| 0.5 | 86.7%    | 11.7%    | 1.6%      | Focused, coherent |
| 1.0 | 66.5%    | 24.5%    | 9.0%      | Balanced          |
| 2.0 | 50.6%    | 30.7%    | 18.6%     | Random, creative  |

---

## Top-K Sampling

Temperature still allows low-probability tokens to appear occasionally. When T is high, even a terrible token has a small chance. Top-K sampling addresses this directly.

### The Algorithm

1. Compute logits as usual.
2. Find the K tokens with the highest logits.
3. Set all other logit values to `-Infinity` (so they get probability 0 after softmax).
4. Apply temperature scaling.
5. Apply softmax to get probabilities over only those K tokens.
6. Sample from that distribution.

### Why This Helps

If K = 40, you only ever sample from the top 40 candidates. You can never accidentally pick a token the model thinks is absurd, no matter how high you set the temperature.

### The Problem With Fixed K

K is a fixed number but distributions are not fixed in shape.

- When the model is **very confident** (e.g., predicting `Paris` after `The capital of France is`), the distribution is extremely peaked. K=40 keeps 39 tokens the model thinks are basically impossible. You get unnecessary noise.

- When the model is **uncertain** (e.g., predicting the next word of a poem), the distribution is flat. K=40 might cut off tokens the model genuinely thinks are reasonable.

A fixed K is always miscalibrated — it is either too loose or too tight depending on the context.

---

## Top-P (Nucleus) Sampling

Top-P sampling, also called **nucleus sampling**, solves the fixed-K problem adaptively.

### The Algorithm

1. Compute logits.
2. Convert to probabilities via softmax (with temperature).
3. Sort tokens by probability, highest first.
4. Walk down the sorted list, accumulating probability mass.
5. Stop when cumulative probability reaches p (e.g., 0.9).
6. The tokens visited form the "nucleus." All others are excluded.
7. Renormalize probabilities within the nucleus and sample.

### Why This Is Adaptive

- When the model is **confident**, the top 1-3 tokens might already account for 90% of the probability mass. The nucleus is tiny. You only sample from the tokens that really matter.

- When the model is **uncertain**, many tokens each have small probabilities. You need to include 30, 50, or more tokens to accumulate 90% of the mass. The nucleus is large, reflecting genuine uncertainty.

The nucleus size adjusts automatically to match the model's confidence. No manual tuning per context needed.

### Worked Example

Suppose after softmax with temperature we have these probabilities:

```
Token      Prob
"Paris"    0.60
"London"   0.20
"Berlin"   0.08
"Rome"     0.05
"Madrid"   0.03
"Vienna"   0.02
"Lisbon"   0.01
... (many more, each tiny)
```

**At p = 0.9:**

| Token    | Prob | Cumulative |
|----------|------|------------|
| "Paris"  | 0.60 | 0.60       |
| "London" | 0.20 | 0.80       |
| "Berlin" | 0.08 | 0.88       |
| "Rome"   | 0.05 | 0.93 ✓     |

We stop at "Rome" because cumulative probability crossed 0.9. The nucleus is `{Paris, London, Berlin, Rome}`. Everything else is excluded.

These four tokens are renormalized to sum to 1.0 and we sample from them.

**At p = 0.9 with a flat distribution:**

If every token had probability 0.01 (100 tokens, uniform), you would need to include all 90 of the top tokens to reach p=0.9. The nucleus is large — as it should be when the model has no real preference.

### Typical Values

- `p = 0.9` or `p = 0.95` are the most common settings.
- `p = 1.0` disables top-p filtering entirely (include all tokens).
- Values below `p = 0.5` can be too restrictive and produce repetitive output.

---

## Typical Settings in Practice

| Use Case                      | Temperature | Sampling | Notes                              |
|-------------------------------|-------------|----------|------------------------------------|
| Creative writing / chat       | 0.7–0.9     | top-p=0.95 | Balanced creativity and coherence  |
| Code generation               | 0.2–0.4     | top-k=10 | Need correctness more than variety |
| Factual Q&A                   | 0.0–0.2     | greedy   | Deterministic, precise answers     |
| Brainstorming / ideation      | 1.0–1.2     | top-p=0.95 | Maximum diversity                  |
| Poetry / creative fiction     | 0.8–1.1     | top-p=0.9 | Surprising word choices welcome    |

---

## Comparison: Output Quality Tradeoffs

| Strategy     | Repetition | Coherence | Creativity | Speed  | Notes                                   |
|--------------|------------|-----------|------------|--------|-----------------------------------------|
| Greedy       | Very High  | High      | Very Low   | Fast   | Loops, boring, deterministic            |
| Temperature  | Medium     | Medium    | Medium     | Fast   | Simple, but low-prob tokens can appear  |
| Top-K        | Low        | High      | Medium     | Fast   | Fixed K miscalibrates with distribution |
| Top-P        | Low        | High      | High       | Fast   | Best general-purpose strategy           |
| Top-P + Temp | Low        | High      | High       | Fast   | This is what GPT-3/4 and Claude use     |

---

## Key Insights

1. **Temperature controls the shape of the distribution** — how peaked vs. flat.
2. **Top-K controls the vocabulary size** — how many tokens are eligible.
3. **Top-P controls the probability mass** — how much of the distribution to cover.
4. **Top-P + Temperature together** is the standard combination used in production LLMs.
5. **Repetition** comes from always picking the same tokens. Sampling with reasonable diversity breaks loops.
6. **There is no universally optimal setting** — the right choice depends on the task.

---

## What Is Next

Lesson 24 will cover **beam search** — a structured alternative to sampling that maintains multiple candidate sequences simultaneously, used heavily in machine translation and tasks that require the single best output rather than diverse generation.

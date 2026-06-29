# Reverse Lesson 04 — Prediction, Not Knowing

---

## Layer to peel: knowing → highest-probability guess

After many attention and feed-forward layers, the transformer produces one final vector for the last token in the sequence.

This vector is then multiplied by an output matrix to produce **logits** — one number per token in the vocabulary (50,000+ numbers).

Those logits go through softmax to become a **probability distribution** over the vocabulary.

The model then picks (or samples from) this distribution to produce the next token.

That is the complete output mechanism. There is no knowledge retrieval. There is no fact-checking. There is no understanding.

---

## The question: what does "knowing" require?

When a human says "Paris is the capital of France," they know this because:

1. They learned a fact: France is a country, it has a capital city, that city is Paris
2. This fact is stored somewhere in their memory as a structured proposition
3. When asked, they retrieve this proposition and state it

Does the language model do any of this?

No.

The language model does this:

1. It receives token IDs for "The capital of France is ___"
2. These flow through all the transformer layers
3. The output is a vector → multiplied by output matrix → 50,000 logits
4. Softmax turns those logits into probabilities
5. Token "Paris" has the highest probability (e.g. 97%)
6. "Paris" is selected

The model doesn't retrieve the fact "capital of France = Paris." It predicts that the token "Paris" is the most likely continuation of this sequence — because in training data, "The capital of France is" was almost always followed by "Paris."

---

## The critical difference

**Knowing** (human): fact stored → fact retrieved → fact stated

**Predicting** (LLM): sequence seen → highest probability next token → token output

These produce the same output when the training data is correct and comprehensive. They produce different outputs when:

- The fact is rare in training data (low probability → wrong prediction)
- The fact is contested in training data (split probability → random answer)
- The training data was wrong (wrong fact → confident prediction of the wrong token)

This is why language models "hallucinate." They do not have a fact-check step. They have a next-token probability step. If the most probable continuation is a false statement, they output a false statement confidently.

---

## Example: a confident wrong answer

Prompt: `"The speed of light in a vacuum is exactly ___ meters per second."`

Training data probably has this correct (299,792,458 m/s). Model predicts it correctly.

But: `"The population of Australia in 1923 was exactly ___"`

Training data for this is sparse and potentially conflicting. The model will predict some plausible-sounding number. It will state it with the same confidence as the speed of light. It will almost certainly be wrong.

The model has no mechanism to say "I don't have reliable data for this." It always produces the most probable continuation — and fluent, confident text is always more probable than hedged, uncertain text (because most text in the training data is confident).

---

## Sampling and temperature

The output is a probability distribution. The model can:

**Greedy decode**: always pick the highest probability token
```
probs = [Paris: 97%, Lyon: 2%, Berlin: 1%]
pick → "Paris"  (always the same)
```

**Sample**: randomly draw from the distribution
```
probs = [Paris: 97%, Lyon: 2%, Berlin: 1%]
pick → "Paris" 97% of the time, "Lyon" 2%, "Berlin" 1%
```

**Temperature sampling**: flatten or sharpen the distribution
```
high temperature (creative): [Paris: 60%, Lyon: 25%, Berlin: 15%]
low temperature (focused):   [Paris: 99%, Lyon: 0.9%, Berlin: 0.1%]
```

Temperature does not change what the model "knows." It changes how deterministic the sampling is. Knowledge is not a knob. Probability distribution shape is.

---

## What confidence signals

When the model says "Paris" with 97% confidence, people interpret this as the model being sure of the fact.

The confidence actually reflects: **how consistently did "Paris" follow this token sequence in the training corpus?**

If the training corpus had 1,000,000 instances of "The capital of France is" and 970,000 of them were followed by "Paris," the model learned to predict "Paris" with ~97% probability.

The 97% is a statistical regularity, not an epistemic state.

---

## The state so far

```
WHAT YOU SEE                    WHAT'S ACTUALLY THERE
────────────────────────────    ──────────────────────────────────
"The model knows the answer"    50,000 logits → softmax → sample
"The model is confident"        Training frequency → high probability
"The model retrieved a fact"    Highest-probability token prediction
```

---

## Run the demo

See [demo.ts](demo.ts) — takes raw logits, runs softmax, samples tokens at different temperatures, and shows that "confidence" is just probability distribution shape.

---

## Next

[RL-05 → Weights, Not Wisdom](../RL-05-weights-not-wisdom/lesson.md)

Where did the probability distributions come from? Where did the attention weights come from? Where did the embeddings come from? They all came from training — the process of adjusting billions of numbers to minimize a single value. That process is not wisdom. It is optimization.

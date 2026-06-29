# Reverse Lesson 03 — Attention Is Arithmetic

---

## Layer to peel: relationships → dot products

We have a sequence of vectors, one per token.

```
"The cat sat on the mat"
  ↓
[v_the, v_cat, v_sat, v_on, v_the, v_mat]
  each v is a vector of 768 numbers
```

Now the **attention mechanism** runs. This is the heart of the transformer. The name "attention" sounds like the model is paying attention to meaning — understanding which words relate to which.

It isn't.

---

## What attention does: one sentence

Consider: `"The animal didn't cross the street because it was too tired."`

What does "it" refer to? "The animal" — not "the street."

A human reading this knows:
- Animals get tired
- Streets don't get tired
- Therefore "it" = "the animal"

A transformer handles "it" using attention. But here is what attention actually computes:

---

## The three matrices: Q, K, V

Each token vector gets multiplied by three learned weight matrices:

```
query  = token_vector × W_Q      (what am I looking for?)
key    = token_vector × W_K      (what do I offer?)
value  = token_vector × W_V      (what information do I carry?)
```

These are just matrix multiplications — standard linear algebra. Numbers in, numbers out.

Then: for every token, compute how much it should "attend to" every other token:

```
attention_score(i, j) = dot_product(query_i, key_j)
                        ÷ sqrt(dimension)
```

A dot product is just: multiply corresponding numbers, add them up.

```
query_i = [0.3, -0.2, 0.8, 0.1]
key_j   = [0.5,  0.4, 0.7, 0.2]

dot product = (0.3)(0.5) + (-0.2)(0.4) + (0.8)(0.7) + (0.1)(0.2)
            = 0.15 - 0.08 + 0.56 + 0.02
            = 0.65
```

One number. That's the "attention score" between token i and token j.

---

## Softmax to get attention weights

The scores get passed through softmax to become probabilities (attention weights):

```
scores  = [0.65, 0.20, 0.80, 0.10]   ← raw dot products
weights = softmax(scores)
        = [0.25, 0.15, 0.45, 0.10]   ← sum to 1.0
```

The token with the highest score gets the most "attention."

---

## Combining with values

Finally, the output for each token is a **weighted sum of value vectors**:

```
output_i = sum(weight_j × value_j)  for all j

         = 0.25 × value_0
         + 0.15 × value_1
         + 0.45 × value_2      ← this token gets most weight
         + 0.10 × value_3
```

This is the output of one attention head for token i: a weighted average of other tokens' value vectors.

No understanding. Just weighted averages.

---

## Back to "it"

So how does the model correctly resolve "it" = "the animal"?

During training, the weights in W_Q, W_K, W_V were adjusted via gradient descent until "it" queries that key-matched "animal" better than "street" — because in millions of sentences with "it was tired," the surrounding context words were more animal-like than street-like.

The model doesn't know animals get tired and streets don't. It learned that the vectors for "animal" and "tired" tend to co-occur in patterns where "it" also appears. The query-key dot products for those vectors happen to be higher.

**Grammar is geometry. Understanding is statistics.**

---

## What attention can and cannot do

Attention CAN:
- Route information from one token position to another
- Weight tokens differently based on learned vector geometry
- Build up complex representations through many layers of attention

Attention CANNOT:
- Look up facts from a knowledge base
- Check if a statement is true
- Reason step by step (without special training)
- Know what words mean outside of their co-occurrence patterns

---

## The state so far

```
WHAT YOU SEE                    WHAT'S ACTUALLY THERE
────────────────────────────    ─────────────────────────────────
"Understanding relationships"   Dot products between vectors
"Knowing which words connect"   Softmax over dot product scores
"Following reference (it)"      Weighted sum of value vectors
```

Three operations. All arithmetic.

---

## Run the demo

See [demo.ts](demo.ts) — implements full single-head attention from scratch, shows every step as explicit numbers, and demonstrates that the output is purely a weighted average — no language knowledge involved.

---

## Next

[RL-04 → Prediction, Not Knowing](../RL-04-prediction-not-knowing/lesson.md)

After many attention layers and feed-forward layers, the model produces logits. This is where people say "the model knows the answer." It doesn't. It predicts the most probable next token.

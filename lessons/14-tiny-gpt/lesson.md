# Lesson 14 — Tiny GPT (The Capstone)

---

## You made it. Let's build a real language model.

This lesson assembles everything from Lessons 00–13 into a working character-level language model.

It will:
1. Read a text file
2. Learn to predict the next character
3. Generate new text that sounds like the training data

This is exactly how GPT-1, GPT-2, GPT-3, and Claude work — just much smaller.

---

## The full architecture diagram

```
Input: "the cat"  (a string)
  │
  ▼
[Tokenizer]  (Lesson 09)
  "t"→20, "h"→8, "e"→5, " "→0, "c"→3, "a"→1, "t"→20
  tokens = [20, 8, 5, 0, 3, 1, 20]
  │
  ▼
[Token Embedding Table]  (Lesson 10)
  each token ID → a vector of length embed_dim (e.g. 32)
  shape: (7 tokens × 32 dims)
  │
  ▼
[Positional Encoding]  (new in this lesson)
  add a position vector to each token embedding
  shape: still (7 × 32) — each position has a unique offset
  │
  ▼
[Transformer Block 1]  (Lesson 13)
  LayerNorm → Multi-Head Attention → residual
  LayerNorm → FFN → residual
  shape: still (7 × 32)
  │
  ▼
[Transformer Block 2]  (same structure, different weights)
  shape: still (7 × 32)
  │
  ▼
[Layer Norm]
  │
  ▼
[Linear Projection to Vocabulary]
  (7 × 32) → (7 × vocab_size)
  each of the 7 positions gets a logit for every possible next character
  │
  ▼
[Softmax]  (Lesson 11)
  → probabilities for each next character
  │
  ▼
[Sample]
  → next character
```

---

## Positional encoding: teaching the model about order

Attention is **order-independent**: it treats all tokens as a set, not a sequence.
If you shuffle the input tokens, the attention weights change, but the model has no way to know the original order.

**Positional encoding** adds position information to each token embedding:

```
final_embedding[position] = token_embedding + position_embedding

position 0: [0.1,  0.2,  ...] ← token embedding
          + [0.05, 0.13, ...] ← learned position embedding (row 0 of pos_table)
          = [0.15, 0.33, ...] ← what the transformer sees
```

There are two styles:
1. **Learned**: a trainable table, just like the token embedding table. Simple.
2. **Sinusoidal** (original transformer paper): fixed pattern using sin/cos functions.

Modern models (GPT, Claude) use learned positional embeddings or variants like RoPE.

---

## Training objective: next-token prediction

Given a sequence of tokens, the model predicts the next token **at every position simultaneously**.

```
Input tokens:   [t, h, e,  , c, a, t]
Target tokens:  [h, e,  , c, a, t, ?]  ← shift by 1
```

For each position i, we compute the loss against target[i]:
```
position 0: model predicts next after "t"    → target is "h"
position 1: model predicts next after "th"   → target is "e"
position 2: model predicts next after "the"  → target is " "
...
```

This is called **teacher forcing** — during training, we always feed the true tokens as input (not the model's own predictions).

**Total loss** = average cross-entropy over all positions:
```
loss = (1/T) × Σ cross_entropy(logits[i], target[i])
```

---

## Text generation: the inference loop

After training, we generate text character by character:

```
Step 1: Start with a seed string: "the"
        Encode: [20, 8, 5]

Step 2: Forward pass → logits for position 2 (last position)
        logits = [1.2, 3.4, 0.1, ...]  (over vocab)
        probs = softmax(logits / temperature)

Step 3: Sample one token from the distribution
        (e.g. sample token 3 = " ")

Step 4: Append to sequence: [20, 8, 5, 3]
        Decode: "the "

Step 5: Repeat from Step 2
        (use last blockSize tokens as context)

Step 6: After many steps, decode the full sequence:
        "the cat sat on the mat"
```

The model generates one token at a time, using its own output as input for the next step.

---

## Context window (block size)

The model can only "see" the last `blockSize` tokens at once.

```
blockSize = 8: model uses at most 8 previous characters to predict the next

"the quick bro" → last 8 chars = "ck bro" → predict "w"
```

Real models:
- GPT-2: 1,024 tokens
- GPT-4: 128,000 tokens
- Claude: 200,000 tokens

Larger context window → model can reason about longer documents.

---

## Hyperparameter choices for tiny GPT

```ts
const config = {
  vocabSize: 65,     // ~65 unique characters in Shakespeare
  embedDim: 32,      // each token → 32-number vector
  blockSize: 8,      // look at last 8 characters
  numHeads: 2,       // 2 attention heads per block
  numLayers: 2,      // stack 2 transformer blocks
  lr: 0.005,         // learning rate
  epochs: 200,       // training passes over the data
};
```

With these settings, the model has ~50,000 parameters total.
Claude has ~500,000,000,000 (500 billion). Same structure, wildly different scale.

---

## The gap between tiny GPT and Claude

| Aspect | Tiny GPT (this lesson) | Claude |
|--------|------------------------|--------|
| Vocab | ~65 chars | ~100k tokens |
| Embedding dim | 32 | ~8,192 |
| Layers | 2 | ~100+ |
| Attention heads | 2 | ~64 |
| Parameters | ~50k | ~500 billion |
| Training data | one short text | trillions of tokens |
| Training time | seconds | months on 1000s of GPUs |
| Training method | next-token prediction | next-token prediction + RLHF |

**RLHF** (Reinforcement Learning from Human Feedback): after the base training, humans rate the model's outputs. A reward model is trained on these ratings. The LLM is then fine-tuned to maximize the reward — making it helpful and safe.

---

## What you've learned

```
Lesson 00: What is a model?           → f(x) = wx + b
Lesson 01: Linear regression          → finding the best line
Lesson 02: Loss function              → MSE measures wrongness
Lesson 03: Gradient descent           → moving w,b downhill
Lesson 04: OOP model                  → clean, reusable classes
Lesson 05: Activation functions       → sigmoid, relu, tanh
Lesson 06: Neural network (MLP)       → stacking layers
Lesson 07: Backpropagation            → chain rule for gradients
Lesson 08: Matrix math                → matmul is everything
Lesson 09: Tokenization               → text → integers
Lesson 10: Embeddings                 → integers → dense vectors
Lesson 11: Softmax                    → logits → probabilities
Lesson 12: Attention                  → tokens look at each other
Lesson 13: Transformer block          → attention + FFN + residuals
Lesson 14: Tiny GPT                   → the complete language model
```

You now understand every fundamental concept behind every modern LLM.

---

## Where to go from here

1. **Train on Shakespeare**: download `tinyshakespeare.txt`, train for real, watch it generate text
2. **Add multi-head attention**: expand from 1 head to 4 or 8
3. **Add Adam optimizer**: replace your gradient descent with Adam (much faster training)
4. **Port to Python/PyTorch**: the same architecture, but with automatic differentiation (no manual backprop)
5. **Read the papers**: "Attention Is All You Need" (2017), "GPT-2" (2019), "GPT-3" (2020)

You're not just a beginner anymore.

---

## Code for this lesson

See [index.ts](index.ts) — a complete character-level GPT: tokenizer, embedding, transformer block, generation loop.

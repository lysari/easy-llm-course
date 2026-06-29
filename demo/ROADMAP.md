# LLM From Scratch — Learning Roadmap

## Where you are now
- [x] Gradient descent (linear regression, procedural)
- [x] OOP model class with training loop
- [x] Shallow neural net (3 neurons, sigmoid, backprop)
- [x] Visualization with canvas + D3

---

## Stage 1 — Neural Network Foundations

### 1.1 Multi-feature input
- Extend `x` from a single number to a vector `number[]`
- Implement dot product manually: `w · x + b`
- Why: every real layer is just this, repeated

### 1.2 Activation functions
- Implement ReLU: `max(0, z)`
- Implement tanh: `(e^z - e^-z) / (e^z + e^-z)`
- Compare how each affects training speed vs sigmoid
- Why: sigmoid causes vanishing gradients in deep nets

### 1.3 Generalized N-layer network
- Replace hard-coded `w1, w2, w3` with arrays/matrices
- Write a generic forward pass loop over layers
- Write generic backprop using the chain rule
- Why: this is the structure every deep learning framework uses

### 1.4 Loss functions
- Mean Squared Error (you already use this)
- Cross-Entropy Loss (for classification)
- Why: loss function choice shapes what the model learns

### 1.5 Mini-batch gradient descent
- Instead of updating on every sample or all samples, update on a random batch
- Implement shuffle + batch split
- Why: how all real training works (SGD, Adam, etc.)

---

## Stage 2 — From Numbers to Words

### 2.1 Tokenization
- Split text into tokens (start with characters, then words)
- Build a vocabulary: `{ word -> index }`
- Why: models work on integers, not strings

### 2.2 One-hot encoding
- Represent token index as a vector of 0s with one 1
- Example: vocab size 100, token 42 → vector of length 100
- Why: the simplest way to turn a category into a number

### 2.3 Embeddings
- Replace one-hot with a learnable dense vector (e.g. length 64)
- Implement an embedding table: `number[][]` (vocab_size × embed_dim)
- Why: similar words end up with similar vectors — this is the magic

### 2.4 Softmax + next-token prediction
- Output a probability over every token in vocabulary
- Implement softmax: `e^z_i / Σ e^z_j`
- Train to predict the next word given previous words
- Why: this is literally how language models are trained

---

## Stage 3 — The Transformer

### 3.1 Dot-product attention (single head)
- Given Query, Key, Value matrices — compute: `softmax(QKᵀ / √d) · V`
- Implement from scratch with matrix math
- Why: the core operation of every LLM

### 3.2 Multi-head attention
- Run attention H times in parallel with different weight matrices
- Concatenate outputs and project back
- Why: lets the model attend to different things simultaneously

### 3.3 Positional encoding
- Add a position signal to each token embedding
- Try sinusoidal encoding first, then learned positions
- Why: attention has no built-in sense of order

### 3.4 Feed-forward sublayer
- Two linear layers with ReLU in between, applied per token
- Why: adds non-linearity and per-position transformation

### 3.5 Layer normalization + residual connections
- Implement `LayerNorm(x + sublayer(x))`
- Why: stabilizes training in deep networks

### 3.6 One transformer block
- Stack: attention → add & norm → FFN → add & norm
- Why: this is the repeating unit in every real LLM

---

## Stage 4 — A Tiny Language Model

### 4.1 Character-level GPT
- Build a tiny transformer that predicts the next character
- Train on a small text file (a poem, a book chapter)
- Generate text by sampling from the output distribution

### 4.2 Training loop with Adam optimizer
- Implement Adam: momentum + adaptive learning rate per parameter
- Why: Adam is the standard; plain SGD is too slow for transformers

### 4.3 Temperature sampling
- Control randomness in generation with a temperature scalar
- Why: lets you tune creative vs. deterministic output

---

## Stage 5 — Going Deeper (optional)

| Topic | Why it matters |
|---|---|
| Byte-Pair Encoding (BPE) tokenizer | How GPT/Claude actually tokenize |
| KV Cache | Why inference is fast |
| Rotary positional embeddings (RoPE) | Used in Llama, Mistral |
| Grouped query attention (GQA) | Reduces memory at large scale |
| Quantization (INT8, INT4) | Run big models on small hardware |
| LoRA fine-tuning | Adapt a model with a fraction of the compute |

---

## Recommended file structure as you progress

```
LLM/
  index.ts          ← done (linear regression scratch)
  linear.ts         ← done (LinearRegression class)
  polynomial.ts     ← done (tiny neural net / MLP)
  preview.ts        ← done (visualization)
  stage1/
    matrix.ts       ← dot product, matrix multiply
    layers.ts       ← generalized N-layer net
    activations.ts  ← relu, tanh, softmax
  stage2/
    tokenizer.ts    ← vocab + tokenization
    embedding.ts    ← embedding table
    lm.ts           ← next-token prediction
  stage3/
    attention.ts    ← scaled dot-product attention
    transformer.ts  ← full transformer block
  stage4/
    gpt.ts          ← tiny character-level GPT
    train.ts        ← training loop with Adam
    generate.ts     ← text generation + sampling
```

---

## Key mental models to build

1. **Everything is matrix multiplication** — attention, embeddings, FFN layers, all of it
2. **Backprop is just the chain rule** — applied repeatedly across layers
3. **A language model is just next-token prediction** — nothing more
4. **Scale is the secret** — the same architecture, with more data and parameters, becomes GPT-4

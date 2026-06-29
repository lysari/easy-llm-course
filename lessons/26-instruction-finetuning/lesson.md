# Lesson 26 — Instruction Fine-Tuning: Teaching the Model to Follow Instructions

## What We Have So Far

By Lesson 25 you have built a complete pre-training stack: a GPT architecture with multi-head attention, feed-forward layers, RoPE positional embeddings, and a training loop that minimizes next-token prediction loss over a large corpus. The result is a **base model** — one that has compressed a vast amount of text into its weights.

But there is a problem.

---

## The Problem with Base GPT Models

Ask a base GPT-3 model to write a poem. A well-trained base model might respond like this:

```
User: Write a poem about the ocean.

Model: Write a poem about the ocean. Write a poem about the mountains.
       Write a poem about the sky. Write a poem about...
```

It completes the document — because that is what it was trained to do. It saw millions of documents that looked like writing prompts, style guides, and exercise sheets. The most statistically likely continuation of "Write a poem about the ocean." is more instructions, not a poem.

**The core issue:** Base models learned to predict the next token, not to be helpful.

The pre-training objective is:

```
Minimize  -sum over t of  log P(x_{t+1} | x_1, ..., x_t)
```

This makes the model a very good text completer. It does not make it an assistant. An assistant must:
- Understand that "Write a poem" is a request, not the beginning of a list
- Respond in a turn-taking conversational format
- Follow the intent of the user, not just the statistical pattern of the training corpus

**Base models are not assistants.** They are world models of text. Fine-tuning bridges that gap.

---

## Supervised Fine-Tuning (SFT)

The solution is simple in principle:

1. Hire human contractors to write high-quality **(instruction, response)** pairs
2. Fine-tune the pre-trained model on this data
3. Use the same next-token prediction loss — but only on the **response** tokens

### Step 1: Collect the Data

A team of annotators writes examples like:

| Instruction | Response |
|---|---|
| What is the capital of France? | The capital of France is Paris. |
| Explain gravity in simple terms. | Gravity is a force that pulls objects toward each other... |
| Write a haiku about autumn. | Crimson leaves descend / The forest holds its last breath / Winter waits below |

This data is small but extremely high quality. InstructGPT used roughly 13,000 training examples for the SFT stage.

### Step 2: Format the Data

Each example is formatted into a prompt that the model can learn from:

```
System: You are a helpful assistant.
User: What is the capital of France?
Assistant: The capital of France is Paris.
```

Special tokens mark the boundaries between turns. The model sees the entire sequence as a flat token stream during training.

### Step 3: Compute Loss Only on Response Tokens

This is the critical implementation detail. During SFT, the model processes the full formatted prompt, but the loss is **masked**:

```
Token stream:  [System: You are a helpful...] [User: What is the capital...] [Assistant: The capital of France is Paris.]
Loss mask:     [     0    0    0    0    0   ] [  0    0    0    0    0   ] [      1         1        1       1       1    ]
```

Tokens in the instruction portion get a loss weight of 0. We do not want the model to predict the user's words — we want it to predict the assistant's words *given* the user's words.

This is still the standard cross-entropy next-token loss, just with some terms zeroed out:

```
Loss = -sum over response tokens t of  log P(x_t | x_{1..t-1})
```

---

## InstructGPT — The Paper That Made This Work at Scale

**InstructGPT** (Ouyang et al., 2022, "Training language models to follow instructions with human feedback") introduced the three-stage pipeline that became the template for every chat model since:

1. **SFT**: Supervised fine-tuning on (instruction, response) pairs
2. **Reward Model**: Train a model to score responses using human preference data
3. **RLHF**: Use reinforcement learning (PPO) to optimize the SFT model against the reward model

Lesson 26 covers Stage 1 — SFT. Stages 2 and 3 (RLHF) are covered in later lessons.

Key result from the paper: the 1.3B parameter InstructGPT model was preferred over the 175B base GPT-3 model by human evaluators **85% of the time**, despite being 100× smaller. Fine-tuning on alignment data matters more than raw scale.

---

## Why Fine-Tuning Is Much Cheaper Than Pre-Training

Pre-training and fine-tuning operate at completely different scales:

| Stage | Cost | Time | Hardware | Data |
|---|---|---|---|---|
| GPT-3 pre-training | ~$4.6 million | Months | ~10,000 A100s | 300 billion tokens |
| InstructGPT SFT | ~$100,000 | Days | Dozens of GPUs | ~13,000 examples |
| LoRA fine-tune (today) | ~$100–$1,000 | Hours | 1–8 GPUs | 10k–100k examples |

Why the dramatic difference?

- **Pre-training** requires seeing hundreds of billions of tokens to learn language structure from scratch. Every parameter must update substantially.
- **Fine-tuning** starts from a model that already knows language. You are nudging it toward a new behavior, not teaching it the fundamentals.
- **Data efficiency**: Because the model already understands grammar, facts, and reasoning, it only needs a few thousand examples to learn the *format* and *intent* of assistant responses.

The rule of thumb: you need 10,000–100,000 high-quality instruction pairs for SFT. More data helps, but diminishing returns set in quickly because the model's capacity for knowledge is already saturated by pre-training.

---

## Catastrophic Forgetting

Fine-tuning comes with a risk. If you train too aggressively on the small SFT dataset, the model's weights shift far from their pre-trained values and it begins to lose the knowledge it accumulated during pre-training. This is called **catastrophic forgetting**.

A fine-tuned model might:
- Become fluent at following instructions but lose factual knowledge
- Overfit to the style of the annotation team
- Fail at tasks not represented in the SFT data

### Solutions

**Low learning rate**: Use a learning rate 10–100× smaller than pre-training. The pre-trained weights are a good starting point; you do not want to move far from them.

**Few epochs**: Train for 1–3 passes over the SFT data, not dozens. Stop early.

**Small dataset**: Counterintuitively, using a small but very high-quality dataset is safer than a large but noisy one. Noisy data pushes the model in incoherent directions.

**LoRA**: Fine-tune only a tiny fraction of the parameters. The rest stay frozen and cannot forget.

---

## LoRA: Low-Rank Adaptation

**LoRA** (Hu et al., 2021, "LoRA: Low-Rank Adaptation of Large Language Models") is the most widely used parameter-efficient fine-tuning method.

### The Problem with Full Fine-Tuning

A linear layer in a transformer has a weight matrix **W** of shape `(d_out, d_in)`. For a 1024-dimensional model, that is 1,048,576 parameters per layer. GPT-2 has 96 layers. Full fine-tuning updates every single one of those parameters.

This is expensive in memory (you need optimizer states for all parameters) and risky (catastrophic forgetting).

### The LoRA Idea

Instead of updating **W** directly, freeze **W** and learn a low-rank **delta**:

```
W_new = W + ΔW
ΔW = A · B
```

Where:
- **W** has shape `(d_out, d_in)` — frozen, never updated
- **A** has shape `(d_out, r)` — trainable
- **B** has shape `(r, d_in)` — trainable
- **r** is the rank, typically 4, 8, or 16 (much smaller than d)

The forward pass becomes:

```
output = x · W^T  +  x · (A · B)^T
       = x · W^T  +  x · B^T · A^T
```

### Why This Works

The **intrinsic dimensionality hypothesis** suggests that the changes needed to adapt a large pre-trained model to a new task lie in a low-dimensional subspace. The full weight matrix has millions of degrees of freedom, but the actual update needed is "explained" by far fewer.

### Parameter Count Comparison

For a single projection layer with `d_in = d_out = 1024`:

| Method | Trainable Parameters |
|---|---|
| Full fine-tune | 1,024 × 1,024 = **1,048,576** |
| LoRA r=4 | 1,024×4 + 4×1,024 = **8,192** (128× fewer) |
| LoRA r=8 | 1,024×8 + 8×1,024 = **16,384** (64× fewer) |
| LoRA r=16 | 1,024×16 + 16×1,024 = **32,768** (32× fewer) |

### Initialization

- **B** is initialized to **zeros**, so ΔW = A·B = 0 at the start of fine-tuning. The model begins fine-tuning exactly at the pre-trained weights.
- **A** is initialized with small random values (Gaussian with small std).

This ensures that fine-tuning starts from the pre-trained behavior and diverges only as needed.

### After Fine-Tuning

Once training is complete, you can **merge** LoRA back into the base weights:

```
W_final = W + A · B
```

The merged model has the same architecture and parameter count as the original — zero inference overhead.

---

## Summary

| Concept | Key Idea |
|---|---|
| Base model limitation | Trained to complete text, not to be helpful |
| SFT | Fine-tune on (instruction, response) pairs with loss only on response |
| InstructGPT | First paper to scale SFT + RLHF to production; 1.3B beats 175B base |
| Cost | SFT is ~50× cheaper than pre-training; needs only 10k–100k examples |
| Catastrophic forgetting | Aggressive fine-tuning overwrites pre-training knowledge |
| LoRA | Freeze W, learn low-rank ΔW = A·B; 32–128× fewer trainable parameters |

---

## What's Next

Lesson 27 covers **RLHF and Reward Modeling** — how human preference data is used to train a reward model, and how that reward model guides PPO to push the SFT model beyond what supervised labels alone can achieve.

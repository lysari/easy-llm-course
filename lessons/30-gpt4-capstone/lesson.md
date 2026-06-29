# Lesson 30 — The Complete Picture: From GPT-1 to GPT-4

You started at y = wx + b. You made it.

Thirty lessons ago, you wrote a line of code that multiplied a weight by an input and called it a model. Today, you understand every layer, every matrix, every gradient that powers the most capable AI systems ever built. That is not a small thing.

---

## The Journey You Just Completed

Before we look forward, look back at what you actually built — in TypeScript, from scratch, with no libraries, no autograd, no shortcuts:

| Lessons | What you built | What it enabled |
|---------|----------------|-----------------|
| 00–04 | Linear regression, loss functions, gradient descent, OOP | The mathematical foundation — weights, gradients, the learning loop |
| 05–07 | Activation functions, neural networks, backpropagation | Multi-layer learning — the engine inside every LLM |
| 08–11 | Matrix math, tokenization, embeddings, softmax | The vocabulary of deep learning — tokens become vectors become probabilities |
| 12–14 | Attention, transformer block, tiny GPT | **GPT-1** — you built a real language model |
| 15–17 | FFN, attention backprop, Adam optimizer | Complete GPT-1 with a working training loop |
| 18–21 | BPE tokenization, multi-head attention, GPT-2 architecture, KV cache | **GPT-2** — subword tokens, parallel attention, fast inference |
| 22–25 | Scaling laws, sampling strategies, in-context learning, RoPE | **GPT-3** — scale, emergent abilities, no fine-tuning needed |
| 26–28 | SFT, reward model, PPO (RLHF) | **InstructGPT / ChatGPT** — alignment, helpfulness, the product era |
| 29 | Mixture of Experts | **GPT-4** — sparse activation, massive capacity at manageable compute |

---

## The Complete Evolution Timeline

### GPT-1 (2018) — 117M parameters

OpenAI's first transformer-based language model. The insight: pre-train on a large unlabeled corpus, then fine-tune on a labeled task.

- Architecture: Transformer decoder only (no encoder, no cross-attention)
- Tokenization: BPE with 40k vocabulary
- Size: 12 layers, 12 attention heads, 768 model dimension
- Training data: BookCorpus (~800M words)
- Key idea: **generative pre-training** — the model learns language structure before it learns any specific task
- Limitation: still requires labeled data and per-task fine-tuning

This is the model you built in Lessons 12–17.

---

### GPT-2 (2019) — 1.5B parameters

Same architecture, dramatically larger. The surprise: language modeling at scale produces zero-shot task performance — no fine-tuning required.

- Same decoder-only transformer, just 10× bigger
- **Pre-LN**: layer normalization moved before the attention/FFN sublayers (more stable training)
- **Weight tying**: the output projection matrix is shared with the token embedding matrix (fewer parameters, better generalization)
- Training data: WebText (~40GB, curated from Reddit outbound links)
- Key idea: **scale unlocks zero-shot generalization**
- Famous for: OpenAI initially withheld it citing misuse concerns — the first AI safety release decision

You implemented the GPT-2 architecture in Lessons 18–21.

---

### GPT-3 (2020) — 175B parameters

The paper that changed everything. Same architecture again, but at a scale that produces qualitatively different behavior.

- 96 layers, 96 attention heads, 12,288 model dimension
- Trained on 300B tokens (Common Crawl, WebText, Wikipedia, books)
- **In-context learning**: give the model a few examples in the prompt, it generalizes — no gradient updates
- **Emergent abilities**: reasoning, translation, code, math — capabilities that appear suddenly at scale
- Key idea: **the prompt is the interface** — the model is a general-purpose task solver
- Limitation: still sometimes harmful, untruthful, inconsistent

You studied the scaling laws behind this in Lesson 22, ICL in Lesson 24, and RoPE (used in later models) in Lesson 25.

---

### InstructGPT / ChatGPT (2022) — RLHF added

GPT-3 quality, but aligned. This is where AI went from a research curiosity to a product used by hundreds of millions.

Three-stage training:
1. **SFT (Supervised Fine-Tuning)**: fine-tune GPT-3 on human-written ideal responses (Lesson 26)
2. **Reward Model**: train a separate model to score responses, learning human preferences (Lesson 27)
3. **PPO**: use the reward model as a signal to fine-tune the policy model via reinforcement learning (Lesson 28)

- Result: a model that follows instructions, refuses harmful requests, and stays on topic
- Key idea: **human feedback can shape model behavior** more precisely than prompting alone
- This is what made GPT-3 into ChatGPT

---

### GPT-4 (2023) — Unknown scale (rumored MoE, ~1T total parameters)

OpenAI has not published full details, but the capabilities and community analysis tell the story:

- **Multimodal**: accepts images as well as text — vision understanding baked into the transformer
- **Likely Mixture of Experts**: multiple specialized sub-networks, only a fraction active per forward pass — huge capacity at manageable per-token compute (Lesson 29)
- **~128k context window**: enabled by RoPE positional encoding with extended scaling (Lesson 25)
- **Constitutional AI / additional safety training**: multiple rounds of RLHF and rule-based filtering
- **Dramatically better reasoning**: chain-of-thought, complex multi-step problems, bar exam performance
- Key idea: **architecture innovation + scale + alignment = the frontier**

---

### Claude (Anthropic, 2023–present)

A parallel lineage that diverges on alignment philosophy:

- **Constitutional AI (CAI)**: instead of using only human labels, Claude uses AI feedback to critique and revise its own outputs against a written "constitution" (a set of principles)
- **RLAIF (RL from AI Feedback)** instead of RLHF: the reward signal comes partly from a trained AI model, not just humans — more scalable, less expensive
- **Emphasis on harmlessness and truthfulness** as first-class training objectives, not afterthoughts
- Key idea: **the model's values can be instilled through principled self-critique**, not just human supervision

The constitutional approach means Claude's alignment is more transparent and auditable than RLHF alone.

---

## What You Can Do Now

You have built every major component of a modern large language model. That means:

**You can read any LLM research paper.**
When a paper says "we use Pre-LN with RoPE positional embeddings and a gated MLP with SwiGLU activation, trained with AdamW," you know exactly what every word means and why those choices were made.

**You can implement a GPT from scratch in any language.**
You have done it in TypeScript. Python, Rust, C — the concepts transfer. The math does not change.

**You can fine-tune models with LoRA.**
Low-Rank Adaptation (LoRA) freezes the original weights and adds small trainable rank decompositions. You understand weight matrices, gradients, and the Adam optimizer — LoRA is just a clever parameter-efficient version of what you already know.

**You understand why prompts work the way they do.**
In-context learning, few-shot prompting, chain-of-thought — these are not tricks. They are exploiting the structure of the attention mechanism and the statistics learned during pre-training. You built that mechanism.

**You can contribute to open-source LLM projects.**
nanoGPT, llama.cpp, vLLM, tinygrad — these codebases are now readable to you. The abstractions, the naming conventions, the design decisions all map back to what you implemented.

---

## Where to Go Next

The curriculum ends here. The field does not. Here is the clearest path forward:

### 1. Reimplement in Python / PyTorch
Your TypeScript implementation built everything by hand: forward passes, backprop, gradient updates. PyTorch's autograd does all of that automatically. Rewrite your transformer in PyTorch — it will take 1/10th the code, and you will understand exactly what the library is doing under the hood.

### 2. Train Llama on a GPU
Use the Hugging Face `transformers` library to load Llama-3 or Mistral, run fine-tuning with LoRA via `peft`, and serve it with `vllm`. You have the conceptual foundation. The library is just scaffolding.

### 3. Read the foundational papers
These four papers are the curriculum in their original form:
- **"Attention Is All You Need"** (Vaswani et al., 2017) — the transformer architecture
- **"Language Models are Few-Shot Learners"** (Brown et al., 2020) — GPT-3
- **"Training language models to follow instructions with human feedback"** (Ouyang et al., 2022) — InstructGPT
- **"Constitutional AI: Harmlessness from AI Feedback"** (Bai et al., 2022) — Claude's alignment approach

You will read them and understand them. That was the goal of this curriculum.

### 4. Contribute to open source
- **nanoGPT** (Karpathy): the cleanest GPT-2 training codebase — a direct extension of what you built
- **llama.cpp**: inference in C/C++ — pure matrix math, no framework, very close to your TypeScript implementation
- **vLLM**: production-grade LLM serving with PagedAttention
- **tinygrad**: a minimal autograd engine — the Python version of what you built in Lessons 05–07

### 5. Specialize
The field is wide. Pick a direction:
- **Multimodal**: vision encoders, image tokenization, cross-modal attention
- **Efficient inference**: quantization (INT8/INT4), speculative decoding, flash attention
- **Safety and alignment**: red-teaming, RLHF variants, interpretability, mechanistic analysis
- **Evaluation**: benchmarking, LLM-as-judge, detecting hallucinations

---

## The Closing Thought

Every language model — GPT-4, Claude, Llama, Gemini — is, at its core, a function that takes a sequence of tokens and outputs a probability distribution over the next token. That is it. The magic is in the billions of parameters trained on the totality of human writing to make that probability distribution remarkably good.

You understand how those parameters are organized (the transformer architecture). You understand how they are initialized and updated (gradient descent, Adam). You understand what they are learning (statistical patterns in language, encoded as attention weights and FFN activations). You understand how the model is aligned to human values (RLHF, CAI). You understand how it is made efficient at scale (MoE, KV cache, RoPE).

You started at `y = wx + b`.

You made it.

Now go build something.

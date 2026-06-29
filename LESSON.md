# LLM From Scratch — Lessons

## Progress

| # | Lesson | Status | Lesson | Code |
|---|--------|--------|--------|------|
| 00 | What Is a Model? | ✅ | [lesson](lessons/00-what-is-a-model/lesson.md) | [index.ts](lessons/00-what-is-a-model/index.ts) |
| 01 | Linear Regression | ✅ | [lesson](lessons/01-linear-regression/lesson.md) | [index.ts](lessons/01-linear-regression/index.ts) |
| 02 | Loss Function | ✅ | [lesson](lessons/02-loss-function/lesson.md) | [index.ts](lessons/02-loss-function/index.ts) |
| 03 | Gradient Descent | ✅ | [lesson](lessons/03-gradient-descent/lesson.md) | [index.ts](lessons/03-gradient-descent/index.ts) |
| 04 | OOP Model | ✅ | [lesson](lessons/04-oop-model/lesson.md) | [index.ts](lessons/04-oop-model/index.ts) |
| 05 | Activation Functions | ✅ | [lesson](lessons/05-activation-functions/lesson.md) | [index.ts](lessons/05-activation-functions/index.ts) |
| 06 | Neural Network (MLP) | ✅ | [lesson](lessons/06-neural-network/lesson.md) | [index.ts](lessons/06-neural-network/index.ts) |
| 07 | Backpropagation | ✅ | [lesson](lessons/07-backpropagation/lesson.md) | [index.ts](lessons/07-backpropagation/index.ts) |
| 08 | Matrix Math | ✅ | [lesson](lessons/08-matrix-math/lesson.md) | [index.ts](lessons/08-matrix-math/index.ts) |
| 09 | Tokenization | ✅ | [lesson](lessons/09-tokenization/lesson.md) | [index.ts](lessons/09-tokenization/index.ts) |
| 10 | Embeddings | ✅ | [lesson](lessons/10-embeddings/lesson.md) | [index.ts](lessons/10-embeddings/index.ts) |
| 11 | Softmax & Next-Token Prediction | ✅ | [lesson](lessons/11-softmax/lesson.md) | [index.ts](lessons/11-softmax/index.ts) |
| 12 | Attention Mechanism | ✅ | [lesson](lessons/12-attention/lesson.md) | [index.ts](lessons/12-attention/index.ts) |
| 13 | Transformer Block | ✅ | [lesson](lessons/13-transformer-block/lesson.md) | [index.ts](lessons/13-transformer-block/index.ts) |
| 14 | Tiny GPT (GPT-1 capstone) | ✅ | [lesson](lessons/14-tiny-gpt/lesson.md) | [index.ts](lessons/14-tiny-gpt/index.ts) |
| | **── Complete GPT-1 ──** | | | |
| 15 | Feed-Forward Network | ⬜ next | [lesson](lessons/15-ffn/lesson.md) | [index.ts](lessons/15-ffn/index.ts) |
| 16 | Attention Backprop | ⬜ | [lesson](lessons/16-attention-backprop/lesson.md) | [index.ts](lessons/16-attention-backprop/index.ts) |
| 17 | Adam Optimizer | ⬜ | [lesson](lessons/17-adam-optimizer/lesson.md) | [index.ts](lessons/17-adam-optimizer/index.ts) |
| | **── GPT-2 Era ──** | | | |
| 18 | BPE Tokenization | ⬜ | [lesson](lessons/18-bpe-tokenization/lesson.md) | [index.ts](lessons/18-bpe-tokenization/index.ts) |
| 19 | Multi-Head Attention | ⬜ | [lesson](lessons/19-multi-head-attention/lesson.md) | [index.ts](lessons/19-multi-head-attention/index.ts) |
| 20 | GPT-2 Architecture | ⬜ | [lesson](lessons/20-gpt2-architecture/lesson.md) | [index.ts](lessons/20-gpt2-architecture/index.ts) |
| 21 | KV Cache | ⬜ | [lesson](lessons/21-kv-cache/lesson.md) | [index.ts](lessons/21-kv-cache/index.ts) |
| | **── GPT-3 Era ──** | | | |
| 22 | Scaling Laws | ⬜ | [lesson](lessons/22-scaling-laws/lesson.md) | [index.ts](lessons/22-scaling-laws/index.ts) |
| 23 | Sampling Strategies | ⬜ | [lesson](lessons/23-sampling-strategies/lesson.md) | [index.ts](lessons/23-sampling-strategies/index.ts) |
| 24 | In-Context Learning | ⬜ | [lesson](lessons/24-in-context-learning/lesson.md) | [index.ts](lessons/24-in-context-learning/index.ts) |
| 25 | RoPE Positional Encoding | ⬜ | [lesson](lessons/25-rope-positional-encoding/lesson.md) | [index.ts](lessons/25-rope-positional-encoding/index.ts) |
| | **── Toward GPT-4 ──** | | | |
| 26 | Instruction Fine-Tuning + LoRA | ⬜ | [lesson](lessons/26-instruction-finetuning/lesson.md) | [index.ts](lessons/26-instruction-finetuning/index.ts) |
| 27 | Reward Modeling | ⬜ | [lesson](lessons/27-reward-model/lesson.md) | [index.ts](lessons/27-reward-model/index.ts) |
| 28 | RLHF with PPO | ⬜ | [lesson](lessons/28-rlhf-ppo/lesson.md) | [index.ts](lessons/28-rlhf-ppo/index.ts) |
| 29 | Mixture of Experts | ⬜ | [lesson](lessons/29-mixture-of-experts/lesson.md) | [index.ts](lessons/29-mixture-of-experts/index.ts) |
| 30 | GPT-4 Capstone | ⬜ | [lesson](lessons/30-gpt4-capstone/lesson.md) | [index.ts](lessons/30-gpt4-capstone/index.ts) |

---

## Folder structure

```
LLM/
├── LESSON.md          ← you are here
├── ROADMAP.md         ← big picture overview
├── index.ts           ← lessons 01–03 (gradient descent)
├── linear.ts          ← lesson 04 (OOP model)
├── polynomial.ts      ← lessons 05–07 (neural net)
├── preview.ts         ← visualization helper
└── lessons/
    ├── 00-what-is-a-model/
    ├── 01-linear-regression/
    ├── 02-loss-function/
    ├── 03-gradient-descent/
    ├── 04-oop-model/
    ├── 05-activation-functions/
    ├── 06-neural-network/
    ├── 07-backpropagation/
    ├── 08-matrix-math/
    ├── 09-tokenization/
    ├── 10-embeddings/
    ├── 11-softmax/
    ├── 12-attention/
    ├── 13-transformer-block/
    ├── 14-tiny-gpt/           ← GPT-1 complete
    ├── 15-ffn/                ← start here next
    ├── 16-attention-backprop/
    ├── 17-adam-optimizer/
    ├── 18-bpe-tokenization/
    ├── 19-multi-head-attention/
    ├── 20-gpt2-architecture/
    ├── 21-kv-cache/
    ├── 22-scaling-laws/
    ├── 23-sampling-strategies/
    ├── 24-in-context-learning/
    ├── 25-rope-positional-encoding/
    ├── 26-instruction-finetuning/
    ├── 27-reward-model/
    ├── 28-rlhf-ppo/
    ├── 29-mixture-of-experts/
    └── 30-gpt4-capstone/
```

---

## The path

```
Numbers → Neural Nets → Words → Attention → GPT-1 → GPT-2 → GPT-3 → GPT-4
  00–03      04–07       08–11     12–13      14–17    18–21    22–25    26–30
```

Each lesson has: concept · math · code example · link to next.

---

## Where you are now
**Next lesson → [15 — Feed-Forward Network](lessons/15-ffn/lesson.md)**
Lesson 14's transformer block was missing the FFN — lesson 15 adds it.

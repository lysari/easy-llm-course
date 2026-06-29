# Lesson 27 — Reward Modeling: Teaching AI What "Good" Means

## Prerequisites
- Lesson 14: Tiny GPT (transformer architecture)
- Lesson 26: Instruction fine-tuning / SFT

---

## The Problem With SFT Alone

Supervised fine-tuning (SFT) taught the model to imitate demonstrations. A human expert wrote good responses; the model learned to mimic them. That works, but it hits a fundamental ceiling.

**How do you define "helpful"?**

Try writing it down. "Be accurate." OK — but accurate in what way? Down to what level of precision? What if the user is wrong and being accurate means correcting them? "Be concise." Fine — but not so concise you omit important warnings. "Be friendly." Sure — but not so friendly it feels sycophantic.

Helpfulness is not a rule. It is a judgment. And human judgment is:

- **Complex**: a good response balances dozens of competing qualities simultaneously
- **Contextual**: the right tone for a grieving user is different from the right tone for a debugging session
- **Inconsistent**: two expert annotators looking at the same response will disagree about 20–30% of the time
- **Hard to articulate**: people can recognize a good response much more easily than they can describe what makes it good

You cannot capture all of this in a dataset of demonstrations. The dataset is finite; the space of situations is infinite. The model will encounter prompts nothing like its training data, and it has no principle for what to do — only patterns to imitate.

There is a second problem: **perverse optimization**. Once you give a model a fixed objective (imitate these demonstrations), it will optimize for that objective. If the demonstrations are imperfect, it will imitate the imperfections. If there are shortcuts, it will take them.

What you actually want is a model that has internalized *the thing the demonstrations were trying to express* — not the demonstrations themselves.

---

## Reinforcement Learning from Human Feedback (RLHF)

RLHF is the framework that addresses this. It was introduced for sequential decision-making by Christiano et al. (2017, "Deep reinforcement learning from human preferences") and applied to LLMs by Stiennon et al. (2020, "Learning to summarize with human feedback"). It is the core alignment technique behind InstructGPT, ChatGPT, Claude, and most deployed LLMs.

The pipeline has five stages:

### Stage 1: Pre-train a base LLM
Train on a huge corpus of text — the internet, books, code — to learn general language understanding. This is the foundation. The model can complete text but has no concept of instruction-following.

### Stage 2: SFT (Lesson 26)
Fine-tune on a dataset of (instruction, ideal response) pairs written by human experts. Now the model knows the *format* of being helpful. But it is still limited by what the demonstrations captured.

### Stage 3: Collect Human Preference Data
This is the key shift. Instead of asking humans "write the ideal response", you ask a much easier question: **"Which of these two responses is better?"**

Show a human labeler:
- A prompt
- Response A (from the SFT model, or a different sampling)
- Response B (from the SFT model, different sample)

The human picks A or B. That's it. Comparative judgments are:
- Faster to collect than demonstrations
- More consistent between annotators
- Robust to cases where neither response is "ideal" — you just want the better one

OpenAI's "Learning to summarize" paper collected 60,000 such comparisons. The labelers were not ML researchers; they were contractors following a rubric.

### Stage 4: Train a Reward Model (this lesson)
A reward model (RM) is a neural network that takes (prompt, response) and outputs a single scalar: the predicted human preference score.

Train it on the comparison data. After training, the RM can score any response — including responses it has never seen — by generalizing the patterns in human preferences.

### Stage 5: RL Optimization (Lesson 28)
Use the RM as a reward signal to fine-tune the LLM with reinforcement learning. The LLM generates responses; the RM scores them; the RL algorithm updates the LLM to generate higher-scoring responses.

---

## The Reward Model Architecture

The reward model uses the same transformer architecture as the LLM (or a smaller version).

The one modification: **a value head**.

In the LLM, the final transformer layer outputs a hidden state for each token position, and a projection head maps those hidden states to vocabulary logits (next-token probabilities).

In the reward model, you replace the projection head with a **value head**: a single linear layer that maps the last token's hidden state to one number.

```
Input: [token_1, token_2, ..., token_T]
  ↓ Token + Positional Embeddings
  ↓ Transformer Layers (same as LLM)
  ↓ Last token hidden state: h_T  [embedDim]
  ↓ Value head: W_v · h_T         [embedDim → 1]
  ↓ Scalar reward: r              [single float]
```

Why the last token? Because by the time the transformer processes token T, its hidden state has attended to the entire sequence — it summarizes the whole (prompt + response). That summary is what you want to score.

---

## Training the Reward Model: The Bradley-Terry Loss

Given a comparison triple (prompt p, preferred response A, dispreferred response B):

1. Feed (p + A) through the RM → scalar score r_A
2. Feed (p + B) through the RM → scalar score r_B
3. We want r_A > r_B

The **Bradley-Terry preference model** gives us a clean probabilistic formulation. The probability that a human prefers A over B is:

```
P(A preferred) = sigmoid(r_A - r_B)
```

If r_A >> r_B, this probability is close to 1. If r_A ≈ r_B, it is close to 0.5 (a coin flip). If r_A << r_B, it is close to 0.

To maximize the likelihood of the observed human preferences (A is better), we maximize log P(A preferred):

```
maximize: log(sigmoid(r_A - r_B))
```

Equivalently, we minimize the **preference loss**:

```
L = -log(sigmoid(r_A - r_B))
```

This loss is zero when r_A - r_B is large (RM is confident and correct), and grows large when r_A ≤ r_B (RM got it wrong or is uncertain).

After training on 60,000 comparisons, OpenAI's reward model achieved ~75% accuracy on held-out comparisons — meaning it agreed with human annotators 75% of the time. Human annotators agreed with *each other* about 73% of the time. The RM had essentially matched human-level consistency.

---

## What the Reward Model Actually Learns

The RM is not explicitly told what "helpful" means. It infers it from the pattern of human preferences.

Across thousands of comparisons, it learns to associate high scores with responses that are:

- **Helpful**: directly address what the user asked
- **Accurate**: do not make factual errors
- **Harmless**: do not help with dangerous requests
- **Honest**: do not pretend to know things the model doesn't know
- **Well-calibrated**: acknowledge uncertainty when appropriate

These are the three H's that Anthropic's Constitutional AI work formalizes: **Helpful, Harmless, Honest**.

The RM has distilled human judgment into a differentiable function. That function can now be used as a training signal.

---

## Reward Hacking

Once you have a differentiable reward function, the LLM will optimize against it — hard. And here is the problem: **the reward model is not the same as actual human preferences**. It is an approximation trained on a finite dataset.

If you optimize the LLM against the RM too aggressively, the LLM will find and exploit the gap between the RM and reality.

**Classic example from early RLHF experiments:**

The RM learned that confident, verbose responses score higher than hedged, brief ones. The LLM learned this too. It started generating responses like:

> "I'm a very helpful assistant! I am always happy to help! Here is my answer: [answer]. I hope that was helpful! I am always here to assist you!"

The RM scored this highly. Humans, reading it, found it annoying and unhelpful.

This is called **reward hacking** or **Goodhart's Law**: when a measure becomes a target, it ceases to be a good measure.

**The solution: KL divergence penalty**

During RL training (Lesson 28), the objective is not just to maximize reward. It is:

```
maximize: r(response) - β * KL(LLM || SFT_model)
```

The KL divergence term measures how far the RL-trained LLM has drifted from the SFT model. If the LLM starts generating text very different from what the SFT model would have generated, the KL penalty increases, counteracting the reward.

This keeps the LLM "in distribution" — close to the range of text where the RM has reliable signal. If the LLM generates something the RM was never trained on, the RM's score is meaningless. The KL penalty prevents this.

The coefficient β controls the tradeoff:
- β = 0: pure reward maximization → aggressive reward hacking
- β very large: stays near SFT model → barely improves
- β ≈ 0.02–0.2: the empirical sweet spot

---

## Summary

| Concept | What it means |
|---|---|
| Preference data | Human comparisons (A vs B), not demonstrations |
| Reward model | Transformer + value head → scalar score |
| Value head | Linear layer mapping last hidden state to scalar |
| Bradley-Terry loss | -log(sigmoid(r_A - r_B)) |
| RM accuracy | ~75% agreement with human labelers |
| Reward hacking | LLM exploits the RM's imperfections |
| KL penalty | Keeps LLM close to SFT model during RL |

The reward model is the bridge between human preferences and machine optimization. It converts the fuzzy, expensive, inconsistent signal of human judgment into a cheap, differentiable, scalable reward function.

Next lesson: using that reward function in RL to fine-tune the LLM.

---

## References

- Christiano et al. (2017). "Deep reinforcement learning from human preferences." NeurIPS.
- Stiennon et al. (2020). "Learning to summarize with human feedback." NeurIPS.
- Ouyang et al. (2022). "Training language models to follow instructions with human feedback (InstructGPT)." NeurIPS.
- Bradley & Terry (1952). "Rank Analysis of Incomplete Block Designs." Biometrika.

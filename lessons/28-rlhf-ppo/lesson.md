# Lesson 28 — RLHF with PPO: Making the Model Helpful

## What You Already Know

In lesson 27 you built a reward model: a neural network that scores a completion as "good" or "bad" based on human preference data. Now the question is: **how do you actually use that reward signal to improve the LLM?**

The answer is Reinforcement Learning from Human Feedback (RLHF), and the algorithm that made it practical is PPO.

---

## Quick RL Refresher for ML People

You don't need to know deep RL to understand RLHF. Here's all you need:

| RL Term | RLHF Meaning |
|---|---|
| **Agent** | The LLM itself |
| **Action** | Generate the next token |
| **Environment** | The conversation / prompt context |
| **Episode** | One full prompt → completion |
| **Reward** | Score from the reward model (end of episode) |
| **Policy π(action\|state)** | LLM's probability distribution over next tokens |

The **goal** is to maximize expected reward:

```
maximize E[R] = E[reward_model(prompt, completion)]
```

The policy is just the LLM's softmax output at each step:

```
π(token_t | context_t) = softmax(LLM(context_t))[token_t]
```

One important detail: reward is **sparse** — you get one number at the end of the full completion, not after each token. This makes learning harder than supervised training where you have a loss signal at every token.

---

## REINFORCE: The Baseline Algorithm

Before PPO, there was REINFORCE (Williams, 1992). It's simple enough to understand in 3 lines:

1. Generate a completion, get reward R
2. Compute the policy gradient:
   ```
   ∇L = -R · Σ_t log π(token_t | context_t)
   ```
3. Update weights: `θ ← θ + α · ∇L`

The logic: if `R > 0`, increase the probability of every token in the completion. If `R < 0`, decrease them. The magnitude of the update scales with R.

### Why REINFORCE fails in practice

The variance is enormous. Two completions that look nearly identical can get very different rewards from the reward model, and REINFORCE will shove the policy in opposite directions on consecutive steps. The training is wildly unstable.

The fix needs to be: **don't update too aggressively in any one step.**

---

## PPO: Proximal Policy Optimization

**Schulman et al., 2017.** PPO became the workhorse of RLHF because it is:
- Stable (won't destroy the model in a bad update)
- Sample efficient (reuses data across multiple gradient steps)
- Simple to implement compared to earlier trust-region methods (TRPO)

### The core idea: clip the update ratio

After collecting a batch of completions with the old policy `π_old`, you want to improve the policy `π_new`. But you don't want to change it so much that the new policy is in a completely different regime.

Define the **probability ratio**:

```
r(t) = π_new(token_t | context_t) / π_old(token_t | context_t)
```

If `r = 1`, the new policy assigns the same probability as the old one — no change.
If `r = 2`, the new policy is twice as likely to pick this token.

The **advantage** A measures how much better this completion was than baseline:

```
A = R - V(state)
```

where `V(state)` is the value function's prediction of expected reward from this state. If `A > 0`, the completion was better than expected — reinforce it. If `A < 0`, it was worse — suppress it.

### The clipped objective

```
L_PPO = min(r · A, clip(r, 1-ε, 1+ε) · A)
```

where `ε = 0.2` (standard default).

Breaking this down:

- `r · A` is the standard policy gradient objective — just scale the gradient by how much the policy changed
- `clip(r, 0.8, 1.2) · A` caps the ratio at 0.8–1.2, so the policy can't move too far
- Taking the `min` of both: if the update would be large, the clipped version kicks in and limits it

Concretely, if `A > 0` (good completion) and `r > 1.2` (policy already moved a lot toward this action), the gradient is zeroed out — "we've already reinforced this enough."

This is the key stability property: **PPO never takes a step so large it leaves the region where the old data is informative.**

---

## KL Divergence Penalty: Preventing Reward Hacking

There's a deeper problem than instability: **reward hacking**.

Your reward model is an imperfect proxy for human preferences. If you optimize against it too hard, the LLM will find completions that score extremely high on the reward model but are actually garbage — it has "hacked" the proxy metric.

Classic example: a reward model trained to prefer verbose helpful answers might get exploited by a policy that generates extremely long completions full of filler text. The reward model gives them high scores; humans would hate them.

The solution: **penalize the policy for drifting too far from the original SFT model.**

Add a KL divergence penalty to the reward:

```
total_reward = reward_model_score - β · KL(π_rl || π_sft)
```

where:

```
KL(π_rl || π_sft) = Σ_t π_rl(token_t) · log(π_rl(token_t) / π_sft(token_t))
                  ≈ mean_t(log π_rl(token_t) - log π_sft(token_t))
```

The coefficient **β** (typically 0.1–0.3) controls the trade-off:
- **β too small**: reward model score climbs fast, but KL diverges → reward hacking
- **β too large**: model stays close to SFT baseline, but doesn't improve much
- **β just right**: the model improves on the reward while staying grounded

This KL term is the reason you need to keep the original SFT model around during RLHF training — it's a reference point, not just an initialization.

---

## The Complete RLHF Training Step

Putting it all together, one PPO training step looks like this:

```
1. Sample prompt x from the dataset
2. Generate completion y ~ π_rl(· | x)          ← using current RL policy
3. Also get log probs from SFT model: log π_sft(y | x)
4. Score completion: R = reward_model(x, y)
5. Compute KL penalty: KL_pen = β · mean(log π_rl - log π_sft)
6. Total reward: R_total = R - KL_pen
7. Compute advantage A = R_total - V(x)         ← subtract value baseline
8. Update policy using PPO clipped objective on A
9. Update value function toward R_total
```

The four models in RLHF training:
1. **SFT model** — frozen reference, provides log probs for KL penalty
2. **RL policy** — the model being trained (initialized from SFT)
3. **Reward model** — frozen, scores completions
4. **Value model** — trained alongside RL policy, estimates expected reward

---

## InstructGPT: The Landmark Result

InstructGPT (Ouyang et al., 2022) was the paper that validated RLHF at scale. The headline result:

> **A 1.3B parameter RLHF-trained model was preferred over the 175B base GPT-3 model by human raters.**

That's a 100x smaller model beating a much larger one — purely through alignment. The RLHF model:
- Followed instructions better
- Produced fewer toxic outputs
- Hallucinated less

The model got smaller but smarter about what humans actually wanted. This established RLHF as the standard post-training procedure and directly led to ChatGPT.

---

## DPO: A Simpler Alternative

PPO is powerful but operationally complex — you need four models running simultaneously, a careful RL training loop, and PPO's own hyperparameters on top of the LLM's.

**Direct Preference Optimization (DPO)** (Rafailov et al., 2023) made a key mathematical insight: you can derive the optimal RLHF policy in closed form and turn the RL problem into a supervised classification problem directly on the LLM.

The DPO loss:

```
L_DPO = -log(sigmoid(β · (log π(y_w|x) - log π(y_l|x) - log(π_ref(y_w|x) - π_ref(y_l|x)))))
```

where:
- `y_w` = the preferred (winning) completion from the human preference data
- `y_l` = the rejected (losing) completion
- `π_ref` = the frozen SFT reference model

**What this does**: it directly pushes up the probability of preferred completions and down the probability of rejected ones, while the reference model terms keep the policy from collapsing.

Compared to PPO:
- No separate reward model needed (trained end-to-end from preferences)
- No RL training loop — just supervised training on the LLM
- Fewer hyperparameters
- Often comparable performance at smaller scale

DPO is now the dominant approach for open-source alignment (used in LLaMA chat variants, Zephyr, etc.). PPO is still used at the frontier labs where reward models are more carefully curated.

---

## Key Takeaways

- RLHF frames language model fine-tuning as RL: the LLM is an agent taking actions (tokens) to maximize reward (human preference scores)
- REINFORCE is conceptually simple but unstable; PPO's clipping constraint stabilizes training
- The KL penalty against the SFT baseline is not optional — without it, reward hacking degrades the model fast
- β controls the alignment/capability trade-off; tuning it matters as much as the learning rate
- InstructGPT proved that alignment quality can compensate for orders-of-magnitude size differences
- DPO removes the RL loop entirely and often works just as well — it's worth knowing both

---

## Next Lesson

Lesson 29: Constitutional AI and RLAIF — how to scale alignment without human labelers.

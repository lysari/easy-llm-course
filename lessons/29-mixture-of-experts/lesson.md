# Lesson 29 — Mixture of Experts: Scaling Smartly

## The Scaling Problem

Every lesson since Lesson 22 (Scaling Laws) has pointed toward the same conclusion: bigger models are smarter. The Chinchilla laws tell us to scale parameters and tokens together. But there is a hidden cost baked into every approach we have studied so far.

**The compute problem:** In a standard transformer, every token passes through every parameter on every forward pass. Double the parameters, and you double the FLOPs per forward pass. That means doubling training cost, doubling inference cost, and doubling memory bandwidth requirements. At the scale of GPT-4 (~1 trillion parameters rumored), this becomes economically brutal.

What if we could have a large number of parameters without making every forward pass expensive?

---

## Mixture of Experts: The Core Idea

Mixture of Experts (MoE) breaks the link between *total parameters* and *active parameters per forward pass*.

The key insight: **use different parameters for different inputs**.

In a standard transformer block, the FFN layer looks like this:

```
Input X  →  FFN(X)  →  Output
```

In an MoE transformer block, you replace the single FFN with N independent FFN networks called **experts**, plus a small **router** network:

```
Input X  →  Router  →  picks K experts
                     →  Expert_1(X)  ×  score_1
                     →  Expert_3(X)  ×  score_3   (K=2, so only 2 of N run)
                     →  weighted sum  →  Output
```

The critical property: **only K out of N experts are activated for any given token**.

- Total parameters = N × (FFN parameters)
- Active parameters per token = K × (FFN parameters)

Typically K=2 and N=8, N=16, or even N=64. You get a large model cheaply.

---

## The Math

### Step 1: Router logits

Given a token representation `X` of shape `(1, embedDim)`, the router computes a score for each expert:

```
logits = X · W_router          shape: (1, N)
scores = softmax(logits)        shape: (1, N)  — N probabilities that sum to 1
```

`W_router` is a small learned matrix of shape `(embedDim, N)`.

For a sequence of T tokens, this extends to:

```
scores = softmax(X · W_router)  shape: (T, N)
```

### Step 2: Top-K selection

For each token, pick the K experts with the highest scores:

```
top_indices[t]  = argsort(scores[t], descending=true)[:K]    K expert indices
top_weights[t]  = scores[t][top_indices[t]]                   K raw scores
```

The K weights are typically renormalized so they sum to 1:

```
top_weights[t] = top_weights[t] / sum(top_weights[t])
```

### Step 3: Weighted output

Only the K selected experts run their forward pass. Their outputs are blended by the routing weights:

```
output[t] = Σ_{k=1}^{K}  top_weights[t][k]  ×  expert_{top_indices[t][k]}(X[t])
```

This is the complete MoE layer. It is differentiable — gradients flow back through the routing weights (via softmax) and through each expert's parameters.

---

## Why This Is Efficient

### Real-world numbers

**GPT-4 (rumored architecture):**
- ~8 experts per layer, 2 active per token
- ~1 trillion total parameters
- ~220 billion active parameters per forward pass
- Cost is comparable to a 220B dense model, but with 1T model capacity

**Mixtral 8×7B (open source, Mistral AI):**
- 8 experts, each with 7B parameters = 47B total parameters
- Only 2 experts active per token = ~13B active parameters per forward pass
- Runs at the cost of a 13B model, competes with 70B dense models

### Expert parallelism

In distributed training and inference, different experts can live on different GPUs. When a token arrives, only the GPUs hosting its top-K experts need to do work. This is called **expert parallelism** and it is a new axis of parallelism alongside tensor parallelism and pipeline parallelism.

The communication overhead is: each GPU sends token representations to the expert GPUs, receives the expert outputs, and combines them. With large batch sizes, this all-to-all communication is amortized efficiently.

---

## The Load Balancing Problem

MoE has a critical failure mode: **expert collapse**.

Without any regularization, the router quickly learns to always route to the same 2 experts. Here is why: if expert 3 and expert 7 happen to perform slightly better early in training, the router increases their scores. This means those experts get more gradient signal and improve faster. The rich get richer. Within a few thousand steps, the router routes nearly 100% of tokens to 2 experts, and the other 6 experts receive almost no gradient and never train.

You end up with a model that has 8 experts on paper but only 2 doing any work — exactly as expensive as a dense model, but with 7/8 of the parameters wasted.

### The fix: auxiliary load balancing loss

We add a differentiable loss term that penalizes uneven routing. Let:

- `f_i` = fraction of tokens (in a batch) routed to expert i
- `P_i` = mean routing probability for expert i across all tokens in the batch

The load balancing loss is:

```
Loss_balance = N × Σ_{i=1}^{N} (f_i × P_i)
```

where N is the number of experts. This quantity is minimized when all experts receive equal load. The factor N normalizes it so that perfectly uniform routing gives a value of 1 (rather than 1/N).

Why multiply `f_i` by `P_i` and not just penalize `f_i` directly? Because `f_i` is computed via argmax (not differentiable), while `P_i` is the mean softmax probability (differentiable). The product gives a differentiable signal: if expert i is over-used, `f_i` is large and gradients flow through `P_i` to reduce that expert's routing logits.

The total training loss becomes:

```
Loss_total = Loss_CE  +  λ × Loss_balance
```

where λ is a small coefficient, typically 0.01. The load balancing loss should not dominate; it is a regularizer.

---

## Expert Specialization

An interesting emergent behavior: experts tend to specialize.

When you track which tokens each expert handles, you find patterns. In language models:
- Some experts specialize in punctuation and formatting tokens
- Some specialize in numbers and mathematical expressions
- Some specialize in specific semantic domains (code, dialogue, narrative)
- Some specialize in specific syntactic roles (verbs, noun phrases)

This is not hard-coded. It emerges from training because it is efficient — a token that is always a number benefits from always routing to the same expert that has learned to process numbers well.

You can visualize this by logging, for each token in your vocabulary, which expert handles it most often. The result is a kind of soft topic model: experts become specialists.

In the TypeScript implementation below, we will track routing statistics and print them after training to see if any specialization emerges in our tiny example.

---

## MoE vs Dense: The Trade-off Summary

| Property | Dense FFN | MoE FFN |
|---|---|---|
| Total parameters | P | N × P |
| Active params per token | P | K × P |
| Training compute | O(P) per token | O(K × P) per token |
| Memory footprint | P | N × P |
| Routing overhead | None | Small (W_router + top-K) |
| Risk of collapse | None | Expert collapse without LB loss |
| Specialization | None | Emergent expert specialization |

The MoE trade-off is: spend more memory to store all N experts, but spend much less compute per forward pass. This is the right trade-off when you are memory-rich (modern GPUs have large HBM) and compute-constrained (training and inference FLOPs are the bottleneck).

---

## What You Will Build

In `index.ts` you will implement:

1. **MoEFFN** — a complete Mixture of Experts FFN layer with router, top-K selection, and weighted combination
2. **loadBalancingLoss** — the auxiliary loss function
3. **MoE GPT** — a tiny GPT with MoE replacing the dense FFN in each transformer block
4. **Routing statistics** — tracking and visualizing which tokens go to which experts
5. **Dense vs MoE comparison** — verifying that MoE matches dense quality with the same active parameter count

This is the architecture behind the most capable publicly known models. Understanding MoE is understanding how the frontier is actually built.

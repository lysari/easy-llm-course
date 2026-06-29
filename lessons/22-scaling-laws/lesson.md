# Lesson 22 — Scaling Laws: Why Bigger is Better (to a Point)

## The Central Question

You have built a small GPT. It works, but its output is mediocre. The obvious
question: if you train a bigger model on more data, how much better will it get?

Before 2020, the honest answer was "we don't know — try it and see." Training
runs cost millions of dollars. Guessing wrong was expensive.

Then the OpenAI team published a paper that changed how the field thinks about
model development entirely.

---

## Kaplan et al. (2020) — Neural Scaling Laws

The paper is titled "Scaling Laws for Neural Language Models" (Kaplan et al.,
OpenAI, 2020). The core finding:

**Language model loss follows predictable power laws with model size, dataset
size, and compute — and the exponents are consistent across many orders of
magnitude.**

This is remarkable. It means you can train a small cheap model, measure its
loss, and predict with reasonable accuracy what a model 1000x larger will
achieve — before you spend the money to train it.

### The Three Power Laws

**Model size (N = number of parameters):**

```
L(N) ≈ (Nc / N)^αN
```

where `αN ≈ 0.076` and `Nc` is a constant fit from data.

Doubling parameters reduces loss by a factor of `2^0.076 ≈ 1.054`. Not
dramatic per doubling, but consistent across 7 orders of magnitude.

**Dataset size (D = number of training tokens):**

```
L(D) ≈ (Dc / D)^αD
```

where `αD ≈ 0.095`. More data helps, but with diminishing returns.

**Compute (C = total FLOPs used for training):**

```
L(C) ≈ (Cc / C)^αC
```

where `αC ≈ 0.050`. This is the most actionable one: if your compute budget
doubles, you can predict exactly how much loss will improve.

### What Power Laws Look Like

On a regular scale, these curves look like they flatten out and improve slowly.
On a log-log scale, they are straight lines — that is the signature of a power
law.

```
Regular scale:          Log-log scale:
loss                    log(loss)
  |*                      |*
  | *                     | *
  |  **                   |  *
  |    ***                |   *
  |       ****            |    *
  |____________           |_________
      params              log(params)
```

The log-log linearity is what makes prediction possible: fit a line to small
experiments, extrapolate to large ones.

### The Key Insight on Optimal Allocation

Kaplan et al. also found that when you have a fixed compute budget C, the
optimal strategy is to scale model size much faster than dataset size:

- Model size: scale as `N ∝ C^0.73`
- Dataset size: scale as `D ∝ C^0.27`

This implied: **use a large model, train it for relatively fewer tokens.**

GPT-3 (175B parameters, 300B tokens) was designed under this principle. It was
the state of the art in 2020.

Then in 2022, a different team looked more carefully and found a serious flaw.

---

## Hoffmann et al. (2022) — Chinchilla and the Compute-Optimal Frontier

The paper is "Training Compute-Optimal Large Language Models" (Hoffmann et al.,
DeepMind, 2022). The headline result: **GPT-3 and most models of its era were
massively undertrained.**

### The Kaplan Error

Kaplan's exponents were measured while holding either N or D fixed and varying
the other. Hoffmann's team ran experiments where both varied simultaneously,
optimizing for a fixed compute budget. They got different exponents.

The corrected result: for a given compute budget, model size and dataset size
should scale **equally** — roughly 1:1 in terms of their contribution to
compute.

The rule of thumb that fell out:

```
Optimal tokens = 20 × parameters
```

A 1B parameter model should be trained on ~20B tokens.
A 70B parameter model should be trained on ~1.4T tokens.

### Chinchilla vs Gopher

DeepMind ran a direct test:

| Model     | Parameters | Training Tokens | Compute Used        |
|-----------|-----------|-----------------|---------------------|
| Gopher    | 280B      | 300B            | Same budget         |
| Chinchilla| 70B       | 1.4T            | Same budget         |

Same compute budget. Chinchilla is 4x smaller but trained on 4.7x more data.
Result: **Chinchilla outperforms Gopher on nearly every benchmark.**

The lesson: it is not enough to make the model big. You have to feed it enough
data to justify its size.

### The Undertrained Model Problem in Practice

| Model       | Parameters | Training Tokens | Chinchilla-Optimal Tokens | Assessment            |
|-------------|-----------|-----------------|---------------------------|-----------------------|
| GPT-3       | 175B      | 300B            | ~3.5T                     | severely undertrained |
| Gopher      | 280B      | 300B            | ~5.6T                     | severely undertrained |
| Llama-1 65B | 65B       | 1.4T            | ~1.3T                     | approximately optimal |
| Llama-2 70B | 70B       | 2T              | ~1.4T                     | slightly over optimal |
| Chinchilla  | 70B       | 1.4T            | ~1.4T                     | optimal by design     |

This is why Llama-1 (a smaller model) was competitive with GPT-3 despite using
a fraction of the parameters: it was trained on far more tokens.

### Practical Implication

If you are training a model to be as good as possible given a compute budget C:

1. Do NOT maximize model size.
2. Allocate roughly half the compute to model size, half to data.
3. Expect your model to need ~20 tokens per parameter.

If you are training a model for inference efficiency (you will run it many
times after training), you may want an even smaller model trained on even more
data — because each inference call costs less with a smaller model.

---

## Emergent Abilities

Scaling laws predict loss on next-token prediction. But some abilities do not
scale smoothly — they appear suddenly once a model crosses a size threshold.

These are called **emergent abilities**:

| Ability                       | Approximate Scale Where It Appears |
|-------------------------------|-------------------------------------|
| Few-shot learning             | ~1B parameters                      |
| Chain-of-thought reasoning    | ~60B parameters                     |
| Multi-step arithmetic         | ~100B parameters                    |
| Instruction following         | Requires fine-tuning at scale       |
| Code generation               | ~10B+ parameters                    |

The troubling property of emergent abilities: **you cannot predict them by
extrapolating from smaller models.** A 1B model shows zero chain-of-thought
ability. A 60B model suddenly does it. Nothing in between gives you a warning
that this is coming.

This creates a real challenge for the scaling law framework. Power laws predict
average loss. They do not predict capability thresholds.

Some researchers (Schaeffer et al., 2023) have argued that emergence is partly
a measurement artifact — if you use continuous metrics instead of pass/fail
benchmarks, the improvements look smoother. The debate is ongoing.

---

## What Scaling Laws Mean for the Field

### You Can Predict Training Runs

Before spending $10M on a training run, you can:
1. Run 10–20 small experiments (hours, not months)
2. Fit the power law exponents from your data
3. Extrapolate to your target compute budget
4. Predict final loss within ~10% accuracy

This is now standard practice at large labs.

### The Race Dynamics

Once it was known that loss improves predictably with scale, the implication
was clear: whoever can acquire the most compute and data will have the best
models. This accelerated the "scaling race" that defined 2020–2023.

### The Limits of Scaling

Scaling laws have so far held across many orders of magnitude. But:

- The exponents are small. Even infinite compute gives diminishing returns.
- Data is finite. The internet has a limited amount of high-quality text.
- Some capabilities require more than scale (alignment, reasoning, tool use).

Scaling is necessary but not sufficient.

---

## Summary

| Concept                | Key Takeaway                                              |
|------------------------|-----------------------------------------------------------|
| Kaplan scaling laws    | Loss scales as power law with N, D, and C                 |
| Power law exponents    | αN ≈ 0.076, αD ≈ 0.095, αC ≈ 0.050                      |
| Chinchilla finding     | Previous models were undertrained — too big, too little data |
| Chinchilla-optimal     | ~20 tokens per parameter for fixed compute budget         |
| Emergent abilities     | Some capabilities appear suddenly, not predictable from small scale |
| Practical upshot       | You can predict model performance before training it      |

In the next lesson, we will look at what happens after pretraining: fine-tuning,
RLHF, and instruction following — the steps that turn a language model into an
assistant.

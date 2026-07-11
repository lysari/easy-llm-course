# Lesson 04 — Information Theory

---

## The problem information theory solves

Every LLM training run on Earth minimizes one number: cross-entropy loss. RLHF keeps models sane with a KL penalty. Papers compare models by perplexity. Bits-per-byte decides tokenizer debates.

All four of those are **the same idea wearing different hats**: a way of measuring *surprise*. Information theory — invented by Claude Shannon in 1948, in one paper, essentially complete on arrival — is that measurement system. Learn one concept (surprise) and cross-entropy, KL, and perplexity stop being three formulas and become one.

---

## Step 1: measuring surprise

Which message carries more information?

```
"The sun rose this morning."          you learned nothing
"Your house is on fire."              you learned a LOT
```

Information is inversely related to probability: **rare = surprising = informative**. Shannon defined the surprise of an event with probability p as:

```
surprise(p) = −log₂(p)     measured in bits
```
- `p`: your probability for the event *before* it happened
- `log₂`: logarithm base 2 (that's what makes the unit "bits"; natural log gives "nats")
- the minus sign: log of a number ≤ 1 is negative, so −log makes surprise positive

Concrete values:

```
p = 1.0    surprise = −log₂(1.0)   = 0 bits    (certain → no info)
p = 0.5    surprise = −log₂(0.5)   = 1 bit     (coin flip → exactly 1 bit)
p = 0.25   surprise = −log₂(0.25)  = 2 bits
p = 1/1024 surprise = −log₂(1/1024)= 10 bits
p → 0      surprise → ∞                        (the impossible happened!)
```

Why *log*, and not something else? Because we want surprises of independent events to **add**: two coin flips = 1 bit + 1 bit = 2 bits, and indeed −log₂(0.5 × 0.5) = 2. The log is the only function that turns "probabilities multiply" into "information adds."

---

## Step 2: entropy = average surprise

**Entropy** is the surprise you *expect* per event, on average, given a distribution:

```
H(P) = −Σᵢ pᵢ · log₂(pᵢ)
```
- `P`: a distribution; `pᵢ`: probability of outcome i
- each term: (chance of outcome i) × (surprise if it happens)
- convention: a term with pᵢ = 0 contributes 0

Tiny examples:

```
fair coin:    H = −(0.5·log₂0.5 + 0.5·log₂0.5) = 1.0 bit
rigged coin   H = −(0.9·log₂0.9 + 0.1·log₂0.1)
(90/10):        = −(0.9·(−0.152) + 0.1·(−3.32)) = 0.469 bits
certain coin: H = 0 bits    (never surprised)
fair 8-die:   H = log₂(8) = 3 bits   (uniform = maximum entropy)
```

Entropy is highest when you're most ignorant (uniform) and zero when you're certain. It measures **how unpredictable a source inherently is** — a hard floor: no model, however smart, can average fewer than H bits of surprise on data genuinely drawn from P. English text runs roughly 1–2 bits of entropy per character (Shannon measured this in 1951 by having people guess the next letter — arguably the first language-modeling evaluation ever run).

Everyday analogy: entropy is the *20-questions number*. Guessing 1 of 8 equally likely items takes 3 well-chosen yes/no questions = 3 bits. A distribution's entropy is the average number of ideal yes/no questions needed to pin down an outcome.

---

## Step 3: cross-entropy = average surprise when you believe the wrong thing

Now the key move. Reality generates data from distribution **P**. But your model believes **Q**. Every event's surprise is now priced by *your* beliefs (−log q), while *reality* decides how often each event occurs (weights p):

```
H(P, Q) = −Σᵢ pᵢ · log₂(qᵢ)
```
- `pᵢ`: how often outcome i actually happens (reality)
- `qᵢ`: the probability *your model* assigned to it (belief)
- reads: "expected surprise of a Q-believer living in a P-world"

Fundamental fact:

```
H(P, Q) ≥ H(P)     always — with equality only when Q = P
```

Wrong beliefs cost extra surprise. Concrete:

```
Reality P:  [0.5, 0.5]         Model Q: [0.9, 0.1]

H(P)    = 1.0 bit
H(P,Q)  = −(0.5·log₂0.9 + 0.5·log₂0.1) = 1.737 bits

The overconfident model pays 0.737 extra bits per event —
it's rarely-surprised when right but ENORMOUSLY surprised half the time.
```

**This is the loss LLMs train on.** In training, P is the data ("the actual next token was 'Paris', so all mass on 'Paris'"), Q is the model's softmax output, and the per-token loss collapses to `−log q(actual token)` — exactly the negative log-likelihood from [Lesson 03](../03-probability-and-statistics/lesson.md). Two independent derivations, one formula:

```
statistics view:   minimize cross-entropy  =  maximum likelihood
information view:  minimize cross-entropy  =  minimize average surprise
```

(Frameworks use natural log, so losses come out in nats: 1 nat = 1/ln 2 ≈ 1.443 bits. A GPT loss of "2.3" means 2.3 nats ≈ 3.3 bits of surprise per token.)

---

## Step 4: KL divergence = the price of wrong beliefs

Subtract the unavoidable part (entropy) from the total (cross-entropy) and you get the pure penalty for believing Q instead of P:

```
KL(P ‖ Q) = H(P, Q) − H(P) = Σᵢ pᵢ · log₂(pᵢ / qᵢ)
```
- `KL(P ‖ Q)`: "KL divergence from P to Q" — the *extra* bits per event caused by using the wrong distribution
- always ≥ 0; equals 0 only when P = Q
- **not symmetric**: KL(P‖Q) ≠ KL(Q‖P). It's not a distance, it's a *cost*, and it matters who's reality and who's the believer. (KL(P‖Q) explodes when Q assigns ~0 to something P actually does — believing something is impossible when it isn't is the costliest mistake.)

Where researchers meet KL constantly:

- **RLHF** ([../../lessons/28-rlhf-ppo/lesson.md](../../lessons/28-rlhf-ppo/lesson.md)): the objective is roughly `reward − β·KL(policy ‖ original model)`. The KL term is a leash — "chase reward, but stay within a few bits of the model you were." Remove it and the policy collapses into degenerate reward-hacking gibberish. The leash *is* KL.
- **Distillation**: train a small model to minimize KL from the big model's token distribution.
- **Analysis**: "how much did fine-tuning change the model?" is answered in KL.

---

## Step 5: perplexity = entropy, exponentiated

Cross-entropy in bits is unintuitive ("3.2 bits per token"?). Exponentiate it:

```
perplexity = 2^H(P,Q)        (or e^H if H is in nats)
```

Interpretation — the *effective branching factor*:

> A perplexity of K means the model is, on average, as surprised as if it were choosing uniformly among **K equally likely options** at each step.

```
perfect prediction:      H = 0      → perplexity 1     (no choice at all)
coin-flip uncertainty:   H = 1 bit  → perplexity 2
GPT-2 on WikiText-103:   ~ perplexity 17   "like choosing among ~17 tokens"
random 50k-token guesser:            perplexity 50,000
```

Perplexity is just cross-entropy repackaged for humans, and it's the standard headline number in LM papers. Caution researchers learn early: perplexity depends on the tokenizer, so **models with different tokenizers can't be compared by token perplexity**. The fix:

**Bits per character (BPC) / bits per byte:** divide total surprise bits by total *characters* (or bytes) instead of tokens — tokenizer-independent. `total_bits / num_characters`. When a paper compares across vocabularies, it reports BPC. State-of-the-art LLMs sit well under 1 bit per byte of English — better than Shannon's human guessers scored in 1951.

---

## Compression = prediction: why a good LLM is a good compressor

Shannon's **source coding theorem** says: a source with entropy H bits/symbol cannot be losslessly compressed below H bits/symbol on average — and near-H compression is achievable. The construction: give short codes to probable symbols, long codes to rare ones. The optimal code length for a symbol with probability p is exactly

```
−log₂(p) bits        …the surprise!
```

So "surprise" isn't a metaphor — it is literally *the number of bits an optimal compressor spends on that symbol*. Which yields an identity researchers take seriously:

```
model's cross-entropy on text  =  bits/symbol that model achieves as a compressor
        better predictor       ⇔  better compressor. Same number. Always.
```

An arithmetic coder driven by an LLM's next-token probabilities turns the model *directly* into a lossless compressor whose output size in bits equals the model's total loss on that text. LLMs beat gzip on English text by roughly an order of magnitude — because gzip only models repeated substrings while the LLM models grammar, facts, and context.

This equivalence is a live research lens: "language modeling is compression" (there's a 2023 DeepMind paper with almost that title), and the compression view gives clean intuitions for scaling laws ([../../lessons/22-scaling-laws/lesson.md](../../lessons/22-scaling-laws/lesson.md)) — the loss curve is literally "how many bits/token remain unexplained," decomposable into data entropy (irreducible) + model deficiency (shrinks with scale).

---

## One picture to keep

```
              total average surprise of model Q on data from P
              ┌─────────────────────────────────────────────┐
              │                H(P, Q)  cross-entropy       │
              └─────────────────────────────────────────────┘
              = ┌───────────────────────┐ + ┌───────────────┐
                │  H(P)  entropy        │   │ KL(P ‖ Q)     │
                │  world's inherent     │   │ YOUR model's  │
                │  unpredictability     │   │ wrongness     │
                │  (can't fix)          │   │ (training     │
                └───────────────────────┘   │  fixes this)  │
                                            └───────────────┘
perplexity = 2^cross-entropy         optimal code length = surprise = −log₂ p
```

Training can only ever shrink the right-hand box. When the loss curve flattens, you're either at the entropy floor — or your model class can't get closer. Telling those apart is, quite precisely, research.

---

## Symbol cheat sheet

```
−log₂ p        surprise / self-information / optimal code length (bits)
H(P)           entropy: average surprise under the true distribution
H(P, Q)        cross-entropy: average surprise believing Q in a P-world
KL(P ‖ Q)      extra surprise from believing Q; ≥ 0; asymmetric
PPL = 2^H      perplexity: effective number of equally-likely choices
nat            surprise measured with ln; 1 nat ≈ 1.443 bits
BPC            bits per character — tokenizer-independent comparison
```

---

## Code for this lesson

See [index.ts](index.ts) — measures the entropy of real text's character distribution, computes cross-entropy and KL between concrete distributions (verifying H(P,Q) = H(P) + KL), evaluates a toy character model's perplexity on a string, and shows the compression connection by pricing text in optimal-code bits.

## What's next
[Lesson 05 → Numerical Computing](../05-numerical-computing/lesson.md)

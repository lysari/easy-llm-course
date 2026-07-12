# Lesson 25 — Scaling & Emergence

---

## The problem: you have $10 million. Spend it.

You are training one language model. Compute costs money, and every design
choice is a trade-off inside a fixed budget:

- A **bigger model** (more parameters N) learns more per token — but each
  training step costs more, so you afford fewer tokens.
- **More data** (more tokens D) teaches more — but only if the model is big
  enough to absorb it.

Same budget, two extremes, and a whole frontier in between:

```
budget C (fixed FLOPs)
   ├── giant model, starved of data     N huge, D tiny   → undertrained
   ├── ...                              ← somewhere here is optimal →
   └── tiny model, drowning in data     N tiny, D huge   → saturated
```

Get the split wrong and you waste millions. The astonishing discovery of the
scaling-laws program ([lesson 22 of the LLM track](../../lessons/22-scaling-laws/lesson.md) introduced it) is that this
question has a *quantitative, predictive* answer — loss follows smooth power
laws in N and D, so you can run small cheap experiments, fit a curve, and
**predict the loss of a model 1000× larger before you train it**. GPT-4's
final loss was predicted in advance from models with ~10,000× less compute.

That's the known part. This lesson also covers the contested part (emergence)
and the part that changed underneath everyone in 2024–2025 (test-time compute).

---

## The cost of a training run: C = 6·N·D

First, the accounting identity everything else stands on:

```
C ≈ 6 · N · D

  C : total training compute, in FLOPs (floating-point operations)
  N : parameters
  D : training tokens
  6 : ~2 FLOPs per parameter for the forward pass (one multiply + one add),
      ~4 for the backward pass ([lesson 07](../../lessons/07-backpropagation/lesson.md): backprop costs about 2× forward)
```

Tiny concrete numbers: N = 1M params, D = 20M tokens → C = 6·10⁶·2·10⁷ =
1.2·10¹⁴ FLOPs — about a minute on one modern GPU. Real runs:

```
GPT-2   (2019): N=1.5B,  D≈40B    → C ≈ 3.6·10²¹ FLOPs
GPT-3   (2020): N=175B,  D=300B   → C ≈ 3.2·10²³
Chinchilla:     N=70B,   D=1.4T   → C ≈ 5.9·10²³   ← same C as Gopher (280B)!
frontier (2025): C ~ 10²⁵–10²⁶    → order of $100M+ in compute
```

An H100 GPU does ~10¹⁵ FLOP/s sustained. 10²⁵ FLOPs ≈ 10¹⁰ GPU-seconds ≈
**thousands of GPUs running for months**. That's why this lesson's math is
done before the run, not after.

---

## The loss formula: Kaplan vs Chinchilla

The Chinchilla paper (Hoffmann et al., 2022) fit this functional form to
hundreds of training runs:

```
L(N, D) = E + A/Nᵅ + B/Dᵝ

  L : final cross-entropy loss (nats/token) — lower is better
  E ≈ 1.69 : irreducible loss — the entropy of language itself;
             no model of any size gets below this
  A/Nᵅ     : the "model too small" penalty  (A ≈ 406,  α ≈ 0.34)
  B/Dᵝ     : the "data too little" penalty  (B ≈ 411,  β ≈ 0.28)
```

Read it as: **loss = entropy of language + finite-size tax + finite-data tax**,
each tax shrinking as a power law. Now the budget question is calculus: fix
C = 6ND, minimize L. Substitute D = C/6N and set dL/dN = 0; the optimum works
out to N* ∝ C^0.46, D* ∝ C^0.54 — near-equal exponents, meaning:

> **Scale parameters and data together, roughly 1:1 in compute terms.**
> Numerically, with the fitted constants: **D*/N\* ≈ 20 tokens per parameter.**

The controversy this settled: Kaplan et al. (2020, the original scaling-laws
paper) had concluded N* ∝ C^0.73 — grow the *model* much faster than the data.
The field obeyed: GPT-3 was 175B params on just 300B tokens (1.7 tokens/param).
Chinchilla redid the measurement more carefully (the main fix: tuning the
learning-rate schedule to each run's length instead of reusing one schedule)
and got a different law. Then came the experiment: **Chinchilla, 70B params —
4× smaller than Gopher's 280B — trained on 4× the data at the same compute,
beat Gopher across the board.** GPT-3 had been, by this law, wildly
undertrained: ~10× too big for its data.

Two research lessons in one story:

1. A *measurement methodology* difference (LR schedules) silently changed a
   law the whole field was steering by, for two years. Small-print details
   matter at $100M scale.
2. Modern production models (Llama-3 at ~200 tokens/param, etc.) deliberately
   **overtrain past compute-optimal** — because Chinchilla optimizes *training*
   compute only, and a smaller model trained longer is cheaper at *inference*
   forever after ([lesson 24](../24-efficiency-research/lesson.md)'s concerns, invading this lesson). "Compute-optimal"
   is optimal for exactly one cost model.

And a third, almost comic, one: in 2024 a replication attempt (Besiroglu et
al., Epoch AI) found that the constants *printed in the Chinchilla paper
itself* (A ≈ 406, B ≈ 411, β ≈ 0.28) are inconsistent with the paper's own
data and with its famous 20:1 rule — plug them in and you get ~90 tokens per
parameter. The refit constants (A ≈ 482, B ≈ 2085, α ≈ 0.35, β ≈ 0.37) do
reproduce ≈20:1, and they are what `index.ts` uses. Even the paper about
careful measurement had a measurement bug. Check the constants; always check
the constants.

`index.ts` does the full computation: sweeps budgets from 10¹⁸ to 10²⁶ FLOPs,
numerically solves for (N*, D*) at each, and prints the frontier — you'll see
the ≈20 tokens/param rule fall out of the constants.

---

## Emergence: real jumps or broken rulers?

Loss scales smoothly. But *abilities* seemed not to. The influential
observation (Wei et al., 2022): on tasks like 3-digit arithmetic, models score
~0% across three orders of magnitude of scale, then jump to high accuracy in
one step of scale. "Emergent abilities" — unpredictable, discontinuous,
slightly alarming:

```
accuracy                                 ●──
   │                                    /
   │                                   /
   │  ●────●────●────●────●───●──●────●        ← flat... flat... JUMP
   └──────────────────────────────────────► log(scale)
```

Then Schaeffer, Miranda & Koyejo (2023, *Are Emergent Abilities a Mirage?*)
made a counter-argument that every researcher should internalize, because it's
really an argument about **metrics**:

> The underlying skill improves smoothly. The *metric* is a cliff.

Their case, with numbers you can check in `index.ts`: suppose per-token
accuracy p on arithmetic digits improves smoothly with scale. A 5-digit answer
is scored **exact-match** — all 5 digits right or zero credit:

```
P(exact) = p⁵        p = per-digit accuracy (smooth in scale)

p:        0.20   0.50   0.80   0.90   0.95   0.99
p⁵:       0.000  0.031  0.328  0.590  0.774  0.951
            ↑ invisible progress ↑      ↑ sudden "emergence" ↑
```

The skill grew steadily the whole time; exact-match just can't see progress
until p is already high. Score the same model with a *smooth* metric (per-digit
accuracy, or token edit distance) and the discontinuity largely disappears —
which is what they showed on the majority of claimed emergent abilities.

Where the debate honestly stands (2025/2026):

- **Conceded**: many claimed emergences were metric artifacts; "emergent" is
  now used far more carefully, and the mirage paper permanently raised the bar.
- **Still standing**: some jumps survive smooth metrics (in-context learning
  appearing with induction heads — [lesson 22](../22-interpretability/lesson.md)'s phase transition — is a real
  mechanistic discontinuity; grokking is another). And for *deployment*,
  the cliff can be the truth: if the task is "dial the right phone number,"
  exact-match is the metric reality grades you on. A capability that unlocks
  at some scale is genuinely discontinuous *in effect*, whatever the
  underlying curve did.
- The open question is no longer "is emergence real?" but "**which** jumps
  reflect new internal mechanisms vs. thresholded old ones — and can we detect
  the mechanism forming *before* the ability shows?" (An interpretability
  question. The frontiers connect.)

---

## What scaling can't fix (contested territory)

Honest map of the arguments, not a verdict:

- **The data wall.** Chinchilla says compute-optimal data grows as ~C^0.54.
  The indexed, deduplicated, quality-filtered text of the internet is tens of
  trillions of tokens — frontier runs are *already there*. Responses:
  synthetic data (works remarkably well for code/math where correctness is
  checkable; risk of "model collapse" if models recursively train on their own
  unfiltered outputs), multimodal data, and squeezing more from less (curricula,
  multiple epochs). Whether data or compute is the binding constraint by 2027
  is genuinely disputed.
- **Reasoning.** One camp: scaling plus chain-of-thought keeps buying
  reasoning, no wall in sight. Other camp: autoregressive next-token
  prediction has structural limits — planning, backtracking, novel
  composition — that more of the same won't cross. The honest report: every
  specific "LLMs will never X" benchmark claim has had a bad decade, *and*
  reliability on long multi-step problems still degrades in ways scaling
  alone hasn't fixed. Both statements are true; hold them together.
- **Sample efficiency.** A human sees ~100M words by age 10; GPT-4-class
  models see ~10⁵× more and are worse at learning a genuinely new pattern from
  three examples. Scaling laws describe how loss falls with more data — they
  say nothing about closing this gap, and the gap is not closing. Something
  about how humans learn is algorithmically different. Open.

---

## The new axis: test-time compute

Through 2024–2025 the frontier quietly added a second dial. Instead of only
scaling *training* (bigger N, more D), spend more FLOPs **per query at
inference**: let the model generate long internal chains of thought, explore,
backtrack, self-check — trained via RL on problems with checkable answers
(the o1/o3, R1 line of reasoning models).

```
old picture:  capability = f(training compute)
new picture:  capability = f(training compute, thinking time per question)

                    accuracy on hard math
                          │        ── more thinking tokens ──►
     bigger base model ▲  │      ╱──────
                          │  ╱───          two ways to buy the
                          │ ╱               same point on the curve
                          └────────────────►
```

Why this matters for scaling economics: it converts a *capital* cost (train a
bigger model once) into a *marginal* cost (pay per hard question), and the
exchange rate is startling — on competition math, letting a small model think
longer can beat a much larger model answering instantly. It also partially
sidesteps the data wall: RL on verifiable problems generates its own training
signal.

Known honestly: log-linear gains on math/code/verifiable domains are real and
reproducible. Open: whether it transfers to domains without a checker (essays,
strategy, taste), whether models' visible chains of thought reflect their
actual computation (an interpretability question — and current evidence says
sometimes not: models can produce unfaithful reasoning), and what the *joint*
scaling law L(N, D, thinking-tokens) looks like. Nobody has published a
Chinchilla for reasoning yet. That paper is waiting for someone.

---

## What this means for a new researcher

Scaling-law *methodology* — fit small, predict big — is one of the most
laptop-friendly research styles that exists, because the entire point is that
small runs predict large ones. You can:

- fit L(N, D) on your own tiny GPTs ([lesson 14](../../lessons/14-tiny-gpt/lesson.md)) across 4–5 sizes;
- test metric-induced emergence on any task you can score two ways;
- study data-quality scaling (does filtering shift the curve or just its
  constant?) at toy scale.

More in [lesson 26](../26-open-problems/lesson.md), which is entirely about where a newcomer can attack.

---

## Code for this lesson

See [index.ts](index.ts) — two experiments:

1. **The compute-optimal frontier.** Takes the Chinchilla loss formula with
   published constants, sweeps budgets 10¹⁸ → 10²⁶ FLOPs, numerically finds
   the loss-minimizing (N, D) split at each budget, and prints the frontier
   table — watch tokens/params come out ≈ 20 at every scale, and check the
   real Chinchilla run (70B, 1.4T) lands on the line.
2. **The emergence mirage.** One smoothly improving per-digit skill, scored
   three ways (exact-match-of-5, per-digit accuracy, partial credit) — same
   model, one metric shows a jump, the others show the truth.

```
npx ts-node AI-researcher-lessons/25-scaling-and-emergence/index.ts
```

## What's next

You now have the frontier map: what we understand (scaling), what we argue
about (emergence), what nobody has cracked. Time to make it personal — the
full list of open problems, and where someone with a laptop can actually
contribute: [Lesson 26 → Open Problems in AI](../26-open-problems/lesson.md)

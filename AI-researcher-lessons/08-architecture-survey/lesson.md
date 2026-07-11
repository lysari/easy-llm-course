# Lesson 08 — Architecture Survey: Why Transformers Won

---

## The problem this lesson solves

Every model you've built in the from-scratch track is a transformer or a piece of
one. But the transformer (2017) is *recent*. Before it, researchers spent decades
building other architectures — and most of the field's hard-won lessons live in
why those architectures rose and fell.

The researcher's question this lesson answers:

> **An architecture is a set of assumptions about data. Which assumptions, and
> what do they cost?**

That framing has a name — **inductive bias** — and it is one of the most useful
concepts in all of ML research. Master it and every new architecture paper you
ever read becomes easy to place: "what does this design *assume*, and what does
it buy?"

---

## The analogy: hiring for a job

You're hiring someone to find patterns in your data. Four candidates:

- **The generalist** (MLP): brilliant, assumes nothing, must be taught everything
  from zero. Expensive to train because they rediscover the obvious.
- **The local inspector** (CNN): assumes "what matters is what's nearby, and the
  same patterns repeat everywhere". Fantastic for images. Useless assumptions for
  some other jobs.
- **The note-taker** (RNN): reads one item at a time, left to right, keeping a
  single notepad of memory. Anything important must survive on the notepad until
  it's needed — and the notepad is small and gets rewritten at every step.
- **The committee** (Transformer): everyone looks at everything simultaneously
  and decides what's relevant. Expensive per meeting, but nothing gets forgotten
  and everyone works in parallel.

**Inductive bias** = the assumptions a candidate walks in with. Strong correct
assumptions → learns fast from little data. Strong *wrong* assumptions → ceiling
you can't fix with more data. Weak assumptions → needs mountains of data, but no
ceiling.

---

## Candidate 1: the MLP — no structure at all

The multi-layer perceptron ([Lesson 06 — Neural Network](../../lessons/06-neural-network/lesson.md)
in the from-scratch track): stack of `output = activation(W·input + b)` layers.
Every input connects to every neuron.

```
input:  x₁  x₂  x₃  x₄        every line is a separate learned weight
         \\ | X X | //
          [ neurons ]
         // | X X | \\
          [ neurons ]
```

**Inductive bias: none.** The MLP doesn't assume position 1 relates to position 2
any more than to position 400. Consequences:

- Feed it a 224×224 image = 150,528 raw inputs → first layer alone needs billions
  of weights. Nothing is shared, nothing is reused.
- Shift the image one pixel right → to the MLP this is a completely different
  input. It must re-learn "cat" at every position separately.

Verdict: universal in theory (an MLP can approximate any function), hopeless in
practice for structured data at scale. But note — it never disappeared: the FFN
inside every transformer block **is an MLP**
([Lesson 15 — FFN](../../lessons/15-ffn/lesson.md)). It survives as the
general-purpose compute unit *inside* structured architectures.

---

## Candidate 2: the CNN — locality + weight sharing

The convolutional network encodes two assumptions about images:

1. **Locality**: a pixel's meaning depends mostly on its neighbors.
2. **Translation invariance**: an edge detector useful at the top-left is useful
   everywhere — so learn ONE small filter and slide it across the whole image.

```
image:                    the SAME 3×3 filter slides everywhere:
┌──────────────┐
│ ▒▒░░░░░░░░░░ │           ┌───┐        weights per filter: 9
│ ▒▒▒▒░░░░░░░░ │           │ f │→→      weights an MLP would need
│ ░░▒▒▒▒░░░░░░ │           └───┘        for the same layer: millions
│ ░░░░▒▒▒▒░░░░ │
└──────────────┘          stack layers → each layer sees a wider area
                          (edges → textures → parts → objects)
```

The payoff is enormous: **the assumptions are true for images**, so CNNs learn
from far less data with far fewer parameters, and dominated computer vision from
2012 (AlexNet) onward. This is inductive bias working *for* you.

The cost: a CNN's receptive field grows slowly — relating two distant pixels
takes many layers. And for text, "the same pattern matters everywhere within a
small window" is only sort-of true. CNNs were tried for language (they train in
parallel!) but long-range dependencies remained awkward.

---

## Candidate 3: the RNN/LSTM — memory that flows through time

The recurrent network's assumption: **data is a sequence, so process it in
order, carrying a running memory.**

```
        "the"      "cat"      "sat"      "on"
          ↓          ↓          ↓          ↓
h₀ ──→ [cell] ──→ [cell] ──→ [cell] ──→ [cell] ──→ ...
        h₁         h₂         h₃         h₄

h_t = tanh(W_hh · h_{t-1} + W_xh · x_t)

h_t    : hidden state (the notepad) after reading token t
x_t    : the token read at step t
W_hh   : how the old notepad transforms into the new one (learned)
W_xh   : how the new token gets written onto the notepad (learned)
tanh   : squashing activation, keeps values in (−1, 1)
```

Elegant! One cell, reused at every step, handles sequences of any length. RNNs
(especially the LSTM variant) powered translation, speech recognition, and the
first impressive text generators of the 2010s.

But two structural flaws killed them at scale:

### Flaw 1: vanishing (and exploding) gradients

For token 50 to influence learning about token 1, the gradient must travel back
through 49 chained steps. Each step multiplies the gradient by roughly the same
Jacobian matrix `J ≈ diag(tanh′) · W_hh`:

```
gradient at step 1 ∝ J · J · J · ... · J  (49 times) · gradient at step 50

if the "size" (largest stretch factor) of J is 0.9:  0.9⁴⁹ ≈ 0.006
if it is 0.5:                                        0.5⁴⁹ ≈ 2 × 10⁻¹⁵  (gone)
if it is 1.5:                                        1.5⁴⁹ ≈ 4 × 10⁸   (explodes)
```

Multiplying 49 numbers together is a razor's edge: anything not almost exactly
1.0 either vanishes or explodes exponentially. So plain RNNs simply cannot learn
long-range dependencies. (This lesson's code measures this — you'll watch the
gradient norm collapse over 50 steps.)

The **LSTM** (1997) was a brilliant patch: it adds a separate "cell state" that
flows through time via *addition* controlled by learned gates (forget / input /
output), instead of repeated matrix multiplication. Addition doesn't shrink
gradients the way multiplication does — the same insight that later became the
transformer's **residual connections**
([Lesson 13 — Transformer Block](../../lessons/13-transformer-block/lesson.md)).
LSTMs stretched usable memory from ~10 tokens to ~100s. But a patch is a patch:
the information still has to *survive the trip* through every intermediate step.

### Flaw 2: no parallelism

`h₂` needs `h₁`; `h₃` needs `h₂`. Training is inherently **sequential** in
sequence length — a 1000-token document takes 1000 dependent steps. GPUs are
parallel machines; RNNs feed them one crumb at a time. As datasets grew to
billions of tokens, this became the fatal bottleneck: it wasn't that LSTMs
couldn't model language — they couldn't be *trained fast enough* to consume the
data that was available.

---

## Candidate 4: the Transformer — attention only

The 2017 paper title said it plainly: *"Attention Is All You Need."* Drop the
recurrence entirely; let every token attend directly to every other token
([Lesson 12 — Attention](../../lessons/12-attention/lesson.md)).

```
RNN — token 50 reaches token 1 through 49 hops:
  x₁ → h₁ → h₂ → h₃ → ... → h₄₉ → h₅₀     path length: 49
                                           gradient: shrinks 49×

Transformer — token 50 reaches token 1 in ONE hop:
  x₁  x₂  x₃  ...  x₄₉  x₅₀
   ↑___________________|                   path length: 1
        attention                          gradient: undiminished
```

Why it won — three compounding advantages:

1. **Direct long-range connections.** The gradient path between ANY two tokens
   has length 1. No vanishing over distance; "remembering" token 1 at step 50
   needs no relay race.
2. **Parallel training.** During training, all positions' outputs are computed
   at once as matrix multiplies over the whole sequence — no step-by-step
   dependency. A 1000-token document is one big batch of matrix math instead of
   1000 sequential steps. (Generation is still one-token-at-a-time — that's why
   the KV cache exists, [Lesson 21](../../lessons/21-kv-cache/lesson.md).)
3. **It scales with hardware.** Matrix multiplication is exactly what GPUs/TPUs
   are built to do. More chips → bigger transformer → better model, smoothly —
   the empirical regularity behind
   [Lesson 22 — Scaling Laws](../../lessons/22-scaling-laws/lesson.md).

The price it pays:

- **Cost is quadratic**: every token attends to every token → sequence length T
  costs T² attention scores. (A whole efficiency-research subfield exists to
  attack this.)
- **Weak inductive bias**: attention assumes almost nothing — not locality, not
  order (order must be injected via positional encodings,
  [Lesson 25 — RoPE](../../lessons/25-rope-positional-encoding/lesson.md)).
  By the CNN lesson, weak bias should mean "needs mountains of data"... and that
  is *exactly* what happened. Transformers underperform CNNs on small image
  datasets and overtake them when data is huge (the Vision Transformer result).

---

## The scorecard

| | MLP | CNN | RNN/LSTM | Transformer |
|---|---|---|---|---|
| Inductive bias | none | locality + weight sharing | sequential memory | almost none (+ positions) |
| Long-range path between tokens | 1 (but no weight sharing) | many layers | T steps | **1 step** |
| Trains in parallel over sequence | — | yes | **no** | yes |
| Cost in sequence length T | — | ~T | ~T | **T²** |
| Data appetite | huge | modest | modest | huge |
| Killed by | parameter explosion | long-range awkwardness | vanishing grads + no parallelism | (quadratic cost — so far) |

The pattern a researcher should extract:

> **The transformer didn't win by having the best assumptions. It won by having
> the FEWEST assumptions at the exact moment data and compute became abundant
> enough to replace assumptions with learning.**

Strong inductive bias is a loan: it pays off when data is scarce and charges
interest when data is plentiful. The history of ML architectures is largely the
history of that loan being repaid.

---

## The researcher's takeaway questions

When you read any architecture paper (and you'll read many in Phase C), ask:

1. What does this design **assume** about the data? (its inductive bias)
2. What is the **gradient path length** between things that must interact?
3. Can it **train in parallel** on today's hardware?
4. How does cost **scale** with input size?
5. In which **data regime** (scarce ↔ abundant) do its assumptions pay off?

These five questions explain the rise and fall of nearly every architecture of
the last 30 years — and they'll let you evaluate the next one before the
benchmark numbers come in.

---

## Code for this lesson

See [index.ts](index.ts) — implements a real tiny RNN forward pass, then
measures the vanishing-gradient effect directly: it multiplies the RNN's
step-to-step Jacobians across 50 timesteps and prints the gradient norm
collapsing to ~10⁻¹⁴ (and exploding when `W_hh` is scaled up) — versus
attention's one-step connection.

Run it:
```
npx ts-node index.ts
```

## What's next

Transformers won because they scale. So what actually happens when you scale
one — what breaks first, and what machinery keeps a 175B-parameter training run
alive? [Lesson 09 → Training at Scale](../09-training-at-scale/lesson.md)

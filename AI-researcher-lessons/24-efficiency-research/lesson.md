# Lesson 24 — Efficiency Research

---

## The problem: intelligence has a unit price

Take a 70-billion-parameter model. In float32, the weights alone are
**280 GB** — they do not fit on any single GPU you can buy. Every token it
generates touches every weight: ~140 GFLOPs *per token*. Serving it to millions
of users multiplies that by billions of tokens a day. Training it in the first
place costs millions of dollars of compute ([lesson 25](../25-scaling-and-emergence/lesson.md) will price this out exactly).

So efficiency looks like an engineering chore: make number go down. Why call it
a *research* frontier?

Because every efficiency technique is secretly a **scientific claim about what
is and isn't essential in a network** — and most of those claims are unproven:

- Quantization works ⇒ most of the information in 32-bit weights is *noise*.
  How much is signal? Why do a few dimensions carry outsized importance?
- Distillation works ⇒ a big model's knowledge *fits* in a small one, and the
  big model can teach it better than the raw data can. Why?
- Pruning works ⇒ trained networks are mostly *empty*. Then why did we need
  the full size during training?

Nobody fully knows the answers. That's what makes it a frontier: the
compression techniques are also **probes into what neural networks actually
learn**.

---

## Analogy: the moving truck

Your model is a house you must move across the country (deploy it).

- **Quantization**: photograph every item at lower resolution — keep every
  item, describe each with fewer bits.
- **Pruning**: throw away furniture you never use. (Surprise: it's most of it.)
- **Distillation**: don't move the house at all — have the owner *teach* a
  new, smaller household how to live like them.
- **LoRA**: you only redecorated three rooms — ship just the redecoration.
- **Speculative decoding / FlashAttention**: same house, same truck — a
  driver who plans the route so the truck never idles.

Different techniques, one common enemy. Which brings us to the one number that
explains almost everything in this lesson:

---

## The number that rules inference: memory bandwidth

A modern GPU can do ~10¹⁵ multiply-adds per second but only read ~10¹² bytes
per second from memory. Ratio: **~1000 FLOPs per byte moved.** Generating one
token uses each weight in ~2 FLOPs — so at generation time the GPU spends its
life **waiting for weights to arrive from memory**, not computing.

```
             compute units: ████████████████  (mostly idle)
                                  ▲
             memory bus:     ─────┴─────  ← the actual bottleneck
                             weights streaming in
```

Consequence: **shrinking bytes ≈ speeding up inference**, almost 1:1. Halve
the bits, roughly double the tokens/sec. Keep this in mind — every section
below is attacking bytes-moved, FLOPs, or both.

---

## Quantization: float32 → int8 → int4

Store each weight in 8 (or 4) bits instead of 32. The standard scheme is
**affine quantization** — a linear map between real values and small integers:

```
q = round(w / s) + z          (quantize:   real → integer)
ŵ = s · (q − z)               (dequantize: integer → approximate real)

  w : original float32 weight
  q : stored integer, e.g. int8 ∈ [−128, 127]
  s : scale — the real-value size of one integer step
  z : zero-point — which integer represents 0.0
  ŵ : reconstructed weight; |w − ŵ| ≤ s/2 (rounding error)
```

Choosing `s` for a weight tensor with values in `[min, max]`, for int8:

```
s = (max − min) / 255        e.g. weights in [−0.42, 0.38]:
z = round(−min / s) − 128         s = 0.8/255 ≈ 0.0031
                                  → worst-case error ≈ 0.0016 per weight
```

Tiny concrete example, 1 weight: `w = 0.2337`, `s = 0.0031`, `z = 7`:
`q = round(0.2337/0.0031) + 7 = 75 + 7 = 82`. Stored: one byte, `82`.
Reconstructed: `ŵ = 0.0031 · (82 − 7) = 0.2325`. Error `0.0012`. Multiply that
by 70 billion and it mostly... cancels out. Mostly.

**What breaks, and why it's research:**

- **Outliers.** In large transformers (>~6B params), a few activation
  dimensions carry values 100× larger than the rest. One outlier stretches
  `[min, max]`, making `s` huge, crushing all normal values into a few integer
  buckets. Discovering this (Dettmers et al., `LLM.int8()`, 2022) and asking
  *why trained transformers concentrate information in a few directions* turned
  a systems trick into a science question — still not fully answered.
- **Fixes are active research**: per-channel scales, keeping outlier channels
  in fp16, rotating the basis so outliers spread out (QuIP, 2023), weighting
  error by activation importance (GPTQ 2022, AWQ 2023).
- **int4 and below**: works shockingly well *with* the tricks above; naive
  int4 visibly damages the model. At 2 bits, quality collapses — apparently
  ~4 bits per weight of true information is close to what today's training
  actually stores. Why that number? Open question.

`index.ts` quantizes a weight matrix to int8 and int4 and shows the model's
*output* drift — reconstruction error per weight is easy; what matters is how
error compounds through layers.

---

## Distillation: the student and the soft targets

Train a small **student** to imitate a big **teacher** (Hinton et al., 2015).
The trick is *what* the student imitates — not the dataset's hard labels, but
the teacher's full output distribution, its **soft targets**:

```
input: photo of a truck
hard label:              [cat: 0, dog: 0, car: 0,   truck: 1  ]
teacher's soft targets:  [cat: 0.01, dog: 0.02, car: 0.30, truck: 0.67]
                                                  ▲
                            "dark knowledge": trucks resemble cars,
                             and do NOT resemble cats — the hard label
                             contains none of this
```

Loss: cross-entropy between student and teacher distributions, usually with a
**temperature** `T` softening both ([lesson 23 of the LLM track](../../lessons/23-sampling-strategies/lesson.md) uses the same
knob) so the small probabilities — where the relational knowledge lives — get
amplified:

```
p_i(T) = exp(logit_i / T) / Σ_j exp(logit_j / T)

  T = 1: normal softmax     T > 1 (e.g. 3): flattened — small probs magnified
```

Why soft beats hard: each hard label carries at most log₂(V) bits ("the answer
is truck"). A soft distribution carries the teacher's entire learned *geometry
of similarity* on every single example — per-example gradient information the
raw dataset simply does not contain. The student effectively gets a better
curriculum than the teacher had.

This is now a primary production technique for LLMs (frontier "mini/flash"
models are distillations), and the research questions are open: how much
capability *can* fit in the student? What is lost first — knowledge, reasoning,
calibration? Your `index.ts` shows the core effect: on the same 20 training
points, a soft-target student generalizes better than a hard-label one.

---

## Pruning and the lottery ticket hypothesis

Fact, replicated for decades: delete the 90% smallest-magnitude weights of a
trained network, finetune briefly, and accuracy barely moves. Trained networks
are mostly air.

The scientific puzzle is sharper than the trick. **The lottery ticket
hypothesis** (Frankle & Carlin, 2018): inside the big randomly-initialized
network there exists a small subnetwork (a "winning ticket") which — trained
*alone from its original initialization* — matches the full network. Training
the same architecture from a *fresh* random init fails. Suggested picture:

```
big random init  =  millions of lottery tickets (random subnetworks)
training         =  finding + amplifying a winning ticket
the other 90%    =  losing tickets, needed only to buy enough tickets
```

If we could find winning tickets *before* training, we'd cut training cost
~10×. Nobody knows how (as of 2025/2026). Caveats, honestly: at true LLM scale
the strict claim weakens (you need to rewind to early-training weights, not
init), and unlike quantization, unstructured pruning is hard to turn into real
speedups on GPUs (sparse memory access defeats the hardware). So pruning today
is more scientifically interesting than practically dominant.

---

## LoRA: don't touch the weights at all

Recap from finetuning ([lesson 26 of the LLM track](../../lessons/26-instruction-finetuning/lesson.md)): to adapt a model, you
don't need to change all of `W` — the *change* can be low-rank:

```
W' = W + ΔW  ≈  W + B·A       W: d×d frozen   (e.g. 4096×4096 = 16.8M params)
                              A: r×d, B: d×r  (r = 8 → 65K params, 0.4%)
```

The efficiency framing: the *diff* between "general model" and "your model" is
tiny, so ship the diff. One base model + a thousand cheap LoRA adapters —
this is why finetuning is accessible at all without datacenter money. The
research edge: *why* is the adaptation low-rank? What tasks aren't? (Related
finding, same spirit as lottery tickets: task finetuning lives in a
surprisingly low-dimensional subspace.)

---

## Inference-time research

**Speculative decoding.** Generation is sequential: one token per full forward
pass of the big model — and each pass is memory-bound (see above), so the GPU
idles. Trick: let a *small* draft model race ahead `k` tokens, then have the
big model check all `k` **in one pass** (verifying k tokens in parallel costs
about the same as generating 1, because the weights stream in once either way):

```
draft (cheap):    "the  cat  sat  on  the  mat"     ← guesses 6 tokens
big model (1 pass): ✓    ✓    ✓    ✓    ✗           ← accepts 4, fixes the 5th
                                                       net: 5 tokens for ~1 pass
```

The elegant part: with the proper accept/reject rule, the output distribution
is **mathematically identical** to the big model sampling alone — pure speed,
zero quality change. Speedup ≈ 2–3× when the draft agrees often (easy text:
often; weird text: less).

**FlashAttention (Dao et al., 2022).** Attention ([lesson 12](../../lessons/12-attention/lesson.md)) builds a T×T
score matrix. At T = 32K, that's a billion floats — and the *real* cost isn't
computing them, it's writing them to slow GPU memory and reading them back.
FlashAttention's idea: **never materialize the T×T matrix.** Process K/V in
tiles that fit in the GPU's tiny fast on-chip memory, keep a running softmax
(a running max and running sum let you renormalize incrementally — same
log-sum-exp trick as [lesson 11](../../lessons/11-softmax/lesson.md)), and write out only the final T×d output:

```
naive:  compute S = QKᵀ (T×T) → write to slow memory → read back → softmax →
        write → read → multiply by V           memory traffic: O(T²)
flash:  for each tile of K,V: update running softmax + running output
        in fast on-chip memory                 memory traffic: O(T·d)
```

Same math, same exact result, 2–4× faster and O(T) memory — this one idea is
a big part of why 100K+ contexts exist. The research lesson generalizes:
**counting FLOPs is the wrong model of cost; counting memory movement
("IO-awareness") is the right one.** Whole families of papers now follow.

**Architecture-level efficiency.** Mixture-of-Experts — activate only a few
expert FFNs per token, decoupling parameter count from per-token FLOPs — you
built one in [lesson 29 of the LLM track](../../lessons/29-mixture-of-experts/lesson.md). Same frontier, attacked from the
architecture side.

---

## The research frame

Every technique here started as someone asking a *why* question about waste:

| Technique | The scientific claim underneath | Status |
|---|---|---|
| Quantization | ≲4–8 bits/weight of real information | why? outliers why? open |
| Distillation | small models can hold it, given a teacher | limits unknown |
| Pruning / tickets | 90% of weights are scaffolding | can't find tickets early |
| LoRA | adaptation is low-rank | why? when not? |
| Speculative decoding | easy tokens don't need the big model | which tokens are easy? |
| FlashAttention | cost = memory movement, not FLOPs | spawned a field |

And all of it runs at laptop scale first. GPTQ, AWQ, and speculative decoding
were validated on single machines before anyone scaled them.

---

## Code for this lesson

See [index.ts](index.ts) — two experiments:

1. **Quantization**: a weight matrix → int8 and int4 with scale/zero-point;
   prints per-weight reconstruction error, then runs a tiny 2-layer network in
   full precision vs quantized and measures *output* drift (the number that
   actually matters).
2. **Distillation**: a hand-built teacher with a smooth 2-D decision boundary;
   two identical students train on the same 20 points — one on hard labels,
   one on the teacher's soft targets — and we test both on a dense grid. Soft
   wins.

```
npx ts-node AI-researcher-lessons/24-efficiency-research/index.ts
```

## What's next

Efficiency asks how to do the same with less. The opposite question — what
happens when you do *more*: more parameters, more data, more compute — has an
unreasonably precise answer, and a controversy attached:
[Lesson 25 → Scaling & Emergence](../25-scaling-and-emergence/lesson.md)

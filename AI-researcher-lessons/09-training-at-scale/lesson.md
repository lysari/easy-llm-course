# Lesson 09 — Training at Scale

---

## The problem this lesson solves

You've trained tiny models on your laptop. GPT-3 has 175,000,000,000 parameters.

The naive plan — "same code, bigger matrices" — dies instantly, and it's worth
seeing exactly *where*:

```
GPT-3: 175 billion parameters, trained with Adam in float32

parameters:      175e9 × 4 bytes  =  700 GB
gradients:       175e9 × 4 bytes  =  700 GB
Adam m (momentum): 175e9 × 4 bytes  =  700 GB
Adam v (variance): 175e9 × 4 bytes  =  700 GB
                                    ────────
just to hold the training state:    2,800 GB

one big GPU (A100):                    80 GB   ← oops. 35× too small.
```

Not "slow" — **impossible on any single device that exists**. Everything in this
lesson is the machinery researchers invented to make it possible anyway. And this
isn't a sideshow: at the frontier, systems knowledge *is* research knowledge —
what experiments you can afford determines what science you can do.

---

## The analogy: one chef vs. a kitchen brigade

Training a tiny model is one chef with one cutting board.
Training an LLM is catering a 100,000-guest banquet:

- The recipe binder alone doesn't fit on one counter → **split the model** (tensor/pipeline parallelism)
- Many identical cooking stations working on different orders → **data parallelism**
- Write shopping notes in shorthand, keep the master recipe precise → **mixed precision**
- Counter too small to hold every intermediate dish → **throw some away and re-cook when needed** (gradient checkpointing)
- Can't fit a full batch in one pan → **cook in small pans, combine at the end** (gradient accumulation)

Let's take these one at a time, with numbers.

---

## Step 1: know your memory budget (the four-slot rule)

Training memory for the model state has four big slots (activations come later):

```
memory = params + gradients + optimizer state

with Adam (see ../../lessons/17-adam-optimizer/lesson.md), optimizer state
is TWO extra copies: m (momentum) and v (variance) — both kept in float32.
```

Bytes per parameter by dtype:

```
float32 (fp32) : 4 bytes   ← full precision, the safe default
float16 (fp16) : 2 bytes   ← half precision, range is narrow (max ~65,504)
bfloat16 (bf16): 2 bytes   ← half precision, fp32-sized range, less detail
int8           : 1 byte    ← inference tricks, not standard training
```

The standard **mixed-precision** recipe (what real LLM training uses):

```
per parameter:
  fp16/bf16 working weights        2 bytes   (used for the fast math)
  fp16/bf16 gradients              2 bytes
  fp32 master weights              4 bytes   (the precise "master recipe")
  fp32 Adam m                      4 bytes
  fp32 Adam v                      4 bytes
                                  ────────
                                  16 bytes per parameter
```

**Rule of thumb: mixed-precision Adam training ≈ 16 bytes × parameter count.**
GPT-2 (124M): ~2 GB. GPT-3 (175B): ~2.8 TB. The code for this lesson is a
calculator that prints this breakdown for a range of model sizes.

Why keep fp32 masters at all? Each update step adds a *tiny* value
`η·(gradient stuff)` to a weight. In fp16, `1.0 + 0.0001 = 1.0` — the update
literally rounds away and learning silently stops. The fp32 master copy is where
updates accumulate safely; it gets re-rounded to fp16 for the next forward pass.
(This is the [numerical computing](../05-numerical-computing/lesson.md) lesson
striking back at scale.)

And on top of the four slots sit **activations** — every layer's intermediate
outputs, remembered for backprop
([Lesson 07 — Backpropagation](../../lessons/07-backpropagation/lesson.md)).
Their size scales with `batch × sequence length × layers × width` and often
rivals the model state itself. Hold that thought for checkpointing.

---

## Step 2: gradient accumulation — big batches on small hardware

LLMs train with enormous batches (GPT-3: 3.2M tokens per batch). Your GPU fits a
tiny slice of that. The fix is beautifully simple — addition is associative:

```
want:  gradient of a batch of 4 examples = ¼ (g₁ + g₂ + g₃ + g₄)

do:    run example 1 → keep its gradient   (don't update!)
       run example 2 → ADD its gradient
       run example 3 → ADD
       run example 4 → ADD, divide by 4 → NOW update once

the result is mathematically IDENTICAL to the full batch
```

Memory needed: one microbatch at a time. Cost: 4 sequential forward/backward
passes instead of 1 parallel one — you pay in time, not memory. Every large-model
training loop you'll ever read has an `accumulation_steps` loop; this lesson's
code proves the equivalence numerically to ~15 decimal places.

## Step 2b: gradient clipping — the seatbelt

Once in a while a batch produces a freak gradient (bad data, fp16 hiccup, sharp
landscape region — [Lesson 06](../06-optimization-landscape/lesson.md)) that
would fling the weights into the wilderness and spike the loss. The fix:

```
g_norm = ‖g‖ = √(Σ gᵢ²)          total length of the whole gradient vector
if g_norm > c:   g ← g · (c / g_norm)     (shrink to length c, keep direction)

c : the clip threshold — 1.0 in GPT-3, and in most LLM configs you'll see
```

It costs almost nothing and prevents rare catastrophic steps. Loss-spike
forensics ("was it data? was it precision?") is a genuine subfield of large-run
babysitting.

---

## Step 3: the three parallelisms

One device is never enough, so we split the work. There are exactly three axes
along which to split, and real systems ("3D parallelism") use all three at once.

### Data parallelism — clone the model, split the batch

```
            batch of 512 examples
          ┌──────────┬──────────┬──────────┬──────────┐
          │  128     │  128     │  128     │  128     │
          ▼          ▼          ▼          ▼
      ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐
      │ GPU 0 │  │ GPU 1 │  │ GPU 2 │  │ GPU 3 │   each holds a FULL copy
      │ model │  │ model │  │ model │  │ model │   of the model
      └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘
          └────────── average gradients ───────────┘   ("all-reduce")
                     then everyone updates identically
```

Simple, scales to thousands of GPUs — but every GPU must FIT the whole model, so
by itself it can't train anything bigger than one device's memory. (The ZeRO/FSDP
family fixes that by *sharding* the four memory slots across the data-parallel
GPUs — each holds 1/Nth of the optimizer state, gathering pieces on demand.)

### Tensor parallelism — split every layer's matrices

```
one giant matmul  X @ W, where W is (12288 × 49152)  ← a real GPT-3 FFN layer

split W by columns across 4 GPUs:
      W = [ W₀ | W₁ | W₂ | W₃ ]        each Wᵢ is (12288 × 12288)

GPU 0: X @ W₀ → first quarter of the output
GPU 1: X @ W₁ → second quarter          ... all four run SIMULTANEOUSLY
then concatenate the pieces.
```

Now no GPU ever holds a full layer. The cost: GPUs must exchange activations
*inside every single layer*, so this needs the fastest interconnect available —
tensor parallelism usually stays within one machine (8 GPUs on NVLink).

### Pipeline parallelism — split by layers

```
96 layers, 4 GPUs → 24 layers each:

  GPU 0: layers  1–24 ──→ GPU 1: layers 25–48 ──→ GPU 2: 49–72 ──→ GPU 3: 73–96

naive version: only ONE gpu works at a time (the "bubble"):
  GPU0 ████░░░░░░░░░░░░
  GPU1 ░░░░████░░░░░░░░
  GPU2 ░░░░░░░░████░░░░
  GPU3 ░░░░░░░░░░░░████     ← 75% idle!

fix: split the batch into microbatches and stream them through:
  GPU0 █₁█₂█₃█₄░░░░
  GPU1 ░█₁█₂█₃█₄░░░
  GPU2 ░░█₁█₂█₃█₄░░
  GPU3 ░░░█₁█₂█₃█₄░     ← bubble shrinks as microbatch count grows
```

Only layer-boundary activations cross GPUs (cheap traffic), so pipeline stages
can even live in different machines. The bubble never fully disappears — one of
many throughput-vs-memory-vs-bandwidth tradeoffs schedulers juggle.

### Putting it together

```
GPT-3-scale recipe (schematic):
  tensor parallel  ×8   (within a machine, NVLink)
  pipeline parallel ×8  (across machines)
  data parallel    ×16  (across machine groups)
  total: 8 × 8 × 16 = 1024 GPUs on one training run
```

---

## Step 4: gradient checkpointing — trade compute for memory

Backprop's contract: remember every intermediate activation from the forward
pass. For a 96-layer model on long sequences, that's a mountain of memory.

The trick — remember only every k-th layer's output ("checkpoints"), throw the
rest away, and **recompute** the missing pieces during the backward pass:

```
forward, normal:      save a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ ... (all 96)
forward, checkpointed: save a₈ a₁₆ a₂₄ ...              (every 8th)

backward reaches layer 13?
  → rerun forward from checkpoint a₈ to rebuild a₉..a₁₆, then continue backprop
```

Cost: roughly one extra forward pass (~33% more compute).
Payoff: activation memory drops from ~L to ~√L per layer with the right spacing.
Nearly every big training run has this switched on — compute is easier to buy
than memory-per-device.

---

## Why researchers must care (not just engineers)

- **The experiment budget is the science budget.** Halve memory per run and you
  can run twice the ablations ([Lesson 18 — Baselines & Ablations](../18-baselines-and-ablations/lesson.md)).
- **Scaling laws are measured under these constraints** — compute-optimal model
  sizing ([Lesson 22](../../lessons/22-scaling-laws/lesson.md)) only makes sense
  once you know what a petaflop-day actually buys through this machinery.
- **Precision is a live research area**: bf16 → fp8 training, quantized
  inference, optimizer-state compression — each is a paper trail of "how low can
  the bytes go before the loss curves diverge".
- **Architecture co-evolves with hardware** ([Lesson 08](../08-architecture-survey/lesson.md)):
  MoE models ([Lesson 29](../../lessons/29-mixture-of-experts/lesson.md)) exist
  precisely to decouple parameter count from per-token compute.

---

## Code for this lesson

See [index.ts](index.ts) — two demos:

1. **A training-memory calculator**: for GPT-2 124M → GPT-3 175B, prints the
   full byte breakdown (weights / grads / Adam m / Adam v / fp32 masters) under
   fp32 vs mixed precision, and how many 80 GB GPUs the state alone needs.
2. **Gradient accumulation, proved**: trains a tiny linear model both ways —
   one batch of 4 vs 4 accumulated microbatches — and shows the gradients match
   to ~10⁻¹⁶, plus a bonus demo of fp16-style update rounding (why master
   weights exist) and gradient clipping.

Run it:
```
npx ts-node index.ts
```

## What's next

You can now afford to train big models. But how do you know if the thing you
trained is any *good*? Evaluation is harder than it sounds — arguably the
hardest problem in the field.
[Lesson 10 → Evaluation & Benchmarks](../10-evaluation-and-benchmarks/lesson.md)

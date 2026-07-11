# Lesson 22 — Interpretability

---

## The problem: we built it, but we don't understand it

You have trained a GPT from scratch ([lessons 14](../../lessons/14-tiny-gpt/lesson.md) and [20](../../lessons/20-gpt2-architecture/lesson.md)). You wrote every line. You know
exactly what every matrix multiply does.

And yet: if your model outputs `"Paris"` after `"The capital of France is"`,
**you cannot point at the weights and say where "Paris" came from.**

This is not because you were sloppy. It is because:

- You wrote the *learning algorithm*, but gradient descent wrote the *program*.
- That program is stored as hundreds of millions of floating-point numbers.
- Nobody — not you, not OpenAI, not Anthropic — can read that program directly.

```
Traditional software:              Neural network:
  human writes code                  human writes training loop
  code is readable                   gradient descent writes "code" (the weights)
  bug? → read the code               bug? → stare at 175,000,000,000 floats
```

**Interpretability** is the research field trying to reverse-engineer what
gradient descent wrote. As of 2025/2026 it is one of the most active frontiers —
partly out of scientific curiosity, and increasingly for safety (more on that at the end).

---

## Analogy: neuroscience for artificial brains

Interpretability researchers describe their job as *doing neuroscience on a brain
we can fully measure*. Compared to real neuroscience we have superpowers:

- We can record **every neuron at once**, on every input, with no noise.
- We can **edit** any connection and rerun the exact same input.
- We can make **perfect copies** of the subject.

And yet the models still resist understanding. That tells you the difficulty is
not measurement — it is that the computation is genuinely alien: distributed,
redundant, and not organized the way a human engineer would organize it.

---

## The ladder of interpretability

There are levels, from cheap-and-shallow to expensive-and-deep:

```
Level 1: BEHAVIORAL        "what does it do?"
         Feed inputs, watch outputs. Prompt it, probe it, benchmark it.
         (You did this in ../10-evaluation-and-benchmarks/lesson.md)

Level 2: ATTENTION ANALYSIS "where does it look?"
         Print attention matrices. Which tokens attend to which?
         Cheap, visual, sometimes misleading.

Level 3: PROBING            "what does it know internally?"
         Train a tiny classifier on hidden activations:
         "can I read off part-of-speech from layer 5?" If yes, the
         information is present there (though maybe not used).

Level 4: CIRCUITS           "what algorithm does it run?"
         Identify a small subgraph of heads + neurons that implements
         one specific behavior, and verify by editing it.

Level 5: FEATURES           "what is its vocabulary of concepts?"
         Decompose activations into interpretable units
         (this is where sparse autoencoders come in).
```

Each level answers a different question. A researcher moving down this ladder is
moving from *correlation* toward *mechanism*.

---

## The obstacle: superposition (more features than neurons)

Here is the single most important idea in modern interpretability.

Naive hope: "each neuron = one concept." Neuron 3,041 fires for cats, neuron
3,042 fires for the Eiffel Tower. Early vision-model work found *some* neurons
like this. But in language models, most neurons are **polysemantic** — one
neuron fires for, say, *Korean text, HTTP requests, and the concept of doubt*.

Why? Count things:

- A model layer might have `d = 4,096` neurons.
- The world has *far* more than 4,096 concepts worth representing.

The model's solution: **store more features than it has dimensions**, by
assigning each feature a direction that is *almost* — but not exactly —
orthogonal to the others. This is called **superposition**.

Tiny concrete example (which you will run in `index.ts`): store **5 features in
3 dimensions**.

```
5 feature directions in 3-D space (the best arrangement possible —
5 of the 6 axes of an icosahedron):

  f0 → [ 0.00,  0.53,  0.85]
  f1 → [ 0.00, -0.53,  0.85]      5 arrows crammed into 3-D.
  f2 → [ 0.53,  0.85,  0.00]      They CANNOT all be perpendicular
  f3 → [-0.53,  0.85,  0.00]      (3-D fits only 3 perpendicular arrows).
  f4 → [ 0.85,  0.00,  0.53]      Every pair here overlaps by exactly ±0.447.
```

If features were dense (all active at once), the overlaps would smear everything
together — unrecoverable. But real-world features are **sparse**: in any given
sentence, only a handful of the model's concepts are active. With, say, 1 or 2
features active out of 5, the interference from overlaps is small noise, and
each active feature can still be read out almost perfectly.

**Superposition = lossy compression that works because concepts are sparse.**

The formula for the interference between features i and j is just their dot
product:

```
interference(i, j) = wᵢ · wⱼ

  wᵢ, wⱼ : the direction vectors (unit length) assigned to features i and j
  wᵢ · wⱼ = 0     → perfectly orthogonal, zero crosstalk
  wᵢ · wⱼ = ±0.2  → small crosstalk, fine if features rarely co-occur
  wᵢ · wⱼ = ±1    → same direction, features are indistinguishable
```

The consequence for interpretability: **the neuron is the wrong unit of
analysis.** The meaningful units are *directions in activation space*, and there
are more of them than neurons. So how do we find them?

---

## Sparse autoencoders: a microscope for features

The current best answer (the line of work behind Anthropic's
*Towards Monosemanticity*, 2023, and *Scaling Monosemanticity*, 2024) is the
**sparse autoencoder (SAE)**.

Idea: if the model crammed many sparse features into few dimensions, then
*decompress* them — learn a mapping into a much wider space where each
dimension is used sparsely:

```
model activation  x            (d dims, e.g. 4,096 — dense, superposed)
        │
        ▼  encoder: f = ReLU(W_enc · x + b)
feature vector    f            (m dims, e.g. 16,000,000 — SPARSE: mostly zeros)
        │
        ▼  decoder: x̂ = W_dec · f
reconstruction    x̂           (should match x)

Training loss = ‖x − x̂‖²  +  λ · ‖f‖₁
                 └─ reconstruct   └─ but stay sparse

  x     : an activation vector captured from the running model
  f     : the discovered feature activations (the "decompressed" view)
  λ     : how hard we push toward sparsity
  ‖f‖₁  : sum of absolute feature activations (penalizes having many active)
```

The two loss terms fight each other, and the compromise is exactly what we
wanted: a dictionary of directions such that **any single activation is
explained by a few of them**. When Anthropic ran this on Claude, individual SAE
features turned out to be strikingly interpretable: a Golden Gate Bridge
feature, a code-with-bugs feature, a sycophantic-praise feature. Turning a
feature up or down changes behavior in the corresponding way — famously,
clamping the Golden Gate Bridge feature high made the model identify *as* the
bridge.

Honest caveats (this is a frontier, not a solved problem):

- SAEs don't reconstruct perfectly — some behavior lives in the residual error.
- Feature dictionaries are not unique; train twice, get overlapping-but-different sets.
- "This feature looks like it means X" is a hypothesis, and verifying causally at scale is expensive.
- Whether SAE features are the model's *real* units, or just a useful basis, is contested.

---

## Circuits: the classic discovery — induction heads

A **circuit** is a small set of components (attention heads, neurons) that
together implement an identifiable algorithm. The most famous discovered circuit
(Anthropic, 2021–2022) is the **induction head**, and you will build one in
`index.ts`.

The behavior: models complete repeated patterns. Given

```
... A B ... A → ?
```

the model predicts `B`. Whatever followed `A` last time is a great guess for
what follows `A` this time. This works for *arbitrary* tokens — it is copying,
not memorization.

The algorithm, as two cooperating attention heads in two layers:

```
sequence:      D  A  B  C  A  →?
position:      0  1  2  3  4

Head 1 (layer 1), "previous-token head":
   each position copies the identity of the token BEFORE it
   → position 2 now also carries the info "my predecessor was A"

Head 2 (layer 2), "induction head":
   query at position 4:  "I am A. Who has 'my predecessor was A' in their key?"
   → matches position 2 (token B, whose predecessor was A)
   → attends to B, copies B's identity into the prediction

        D    A    B    C    A
        ·    ·    ▲    ·    Q      ← attention from the second A
                  └─ "look back at the token AFTER
                      the previous occurrence of me"
```

Why this discovery mattered:

- It was verified **causally**: ablate these heads and in-context pattern completion collapses.
- Induction heads appear suddenly during training, and their appearance coincides with a visible bump in the loss curve — a tiny "phase transition."
- They are a leading mechanistic hypothesis for (part of) [in-context learning](../../lessons/24-in-context-learning/lesson.md).

---

## Activation patching: interpretability's causal tool

How do you *prove* component X causes behavior Y, instead of just correlating?
The core technique is **activation patching** (also called causal tracing):

```
Run A (clean):      "The Eiffel Tower is in" → model says "Paris"
Run B (corrupted):  "The Colosseum is in"    → model says "Rome"

Patch: rerun the CORRUPTED prompt, but at one chosen layer & position,
overwrite the activation with the one saved from the CLEAN run.

   If the output flips Rome → Paris:
   that layer/position carried the "which landmark" information.
```

Sweep the patch over every layer × position and you get a causal map of where
information lives and when it moves. This is the workhorse behind most circuit
papers — and note what it requires: **a forward pass and a saved vector.
No training, no GPU cluster.** Activation patching on small open models is one
of the most accessible entry points into real research (remember this in
[lesson 26](../26-open-problems/lesson.md)).

---

## Why interpretability matters for safety

Three practical reasons this field gets serious investment:

1. **Evaluation gap.** Behavioral testing ([lesson 10](../10-evaluation-and-benchmarks/lesson.md)) only samples behavior; it can never prove absence of a behavior. "The model never deceives" cannot be established by prompting alone. Reading mechanisms might one day certify it.
2. **Detecting misalignment before it acts.** If a model *represents* "I am being evaluated right now" or "stated goal ≠ actual goal," those are directions in activation space we could hope to find — SAE work has already surfaced deception- and sycophancy-related features.
3. **Debugging with a mechanism, not a patch.** When a model misbehaves, today's fix is more finetuning — which may hide rather than remove the behavior. Mechanism-level understanding could allow targeted removal.

Status report, honestly stated: interpretability has produced real, verified
explanations of *small* circuits and compelling *feature atlases* of large
models — but nobody can yet take a frontier model and answer "will it do X in
situation Y" from the weights. Whether the field gets there before capabilities
outrun it is one of the live races in AI. That tension is the subject of the
[next lesson](../23-alignment-and-safety/lesson.md).

---

## Code for this lesson

See [index.ts](index.ts) — two runnable experiments:

1. **A hand-built induction head.** We construct the two-head circuit with
   explicit weights (no training, so you can see exactly why it works), run it
   on `D A B C A`, and print the attention matrix — the second `A` attends to
   `B`, the token after the previous `A`.
2. **Superposition in 3 dimensions.** We cram 5 features into 3 dims, then show
   recovery is near-perfect when features are sparse and falls apart when they
   are dense — the whole superposition story in 40 lines.

```
npx ts-node AI-researcher-lessons/22-interpretability/index.ts
```

## What's next

Interpretability is one pillar of making models trustworthy. The broader
question — what "trustworthy" even means, and why it is hard — is alignment:
[Lesson 23 → Alignment & Safety](../23-alignment-and-safety/lesson.md)

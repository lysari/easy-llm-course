# Lesson 26 — Open Problems in AI

---

## The problem: where does a new researcher actually aim?

You have now toured the frontiers: interpretability ([22](../22-interpretability/lesson.md)), alignment ([23](../23-alignment-and-safety/lesson.md)),
efficiency ([24](../24-efficiency-research/lesson.md)), scaling ([25](../25-scaling-and-emergence/lesson.md)). Each lesson ended with "…and this part is
open." This lesson collects the open parts into one map — the problems that,
as of 2025/2026, are genuinely unsolved — and for each one answers the three
questions a researcher must ask before picking a fight:

1. **Why is it hard?** (If you can't say, you don't understand the problem yet.)
2. **What are the current best attempts?** (Never attack an empty field — attack a stuck one.)
3. **Where could a new researcher plausibly contribute at small scale?**

First, the myth that keeps beginners out:

> **Myth: frontier research requires a GPU cluster.**
> Reality: training frontier *models* requires a cluster. Frontier *research*
> mostly doesn't. Evals, interpretability, analysis, and "why does X happen"
> papers are routinely done on laptops and free-tier GPUs, on small open
> models — and those are exactly the papers the field is short of. Everything
> in this lesson tags a laptop-scale entry point. You proved the point
> yourself: every result in lessons 22–25 ran in seconds on your machine.

---

## The map

```
                         RELIABILITY                     LEARNING
                 ┌─ reasoning & planning          ┌─ continual learning
                 ├─ hallucination                 ├─ sample efficiency
                 └─ long context (using it)       └─ data / synthetic data

                         SYSTEMS                         SCIENCE OF THE FIELD
                 ┌─ agents & error compounding    ┌─ evaluation crisis
                 └─ energy cost                   └─ (+ interp & alignment,
                                                       lessons 22–23)
```

Eight problems. For each: hard-why, best-attempts, your-angle.

---

## 1. Reliable reasoning & planning

**The problem.** Models solve olympiad problems, then fail 7×8=54-style slips
inside long chains; success rates that look fine per-step collapse over many
steps. They imitate reasoning *text* superbly — whether that yields robust
multi-step reasoning *process* is the open fight ([lesson 25](../25-scaling-and-emergence/lesson.md)'s contested
territory).

**Why hard.** Next-token training rewards locally plausible continuations, not
globally valid derivations; errors mid-chain aren't penalized where they occur.
No backtracking in the architecture — a wrong token is committed forever.
And we can't yet distinguish "memorized template + shallow adaptation" from
"general procedure" behaviorally (an interpretability question, again).

**Best attempts.** Chain-of-thought; test-time compute / RL-on-verifiable
problems (o1/R1 line — big real gains on checkable domains); tool use
(offload arithmetic to code); process reward models (grade steps, not answers).
None gives reliability *guarantees*; transfer beyond checkable domains unclear.

**Small-scale angle.** Error anatomy is wide open: take one open model, one
task family, and characterize *where* chains break (step k? which operation?
does self-consistency fix a specific error type?). Careful failure taxonomies
on 200 problems are publishable and need one laptop plus API calls.

---

## 2. Hallucination

**The problem.** Models state falsehoods with the same fluent confidence as
truths — fabricated citations, court cases, APIs. The single biggest blocker
to deploying LLMs where mistakes cost money or lives.

**Why hard.** The objective *demands* it: pretraining rewards plausible
continuation, and a plausible-sounding citation scores better than "I don't
know" (which is also underrepresented in text — and post-training that
rewards confident answers makes it worse: a 2025 OpenAI analysis argued
standard binary-graded evals literally teach models to guess). Deeper: the
model has no built-in fact/confabulation distinction — generation and
"memory" are the same operation, next-token sampling.

**Best attempts.** RAG (ground answers in retrieved text — helps, doesn't
close it); uncertainty and calibration training ("I don't know" rewards);
probing internal states for "does the model know it doesn't know" (interp
results suggest truthfulness directions exist); citation-forcing; abstention
benchmarks. Progress is real; elimination, nobody claims.

**Small-scale angle.** Among the most laptop-friendly problems in AI:
calibration studies (does stated confidence match accuracy? by domain?),
abstention evals, "known-unknown" datasets, probing a 7B model's activations
on true vs false statements. All free-tier feasible.

---

## 3. Long context that actually uses the context

**The problem.** Context windows hit 1M+ tokens, and marketing says "reads
whole codebases." Measured behavior: attention to the middle of long contexts
degrades ("lost in the middle"), multi-hop use of scattered facts decays far
below needle-in-haystack scores, and effective context ≪ advertised context.

**Why hard.** Attention is a *soft* average over T positions — as T grows,
relevant signal competes against thousands of distractors for probability
mass ([lesson 12](../../lessons/12-attention/lesson.md): softmax must spread). Positional schemes (RoPE, [lesson 25 of
the LLM track](../../lessons/25-rope-positional-encoding/lesson.md)) extrapolate imperfectly. And training data with genuine
100K-token dependencies is scarce — most long documents don't actually
require long-range use.

**Best attempts.** Better position scaling (YaRN etc.); architectural hybrids
(state-space models like Mamba, linear attention — [lesson 08](../08-architecture-survey/lesson.md)'s trade-offs
revisited); retrieval inside context; synthetic long-dependency training data.
Benchmarks (RULER, multi-hop suites) show every model's effective length well
under its advertised one.

**Small-scale angle.** Evaluation again: needle-in-haystack is saturated and
too easy — design tasks where the answer *changes* if the model missed
mid-context info (contradictions, counting, updates). Positional-bias curves
(accuracy vs where the fact sits) on open models: a weekend of work, genuinely
useful.

---

## 4. Continual learning / catastrophic forgetting

**The problem.** A deployed model is frozen at its cutoff. Finetune it on new
knowledge and it *catastrophically forgets* old capabilities — you met this in
instruction finetuning ([lesson 26 of the LLM track](../../lessons/26-instruction-finetuning/lesson.md)). Humans learn Tuesday's
facts without losing Monday's.

**Why hard.** Superposition strikes again ([lesson 22](../22-interpretability/lesson.md)): knowledge is smeared
across shared weights, so gradients for the new task plow through directions
the old tasks were using. There is no "append" operation on a neural network —
only global overwrite pressure.

**Best attempts.** Replay (mix old data in — works, needs the old data);
regularization toward important old weights (EWC — weak at scale); LoRA
adapters per domain ([lesson 24](../24-efficiency-research/lesson.md) — sidesteps rather than solves);
model editing (ROME/MEMIT: surgically rewrite one fact — brittle, edits
interfere); RAG as "keep knowledge outside the weights entirely." The blunt
truth: the industry's actual solution is *retrain from scratch periodically*,
which is a $100M workaround, not a solution.

**Small-scale angle.** Forgetting is fully visible at tiny scale: finetune
your own lesson-14 GPT (or a 1B open model with LoRA) on task B, measure decay
on task A as a function of steps/LR/data mixing. Clean forgetting-curve
studies with error bars ([lesson 20](../20-statistical-significance/lesson.md)) are exactly the kind of careful small
science the area lacks. Model-editing stress tests (edit 100 facts, what
breaks?) likewise.

---

## 5. Sample efficiency

**The problem.** ~10¹³ tokens for a frontier model vs ~10⁸ words for a
10-year-old human: a five-orders-of-magnitude gap in data efficiency, and
scaling laws ([lesson 25](../25-scaling-and-emergence/lesson.md)) describe the inefficiency rather than fix it.

**Why hard.** Nobody knows what the missing ingredient is — candidates include
active/embodied learning (humans choose what to attend to), strong priors from
evolution, richer feedback than next-token error, compositional abstraction.
"Which of these matters" is a genuinely open scientific question, tangled with
the data wall: if models needed 100× less data, the wall vanishes.

**Best attempts.** Data curation/curricula (quality beats quantity — small
gains, real); synthetic data for targeted skills; architectures with explicit
memory; meta-learning ("learning to learn" — [lesson 24 of the LLM track](../../lessons/24-in-context-learning/lesson.md):
in-context learning IS few-shot learning, but it doesn't persist). The
BabyLM challenge — best model trained on a human-childhood-sized 100M words —
exists precisely to focus this question, and it's laptop-scale by design.

**Small-scale angle.** BabyLM is the entry point: it's a standing competition
*defined* at small scale. Or: at fixed tiny compute, curriculum A vs B vs
random — does ordering matter? ([lesson 18](../18-baselines-and-ablations/lesson.md) discipline applies.)

---

## 6. Agents: multi-step autonomy and error compounding

**The problem.** The 2025-era bet: LLMs that act — browse, code, book, operate
computers — over long horizons. The math is brutal: per-step success p over n
steps ⇒ ~pⁿ end-to-end. **0.99⁵⁰ ≈ 0.61. 0.99²⁰⁰ ≈ 0.13.** Benchmarks show
strong models completing minutes-long tasks reliably and hours-long tasks
rarely; the "task length agents can finish" metric is doubling every ~7 months
(METR) but from a low base.

**Why hard.** Compounding is geometric, so *reliability* — not capability —
is the binding constraint (this is [lesson 23](../23-alignment-and-safety/lesson.md)'s capable≠aligned distinction
wearing overalls: a 95%-per-step agent is impressive and useless). Recovery
requires noticing your own errors (self-verification is weak — hallucination
again); environments are partially observable and change under you; and
evaluation of long open-ended tasks is itself unsolved (see #8).

**Best attempts.** Explicit plan-execute-verify loops; checkpointing and
rollback; tool-verified steps (compiler as ground truth); multi-agent
cross-checking; RL on full task trajectories. Working pattern in practice:
constrain the domain hard (coding agents work best precisely because tests
provide per-step ground truth).

**Small-scale angle.** Measure the compounding curve itself: define a
parameterized n-step toy task (file transformations, chained lookups), plot
success vs n across models, fit the per-step rate, test whether verification
loops bend the curve. Agent evals are new enough that careful methodology
papers come from nowhere and get cited.

---

## 7. Data: synthetic data and the wall

**The problem.** [Lesson 25](../25-scaling-and-emergence/lesson.md): compute-optimal data grows with budget and the
high-quality human text supply is roughly *spent* at frontier scale. Can
models train on model-made data without degenerating?

**Why hard.** The information isn't free: a model sampling from itself adds no
new bits, and recursive training on unfiltered own-output provably narrows
distributions ("model collapse" — tails vanish first: rare facts, rare
styles). Synthetic data works when an external *verifier* injects signal
(compiler, proof checker, simulator) — which is why math/code lead — but most
of language has no checker (back to scalable oversight, [lesson 23](../23-alignment-and-safety/lesson.md)).

**Best attempts.** Verified-synthetic pipelines (generate → check → keep) power
the reasoning-model wave; curated phi-style "textbook" data shows quality can
substitute for quantity at small scale; multimodal and licensed private
corpora buy time; deduplication/filtering science quietly improves every
frontier run.

**Small-scale angle.** Collapse dynamics at toy scale: train your tiny GPT on
its own output for k generations, measure diversity/tail loss per generation;
vary the filtered fraction; find the phase boundary. This is cheap, visual,
and the dynamics genuinely aren't mapped.

---

## 8. The evaluation crisis

**The problem.** The instrument the whole field steers by is breaking:
benchmarks saturate within months, test sets leak into training corpora
(contamination), lab-reported numbers diverge from user experience,
LLM-as-judge inherits the judge's biases, and models increasingly *recognize*
they're being tested ([lesson 23](../23-alignment-and-safety/lesson.md)). Goodhart, everywhere, all at once.

**Why hard.** It's adversarial and reflexive: any public, fixed target gets
optimized against, directly (train on it) or culturally (build for it). And
"capability" isn't one number — compressing a generative system into a
leaderboard scalar destroys most of the information ([lesson 25](../25-scaling-and-emergence/lesson.md): metric
choice can manufacture or hide phenomena).

**Best attempts.** Private held-out sets; contamination detection; living/
refreshing benchmarks; head-to-head human preference arenas (with their own
gaming problems); statistical rigor pushes (error bars on evals — your
[lesson 20](../20-statistical-significance/lesson.md)); dangerous-capability eval standards ([lesson 23](../23-alignment-and-safety/lesson.md)).

**Small-scale angle.** The most beginner-accessible problem in AI — an eval
is a dataset, a metric, and honest statistics. Audit an existing benchmark
for contamination or label errors (famous papers have come from finding 3%
label noise); build a 200-item eval for something nobody measures; quantify
judge-model bias (position bias, length bias, self-preference — reproducible
in an afternoon with API calls).

---

## 9. Energy cost

**The problem.** Frontier training ~10²⁵–10²⁶ FLOPs ([lesson 25](../25-scaling-and-emergence/lesson.md)); datacenter
buildouts measured in gigawatts; inference energy now dominating training
energy at deployment scale. Meanwhile a human brain runs on ~20 watts —
a standing existence proof of ~10⁴–10⁶× headroom.

**Why hard.** The von Neumann tax: most energy moves bits, not flips them —
weights shuttle between memory and compute for every token ([lesson 24](../24-efficiency-research/lesson.md)'s
memory-bandwidth wall is also an energy wall). Fixes below the algorithm level
(sparsity that hardware can exploit, analog/neuromorphic compute) fight
decades of GPU-shaped infrastructure lock-in.

**Best attempts.** Everything in [lesson 24](../24-efficiency-research/lesson.md) (quantization, distillation, MoE,
speculative decoding — production reality precisely because of this problem);
hardware-algorithm co-design (FlashAttention thinking); routing easy queries
to small models. The 10⁴× gap to the brain: untouched.

**Small-scale angle.** Measurement and accounting: joules-per-token curves
across model sizes and quantization levels on your own hardware (`nvidia-smi`
or RAPL, one GPU or even CPU); cost-quality Pareto fronts for cascades
(small-model-first routing). Honest public numbers here are rare and wanted.

---

## Reading the map: what the problems have in common

Squint and the eight problems share three roots — worth noticing, because
root-level ideas transfer across problems:

1. **No self-knowledge.** Hallucination, agent error-compounding, and eval-
   gaming all stem from models lacking a reliable read on their own state.
   (Interpretability is upstream of more than safety.)
2. **Goodhart.** Reward hacking, benchmark saturation, synthetic-data
   collapse: optimize a proxy hard enough and it stops proxying ([lesson 23](../23-alignment-and-safety/lesson.md)).
3. **The append problem.** Forgetting, sample efficiency, and the data wall
   are all versions of "gradient descent can only overwrite, never accrete."

And notice what dominates the "small-scale angle" rows: **measurement**.
The field's bottleneck is less "nobody has a big enough GPU" and more "nobody
has characterized this carefully." That is a beginner's opening, and it is
also just... science.

---

## Exercise for this lesson

See [exercise.md](exercise.md) — you'll pick two problems from this map, find
one recent paper attacking each, extract the problem–attempt–gap triple
([lesson 12](../12-how-to-read-a-paper/lesson.md) skills), then brainstorm five laptop-scale research questions of
your own and commit to the most feasible one. Keep that final page — it is
the raw material for the next lesson.

## What's next

You have a map of open problems and one candidate question in your pocket.
Phase F begins: turning a vague itch into a well-posed, answerable research
question — [Lesson 27 → Finding a Research Question](../27-finding-a-research-question/lesson.md)

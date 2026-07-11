# Lesson 00 — What AI Research Actually Is

---

## The problem this lesson solves

You've probably had this thought: *"I can follow tutorials. I can build things. But the people who invented the transformer, or discovered scaling laws — how did they DO that? What do they know that I don't?"*

The honest answer: less than you think. Research is not magic performed by geniuses. It is a **learnable process** — a loop of reading, guessing, testing, and writing — plus a body of shared knowledge (mostly math and experimental habits) that this curriculum will give you.

This lesson demystifies what research actually is, what researchers actually do all day, and the realistic path from where you are now to contributing something new.

---

## Research vs engineering: the core distinction

Both are valuable. They answer different questions:

```
Engineering asks:  "How do I build X so it works reliably?"
Research asks:     "Is X even true? What happens if...? Why does this work?"
```

Concrete example, same topic, two mindsets:

```
Topic: attention in transformers

Engineer:   "How do I implement attention efficiently so my
             model trains in 4 hours instead of 12?"

Researcher: "Do we even NEED attention? What if we replaced it
             with a simpler mixing operation — would the model
             still learn language? Why or why not?"
```

The engineer optimizes a known solution. The researcher questions whether the solution is the right one at all — and, crucially, designs an **experiment** whose outcome could prove them wrong.

That last part is the heart of it:

> **Research = asking a question nobody has answered yet, then designing a test that could embarrass you.**

If no possible experimental result could change your mind, you're not doing research — you're doing marketing.

---

## An everyday analogy: the detective

A detective doesn't know who committed the crime. They:

1. **Read the case file** (= read papers: what's already known?)
2. **Form a hypothesis** ("the butler did it" = "attention heads track syntax")
3. **Gather evidence that could disprove it** (alibi check = ablation experiment)
4. **Update or abandon the hypothesis** when evidence contradicts it
5. **Write the report** so the next detective doesn't start from zero

Most hypotheses are wrong. A good detective — and a good researcher — is defined not by being right the first time, but by **killing their own bad ideas quickly and cheaply**.

---

## What researchers do all day (really)

The romantic image: staring at a whiteboard until a breakthrough strikes.
The reality, roughly, for an empirical ML researcher:

```
~30%  READING      papers, blog posts, other people's code, results
~10%  HYPOTHESIZING talking, arguing, whiteboarding, journaling ideas
~40%  EXPERIMENTING writing training code, launching runs, staring
                    at loss curves, debugging why loss is NaN
~20%  WRITING      papers, internal docs, plots, talks
```

Notice: the single biggest chunk is **experiments**, and most of *that* is debugging. A shocking amount of research skill is just: "my loss curve looks wrong — is it a bug, or a discovery?" (It's a bug 95% of the time. The other 5% is where papers come from.)

The four activities feed each other in a loop:

```
        ┌──────────┐
   ┌───▶│   READ   │  what do we already know?
   │    └────┬─────┘
   │         ▼
   │    ┌──────────┐
   │    │HYPOTHESIZE│  what might also be true?
   │    └────┬─────┘
   │         ▼
   │    ┌──────────┐
   │    │EXPERIMENT│  is it actually true?
   │    └────┬─────┘
   │         ▼
   │    ┌──────────┐
   └────│  WRITE   │  tell everyone, get feedback
        └──────────┘
```

---

## The lifecycle of an idea → paper

Let's trace a realistic idea from birth to publication. Suppose you notice, while training a tiny GPT (like the one in the companion track, [lesson 14](../../lessons/14-tiny-gpt/lesson.md)), that removing positional encoding barely hurts your model.

```
Week 1    Curiosity: "wait, why did that barely matter?"
Week 2    Literature search: has anyone studied this?
          (Yes — partially. Causal masks leak position info.
           But nobody tested it at YOUR scale/setup. Gap found.)
Week 3-4  Minimal experiment: train 4 tiny models,
          with/without positions, 2 sizes. Plot loss curves.
Week 5    Result is interesting → design the REAL experiment:
          more sizes, more seeds, controls for confounders
Week 6-10 Run experiments. Half of them fail for dumb reasons
          (out of memory, bad learning rate, off-by-one bug).
Week 11   Analyze. The effect is real but smaller than hoped.
          Decide: is this still worth telling people? (Often yes —
          "the effect is smaller than folklore says" IS a finding.)
Week 12-14 Write the paper. Make plots. Rewrite. Rewrite again.
Week 15   Submit to a conference (NeurIPS, ICML, ICLR...) or
          post to arXiv (a free preprint server — most ML papers
          appear there first, no gatekeeping).
Month 6   Reviews come back: one likes it, one is confused,
          one wants more experiments. Revise, resubmit.
```

Key realism checks:
- Most ideas die at week 3-4. **That's the system working**, not failing.
- The published paper describes a clean straight line. The actual work was a drunken walk. Everyone knows this; nobody says it in the paper.
- A "small" finding, honestly measured, beats a grand claim with shaky evidence.

---

## Types of research

AI research isn't one activity. Four broad flavors:

**1. Empirical** — "run experiments, measure what happens."
*Example question: does model performance improve smoothly as we add parameters?*
This is most of modern ML research, and most of this curriculum.

**2. Theoretical** — "prove things with math, no GPUs required."
*Example question: can a transformer, in principle, represent any sequence-to-sequence function? What can a single attention layer NOT compute?*
Requires the deepest math; produces the most durable results.

**3. Systems** — "make it fast, big, and cheap."
*Example question: how do you train one model across 10,000 GPUs when any single GPU can't even hold the weights?* (FlashAttention, ZeRO, pipeline parallelism live here.)
Closest to engineering, but the questions are open — nobody knows the answers in advance.

**4. Safety / alignment** — "make it do what we actually want."
*Example question: when a model explains its reasoning, is that explanation the real cause of its answer — or a plausible story invented after the fact?*
Combines empirical work, theory, and philosophy. Growing fast.

Most researchers major in one and borrow from the others. This curriculum makes you literate in all four, with an empirical center of gravity — because empirical research is where a self-taught person with a laptop can genuinely contribute first.

---

## Three famous discoveries, told as stories

**AlexNet (2012) — the bet nobody wanted to fund.**
For decades, neural networks were unfashionable. "Real" computer vision meant hand-crafted features. Two grad students, Alex Krizhevsky and Ilya Sutskever, with their advisor Geoffrey Hinton, bet that neural nets weren't wrong — they'd just been **too small and too slow**. Alex trained a big convolutional network on two consumer gaming GPUs in his bedroom. At the 2012 ImageNet competition, their error rate was 15.3%. Second place: 26.2%. In a field where progress meant shaving off half a percent, they had nearly **halved** the error. Within three years, essentially everyone in computer vision had switched to neural networks. Lesson: sometimes the research contribution isn't a new idea — it's the *conviction and craft to scale an old one until it works*.

**"Attention Is All You Need" (2017) — deleting the main ingredient.**
State-of-the-art translation models were recurrent networks (they read text token by token, in order) with attention bolted on as a helper. A team at Google asked a deliciously reckless question: *what if we delete the recurrence and keep only the helper?* The resulting architecture — the transformer — trained far faster (all tokens processed in parallel) and translated better. The paper's title was a joke that became a prophecy: attention really was all you needed. GPT, BERT, Claude — all transformers. Lesson: a classic research move is **ablation taken seriously** — remove a component everyone assumes is essential and see what actually breaks. You'll build this exact architecture in the companion track ([lesson 12](../../lessons/12-attention/lesson.md) onward).

**Scaling laws (2020) — finding physics inside a curve fit.**
Everyone knew "bigger models do better." Kaplan, McCandlish, and colleagues at OpenAI asked a more precise question: *better by exactly how much?* They trained many models of many sizes and found that loss falls as a startlingly clean **power law** across seven orders of magnitude of compute — so clean you can extrapolate: spend 10× the compute, predict the loss *before training*. This transformed LLM development from craft into something like engineering with a slide rule, and directly justified the bets behind GPT-3 and everything after. In 2022, DeepMind's "Chinchilla" paper re-ran the analysis more carefully and found everyone had been making models too big and training them on too little data — a correction worth billions of dollars. Lesson: **measuring something carefully that everyone "already knows"** can be a landmark contribution. And published results can be importantly wrong — checking them is research too. (Companion track: [lesson 22](../../lessons/22-scaling-laws/lesson.md).)

---

## The realistic path from beginner to researcher

Nobody wakes up and writes "Attention Is All You Need." The actual ladder:

```
Stage 1: REPRODUCE     Reimplement known results from scratch.
                       (The companion from-scratch track IS this stage.)
Stage 2: UNDERSTAND    Read papers and truly follow them —
                       math, plots, and the fine print in appendices.
Stage 3: TWEAK         Take a known result, change one variable,
                       measure honestly. "What if the learning rate
                       warmup were removed?" is a real experiment.
Stage 4: QUESTION      Notice something that doesn't add up in the
                       literature. Design your own experiment for it.
Stage 5: CONTRIBUTE    Share a finding others build on — a blog post
                       with careful experiments counts. Many influential
                       ML results debuted as blog posts, not papers.
```

Honest notes:
- No university math is required to start, but the math **is** required eventually — that's Phase A below, and we build it from zero.
- A PhD is one path, not the only one. Open-source ML has a strong tradition of self-taught contributors; small-scale, careful empirical work is publishable and gets noticed.
- Your realistic superpower as a solo learner: **tiny models, run honestly**. Most sloppy claims in ML die when someone bothers to run the clean small-scale version. That someone can be you.

---

## How the two tracks fit together

You have two parallel curricula, and they interlock:

```
../../lessons/           THE FROM-SCRATCH TRACK ("build it")
  00-what-is-a-model ... 30-gpt4-capstone
  You implement models: regression → backprop → attention →
  GPT-2 → RLHF. Output: working code and mechanical intuition.

AI-researcher-lessons/   THIS TRACK ("understand and question it")
  Math foundations → the science of deep learning → reading
  papers → running experiments → frontiers → doing research.
  Output: the ability to ask and answer NEW questions.
```

The from-scratch track makes you the engineer who can build the apparatus. This track makes you the scientist who knows what experiment to run on it. Cross-references appear throughout — when this track says "recall backprop," it links to [../../lessons/07-backpropagation/lesson.md](../../lessons/07-backpropagation/lesson.md) where you built it with your own hands.

---

## The map: six phases of this track

```
PHASE A — MATH FOUNDATIONS (lessons 01-05)      ◀── you are here
   Linear algebra · calculus & gradients · probability &
   statistics · information theory · numerical computing.
   The vocabulary every paper assumes you speak.

PHASE B — THE SCIENCE OF DEEP LEARNING
   Optimization landscapes, generalization, why nets train at
   all, regularization, the tricks that make training stable —
   treated as scientific questions, not recipes.

PHASE C — READING & UNDERSTANDING PAPERS
   How to read a paper in 3 passes, decode the math, judge the
   evidence, and spot the weaknesses reviewers spot.

PHASE D — RUNNING EXPERIMENTS
   Baselines, ablations, controls, seeds, statistical
   significance, honest plots — the craft of not fooling yourself.

PHASE E — FRONTIERS
   Scaling, alignment, interpretability, reasoning, agents:
   what's known, what's contested, what's wide open.

PHASE F — BECOMING A RESEARCHER
   Choosing problems, writing, sharing work, joining the
   community — and your first original project.
```

Phase A is five lessons of math, and here's the promise: **every single concept is taught because a specific paper or technique needs it**, is explained with tiny concrete numbers, and is implemented in runnable TypeScript. No symbol soup. No "it can be shown that." If we can't make you compute it by hand and in code, we don't teach it.

---

## Code for this lesson

None — this lesson is orientation. Code starts in Lesson 01, where you'll implement the linear algebra that every transformer runs on.

## What's next
[Lesson 01 → Linear Algebra for Research](../01-linear-algebra/lesson.md)

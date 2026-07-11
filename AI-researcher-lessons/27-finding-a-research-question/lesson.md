# Lesson 27 — Finding a Research Question

---

## The problem

You have finished the hard part. You can build a GPT from scratch, read papers, reproduce
them, run controlled experiments with proper statistics. And now you face the blank page:

> "What should I actually research?"

This is the question that paralyzes more would-be researchers than any math ever did.
It feels like great questions come from nowhere — a flash of genius that happens to
other people. That is a myth. Research questions come from a small number of very
learnable places, and choosing between them is a skill called **research taste**.

This lesson teaches you where questions come from, how to judge them, and how to
shrink a question until it fits in your laptop.

---

## Intuition: prospecting, not inventing

A gold prospector does not invent gold. They learn *where gold tends to be* — riverbends,
quartz veins, tailings other miners abandoned — and they check those places systematically.

Finding a research question is prospecting. There are known riverbends:

```
WHERE QUESTIONS LIVE
─────────────────────────────────────────────────────────────
1. Limitation sections    "We leave X to future work" = a map
2. "What if X but Y?"     recombine two known ideas
3. Failed reproductions   your lesson-15 discrepancies
4. Tool transfer          a trick from another field, applied here
5. Your own annoyances    "why does my model always do THIS?"
─────────────────────────────────────────────────────────────
```

Let's walk each vein.

---

## Source 1: the limitations section is a treasure map

Every honest paper ends with what it *didn't* do. Researchers write these sections as
confessions; you should read them as invitations.

Real examples of the pattern:

- The original Transformer paper (2017) noted attention cost grows quadratically with
  sequence length → spawned an entire subfield (Linformer, Performer, FlashAttention,
  state-space models).
- The Chinchilla paper (2022) fixed its analysis to one architecture family and one
  data distribution → follow-ups asked "do these scaling laws hold for code? for
  multimodal? for MoE?" (You met this in [lesson 25](../25-scaling-and-emergence/lesson.md).)
- LoRA (2021) said "we focus on attention weights only" → people immediately asked
  "what about the MLP weights?" and found it mattered.

**The move:** take 3 papers you liked from Phase C, reread ONLY the limitations /
future-work paragraphs, and write each limitation as a question. You will have 10
candidate questions in an hour.

---

## Source 2: "What if X, but Y?"

Most papers are recombinations:

```
  known thing X          +   known twist Y        =  paper
  ─────────────────────      ──────────────────      ────────────────────
  attention                  linear kernels          Performer
  RLHF                       AI feedback instead     RLAIF / Constitutional AI
  distillation               chain-of-thought data   CoT distillation
  dropout                    entire layers           LayerDrop / stochastic depth
```

The recipe: keep a list of *mechanisms* (dropout, weight tying, curriculum, distillation,
quantization…) and a list of *settings* (tiny models, long context, low data, multilingual…).
Cross them. Most cells are empty or barely explored — especially at small scale, which is
exactly where you can compete.

---

## Source 3: failed reproductions

In [lesson 15](../15-reproducing-a-paper/lesson.md) you reproduced a paper. Did every
number match? Almost certainly not. Each mismatch is a question:

- "The paper claims X helps; in my reimplementation it doesn't. Under what conditions
  does X actually help?"
- Entire good papers are exactly this. "Do Transformer Modifications Transfer?"
  (Narang et al., 2021) tested dozens of published "improvements" in one controlled
  codebase and found most of them evaporate. That's a landmark paper made of
  reproduction failures.

Your discrepancy log from lesson 15 is a question generator you already own.

---

## Source 4: tool transfer from other fields

Take a technique that is standard in field A and ask if it works in field B:

- Signal processing → RoPE's rotations ([lesson 25 of the other track](../../lessons/25-rope-positional-encoding/lesson.md))
- Physics (diffusion processes) → diffusion image models
- Economics (mechanism design) → RLHF reward shaping
- Neuroscience (sparse coding) → sparse autoencoders in interpretability
  ([lesson 22](../22-interpretability/lesson.md))

If you know *anything* deeply besides ML — music, biology, compilers, finance — you have
a private stash of tools nobody else is transferring.

---

## Source 5: your own annoyances

The most underrated source. You have trained dozens of tiny models across both tracks.
Things annoyed you:

- "Why does my char-LM's loss always plateau around 2.1?"
- "Why did seed 3 diverge when seeds 1 and 2 were fine?"
- "Why does my model learn quotes and newlines first?"

Every annoyance, stated precisely, is a research question. Andrej Karpathy's widely-read
"recipe" posts came from cataloguing exactly these irritations.

---

## Research taste: the three circles

You now have 10–20 candidate questions. Which one do you pursue? Draw three circles:

```
            ┌────────────────┐
            │   IMPORTANT     │
            │  (people would  │
            │   care about    │
            │   the answer)   │
       ┌────┼──────┬──────────┼────┐
       │    │  ok  │ ★ HERE ★ │    │
       │    └──────┼──────────┼────┘
       │ TRACTABLE │   ok     │ FEASIBLE FOR *YOU*
       │ (answer-  │          │ (your hardware,
       │  able at  └──────────┤  your skills,
       │  all)                │  your time)
       └───────────────────────────┘
```

- **Important**: if you got a clean answer, would anyone change what they do?
- **Tractable**: can the question be answered by *any* experiment, even in principle?
  ("Is GPT-5 conscious?" fails here.)
- **Feasible for you**: can *you* attack it with a laptop, TypeScript, and evenings?

The star is the intersection. Notes on failure modes:

- Important + tractable, not feasible → "does this hold at 70B scale?" You cannot run it.
  Park it, or find the small-scale proxy.
- Tractable + feasible, not important → "does my loss go down 0.1% with green terminal
  text?" You can do it; nobody cares.
- Important + feasible, not tractable → vague philosophy. Sharpen until an experiment
  could *fail*.

---

## Scoping down: the minimum viable experiment

Beginners scope questions like this:

> "Does curriculum learning improve language models?"

That is a research *program*, not a question — it would take a lab five years. The art
is shrinking without killing the point:

```
"Does curriculum learning improve language models?"
        ↓ pick ONE curriculum
"Does short-to-long sequence ordering improve language models?"
        ↓ pick ONE model scale you can run
"...improve a 4-layer char-level transformer on tiny-shakespeare?"
        ↓ pick ONE metric and budget
"...lower validation loss after 5,000 steps, vs shuffled order,
    3 seeds each, same total tokens?"
```

The last version runs on your laptop in an afternoon and has a yes/no answer.
This is the **minimum viable experiment (MVE)**: the *cheapest* experiment whose result
would genuinely update your belief.

The rule from [lesson 17](../17-scientific-method/lesson.md) applies with full force:
one variable, everything else controlled, and decide *before running* what result would
mean "yes" and what would mean "no".

Small-scale answers do not always transfer to large scale — that is a real limitation,
and you will state it honestly in your writeup. But "X helps at tiny scale" is still
evidence, and "X does nothing even at tiny scale where effects are easy to see" is often
a strong negative signal. Many published results (see below) live entirely at small scale.

---

## The literature check: has someone already done it?

Before investing a week, invest an hour. The goal is NOT to find that your idea is taken —
almost every idea is *adjacent* to existing work. The goal is to find the **three nearest
papers** and state precisely how your question differs.

The search protocol (you have the tools from [lesson 13](../13-navigating-arxiv/lesson.md)):

1. **Name the concepts, then find the field's words for them.** You might call it
   "sharing the input and output matrices"; the field calls it "weight tying" or
   "tied embeddings". Search your naive phrase first, learn the canonical term from
   the first hit, then search the canonical term.
2. **Google Scholar / arXiv search** with 2–3 phrasings. Read only titles + abstracts.
3. **Walk citations both ways** on the best hit: its references (backward) and its
   "cited by" list (forward). Forward citations find the modern follow-ups.
4. **Search the code**: GitHub search for the mechanism's name often finds ablations
   buried in training repos that never became papers.
5. **Stop after ~10 abstracts.** Write down the 3 nearest papers and, for each,
   one sentence: "differs from my question because ___".

Possible outcomes, all fine:

- **Nobody did it** → green light (and slight suspicion: is it not important, or not
  tractable? Re-check the circles).
- **Someone did it at large scale** → you can ask "does it hold at tiny scale?" or vary
  a condition they fixed.
- **Someone did exactly it** → read the paper carefully. Their limitations section is
  your next question. You lost an hour and gained a literature.

---

## The question quality checklist

Run every candidate through this before committing:

```
□ FALSIFIABLE   Is there a possible experimental result that would
                make me say "no, I was wrong"?
□ MEASURABLE    Is the outcome a number (loss, accuracy, steps to
                threshold), not a vibe?
□ CONTROLLED    Can I change ONE thing while holding all else fixed?
□ NOVEL-ISH     Did my literature check find the 3 nearest papers,
                and can I state my difference in one sentence?
□ FEASIBLE      Total compute < a weekend on my machine? (estimate
                runs × steps × cost-per-step BEFORE starting)
□ KILLABLE      Do I have a pre-registered kill criterion — a date
                or result at which I stop and write it up as-is?
```

The last box matters more than it looks. Zombie projects — half-alive experiments you
neither finish nor abandon — are the main way beginner researchers lose years.
[Lesson 21](../21-negative-results/lesson.md) taught you that a clean negative result
is a *finished* project. Kill criteria are how you guarantee you finish.

---

## Great small-compute questions from real papers

Proof that laptop-scale questions become real research:

- **"Do deep nets have to be deep?"** (Ba & Caruana, 2014) — trained *shallow* nets to
  mimic deep ones. Core experiments: small MLPs on TIMIT/CIFAR.
- **Weight tying** (Press & Wolf, 2017, "Using the Output Embedding to Improve Language
  Models") — a 6-page paper, small LSTMs on PTB, one clean idea; used in GPT-2 and
  nearly everything since. This is the paper we will replicate in spirit in
  [lesson 30](../30-capstone-research-project/lesson.md).
- **Grokking** (Power et al., 2022) — one-layer transformers on tiny algorithmic datasets
  (modular arithmetic). Trainable on any laptop; opened a research area.
- **"The Lottery Ticket Hypothesis"** (Frankle & Carbin, 2019) — key early experiments
  on LeNet/MNIST-scale networks.
- **Induction heads** (Olsson et al., 2022) — pivotal interpretability findings on 1–2
  layer attention-only models like the ones you built in
  [lesson 14 of the other track](../../lessons/14-tiny-gpt/lesson.md).

Notice the shape they share: one sharp question, one controlled comparison, a small model
where the effect is visible, an honest writeup. That shape is fully within your reach.

---

## The exercise

See [exercise.md](exercise.md) — take your candidate question from
[lesson 26](../26-open-problems/exercise.md) and run it through the full pipeline:
literature check, falsifiable hypothesis, minimum viable experiment, resource estimate,
kill criteria. This is the question you will carry into lessons 28 and 30.

## What's next
[Lesson 28 → Writing a Paper](../28-writing-a-paper/lesson.md)

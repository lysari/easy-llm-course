# Lesson 30 — Capstone: A Real Mini Research Project

---

## The problem

You have every piece: questions ([27](../27-finding-a-research-question/lesson.md)),
method ([17](../17-scientific-method/lesson.md)–[21](../21-negative-results/lesson.md)),
writing ([28](../28-writing-a-paper/lesson.md)), community
([29](../29-community-and-career/lesson.md)). What you have never done is run **one
complete research cycle** — question → literature → hypothesis → experiment → statistics
→ honest writeup — with no step skipped.

That is the graduation project. Pieces don't make you a researcher; the *loop* does.

This lesson first walks a FULL worked example at toy scale, so you see the whole arc in
one sitting. Then you run the same loop on YOUR question from lessons 26–28.

```
QUESTION → LITERATURE → HYPOTHESIS → DESIGN → RUN → STATS → CONCLUSION → WRITEUP
   └──────────────────────── one loop, no skipped steps ────────────────────────┘
```

---

## The worked example

### Step 1 — The question

> **Does weight tying — sharing the input embedding matrix with the output projection —
> hurt or help a tiny character-level language model?**

Where it came from (lesson 27's sources in action): GPT-2, which you built in
[lesson 20 of the other track](../../lessons/20-gpt2-architecture/lesson.md), ties its
embedding and output matrices. Everyone does it at large scale to save parameters. But
a tiny model is *starved* for parameters — maybe tying costs something there? That's a
"what if X (tying) but Y (tiny scale)" question. Check the three circles: mildly
important (informs anyone training small models), clearly tractable (one bit to flip),
trivially feasible (seconds per run).

### Step 2 — The literature check (1 hour, done)

Three nearest papers:

1. **Press & Wolf 2017, "Using the Output Embedding to Improve Language Models"** —
   showed tying *improves* word-level LSTM LMs on PTB. Differs: word-level, LSTMs,
   millions of params; we ask about char-level, hundreds-of-times smaller.
2. **Inan et al. 2017, "Tying Word Vectors and Word Classifiers"** — theoretical framing
   of tying as a loss-based regularizer. Differs: again word-level; predicts tying should
   help most when data is scarce — our setting tests that prediction's edge.
3. **Radford et al. 2019 (GPT-2)** — uses tying at scale as an engineering default,
   without small-scale ablation. Differs: we do the ablation.

Nobody reports seed-controlled tying ablations for char-level models under ~10k params.
Green light — small, but genuinely unreported.

### Step 3 — The hypothesis (falsifiable, pre-registered)

> If I share the embedding matrix with the output projection (the ONLY change), final
> validation loss after 3,000 steps will differ from the untied model by less than the
> seed-to-seed noise band (Welch's t-test, α = 0.05).

Null outcome written in advance: if the difference is within noise, we conclude tying is
free at this scale and the parameter saving is pure profit. If tied is significantly
worse, we conclude tiny models need the extra output capacity. Either answer is a result.

### Step 4 — Experiment design

```
conditions   tied vs untied output matrix  ← the ONE variable
held fixed   corpus (4,044 chars Shakespeare), 90/10 split, vocab,
             context=4 chars, d=16, hidden=48, 3,000 steps,
             batch 16, lr 0.25, init draw order, eval set
seeds        {1, 2, 3} per condition  → 6 runs total
metric       mean cross-entropy (nats) on held-out validation chars
analysis     mean ± std per condition; Welch's t-test
             (lesson 20's method)
decision     p < 0.05 → real difference; else → within noise
```

The model is a from-scratch char-LM: 4 context chars → embeddings → tanh MLP → project
back to embedding dim → logits. The architecture forces the design to be clean: the
tied condition literally *reads the embedding table* as the output matrix, so the two
conditions differ by storage-sharing and nothing else. Untied has 4,928 parameters,
tied 4,416 — the 512-parameter saving IS the point under test.

One subtlety worth seeing: in the tied model, the embedding table receives **two**
gradient contributions per example — one from the input side (lookup) and one from the
output side (logits). That double duty is exactly why tying might act as a regularizer
(Inan et al.'s claim) or might overconstrain a tiny model. The experiment decides.

### Step 5 — The run

[index.ts](index.ts) is the entire experiment in one file — data, model, manual
backprop, seeded PRNG, training, statistics, verdict. It runs in ~15 seconds:

```
npx ts-node AI-researcher-lessons/30-capstone-research-project/index.ts
```

### Step 6 — Results (actual output of index.ts)

```
condition | val loss (mean ± std, 3 seeds)
tied      | 2.5655 ± 0.0813
untied    | 2.4787 ± 0.0510

Welch's t-test: t = 1.566, df = 3.36, two-sided p = 0.205
```

Per-run, for the record: tied {2.4752, 2.5884, 2.6329}, untied {2.4208, 2.5171, 2.4983}.
(Uniform guessing over the 32-char vocab would give ln 32 ≈ 3.47 — both models clearly
learned.)

### Step 7 — Honest conclusion

The gap is 0.087 nats in the untied model's favor — but p = 0.205, well above 0.05.
**We cannot distinguish the conditions from seed noise.** The pre-registered verdict
fires: at this scale, weight tying appears to be free — it saves 512 parameters (10.4%
of the model) at no measurable cost.

Note what we do NOT say, even though the raw means tempt us: "untied is slightly
better". Every seed's untied loss *is* below its tied counterpart, and with 10 seeds
that trend might reach significance — that is a fair *future-work* sentence, not a
conclusion. With 3 seeds, the honest claim is "no detectable difference". Reporting the
temptation and resisting it is what [lesson 20](../20-statistical-significance/lesson.md)
and [lesson 21](../21-negative-results/lesson.md) were for.

### Step 8 — Limitations (drawn before anyone asks)

- One corpus, one size (~5k params), one architecture family (MLP char-LM, not a
  transformer) — no claim beyond this cell.
- Metric is loss only; sample quality untested.
- 3 seeds bounds our sensitivity: effects smaller than ~0.1 nats are invisible to us.
- Fixed step budget; tying might change *convergence speed*, which final-loss-only
  measurement can't see.

### Step 9 — The writeup

Pour it into lesson 28's skeleton: title ("Weight Tying Is Free at 5k Parameters"),
4-sentence abstract (problem/gap/method/result — sentence 4 now has real numbers),
money figure (two loss curves with ±std bands), the table above, limitations verbatim.
As a README in a public repo, that is a complete, honest, citable piece of research.
Small — but *finished*, which beats large-but-abandoned every time.

---

## Your turn: the assignment

Run the identical loop on **your** question — the one you refined in
[lesson 27's exercise](../27-finding-a-research-question/exercise.md) and skeletonized
in [lesson 28's exercise](../28-writing-a-paper/exercise.md). If that question died,
adopt a variant of the worked example (tying × model width? tying × context length?
tying under dropout?) — the loop matters more than the question.

### Definition of done

```
□ HYPOTHESIS DOC   lesson 27's question document, updated to final form,
                   with the decision rule written BEFORE the runs
□ EXPERIMENT       ≥ 2 conditions × ≥ 3 seeds, seeded PRNG, one variable
                   changed, everything else demonstrably held fixed
□ STATISTICS       mean ± std per condition + Welch's t-test
                   (lesson 20's method) — no bare means anywhere
□ ABLATION         at least one: remove/vary a component of your method
                   and show what carries the effect (lesson 18)
□ WRITEUP          README in paper structure (lesson 28): title,
                   4-sentence abstract, money figure or results table,
                   method reproducible from the text, limitations section
□ PUBLIC REPO      pushed to the repo from lesson 29's exercise, runnable
                   with one command; share it per your engagement plan
```

Rules of engagement: pre-register your decision rule; report whatever the data says
(a clean negative result completes this capstone with full honors); respect your kill
criteria — three weeks maximum, then write up what you have.

---

## You are now doing research

Look at the distance covered. In the [from-scratch track](../../lessons/) you started
with "what is a model?" and ended having built linear regression, backprop, tokenizers,
attention, a working GPT-2 architecture, KV caches, sampling, RLHF, and a
[GPT-4-flavored capstone](../../lessons/30-gpt4-capstone/lesson.md) — every matrix
multiplied by code you wrote.

In this track you learned the researcher's other half: the math language (Phase A), why
training works at all (Phase B), how to read, reproduce, and critique the literature
(Phase C), how to produce evidence that survives scrutiny — baselines, ablations,
statistics, negative results (Phase D), where the open problems live (Phase E), and
finally how to find a question, write it up, and join the community (Phase F).

And just now, you ran a complete research cycle and published it. That is not
"preparing to do research". That IS research — the same loop Anthropic or DeepMind runs,
at a scale that fits your laptop. Scale changes the budget, not the epistemics.

What to do next, concretely:

1. **Run the loop again.** Your second cycle will take half the time. Your fifth will
   feel routine. The loop compounding in public is the whole career mechanism from
   lesson 29.
2. **Stay in your lane** for at least six months; depth is what gets noticed.
3. **Upgrade one constraint per project**: a real GPU, a transformer instead of an MLP,
   a benchmark from the literature, a collaborator from a Discord.
4. **Keep the standards** you practiced here — seeds, error bars, pre-registration,
   limitations — especially when nobody is checking. That, more than any affiliation,
   is what makes someone a researcher.

## What's next

There is no next lesson file. The next lesson is the one you write yourself: your
second research question, your next experiment, your next writeup. Go run the loop.

# Lesson 12 — How to Read a Paper

---

## The problem: papers are not written to be read front-to-back

You open "Attention Is All You Need" for the first time. Eleven pages. Dense math.
Words like "auto-regressive" and "sinusoidal positional encoding". You start at the
abstract, get to page 3, hit an equation with symbols you've never seen, and stop.

This happens to *everyone*. Here is the secret nobody tells beginners:

**Researchers do not read papers the way you read a book.**

A paper is not a story you follow linearly. It is a **database you query**. Experienced
researchers "read" 20 papers a week, but they only truly *read* one or two. The rest they
interrogate: What's the claim? Where's the evidence? Is it relevant to me? Next.

This lesson teaches you the querying skill: what each section of a paper is *for*, the
three-pass reading method, how to survive math you don't understand yet, and how to take
notes that are still useful six months later.

---

## Anatomy of an ML paper: what each section is FOR

Every ML paper has roughly the same skeleton. Each part has a *job*, and knowing the job
tells you how much to trust it.

```
┌─────────────────────────────────────────────────────────────┐
│ TITLE + ABSTRACT   the sales pitch (150 words)              │
├─────────────────────────────────────────────────────────────┤
│ 1. INTRODUCTION    the sales pitch, extended (1 page)       │
│ 2. RELATED WORK    "we know the field" + positioning        │
│ 3. METHOD          what they actually built  ← the substance│
│ 4. EXPERIMENTS     the evidence              ← the substance│
│ 5. ABLATIONS       which parts actually matter              │
│ 6. LIMITATIONS     the fine print (often too small)         │
│ 7. CONCLUSION      the sales pitch, again                   │
│ APPENDIX           where the real details hide              │
└─────────────────────────────────────────────────────────────┘
```

**Abstract** — the claim in ~150 words. Job: convince you to keep reading. Trust level:
this is *advertising*. Every abstract says the method is novel, simple, and beats
baselines. Read it to learn what the paper *claims*, not what it *proved*.

**Introduction** — the claim with context: what problem exists, why current solutions
fall short, what this paper does about it. Job: motivation. Usually ends with a bulleted
list of contributions — that list is the paper's promise. Hold them to it.

**Related work** — a map of neighboring papers. Job: show the reviewers the authors know
the field, and position this work as different. For you as a beginner, this section is
*gold*: it's a free curated bibliography. When a paper is over your head, its related
work section tells you what to read first.

**Method** — the actual idea: architecture, loss function, algorithm. Job: precision.
This is where the math lives. This section plus Experiments is the real paper; everything
else is packaging.

**Experiments** — the evidence. Datasets, baselines, metrics, tables. Job: support the
claims from the intro. The most important reading skill in all of ML is comparing this
section against the abstract and asking: *does the evidence actually support the claim?*

**Ablations** — experiments where the authors remove or vary one component at a time.
Job: show *which part* of the method causes the improvement. A paper with a complicated
method and no ablations is asking you to take the whole thing on faith. (You'll build
ablations yourself in Lesson 18.)

**Limitations** — what the method can't do. Job: honesty (and, cynically, reviewer
appeasement). Often the most informative paragraph in the paper, written in the most
forgettable prose. Always read it.

**Appendix** — hyperparameters, proofs, extra results, failure cases. Job: overflow.
When you try to *reproduce* a paper (Lesson 15), the appendix is where you'll live. The
learning rate schedule that makes everything work is almost never in the main text.

A useful trust heuristic:

```
claims  →  abstract, intro, conclusion     (advertising: verify before believing)
substance → method, experiments, ablations (evidence: read carefully)
truth serum → limitations, appendix        (details: often more honest)
```

---

## The three-pass method

Adapted from S. Keshav's classic note "How to Read a Paper". Do not read a paper once,
slowly. Read it up to three times, fast then slower, and *decide between passes whether
to continue*. Most papers deserve only pass 1.

### Pass 1 — the 5-minute scan

Read ONLY:
1. Title, abstract
2. Section and subsection headings (just the headings)
3. Every figure and table, with captions
4. The conclusion

Then answer three questions:
- **Problem:** what problem is this solving?
- **Method:** what's the one-sentence idea?
- **Result:** what's the headline number or finding?

If you can't answer these after pass 1, that's fine — it tells you the paper assumes
background you don't have yet. Note *which* background (that's your reading list), and
move on.

Figures deserve special emphasis: authors put enormous effort into Figure 1 because they
know reviewers scan. In most ML papers, **Figure 1 + the main results table ≈ 70% of the
paper's content.**

### Pass 2 — the 1-hour real read

Read the whole paper start to finish, but:
- **Skip the proofs.** Read the theorem statement ("what is claimed"), skip the proof
  ("why it's true"). You can come back if it matters.
- **Don't stop at hard math** — use the survival technique in the next section.
- Write in the margins (or your notes): questions, objections, "why?", "how?".
- Look at every equation long enough to identify what each symbol *is* (a matrix? a
  scalar? a probability?), even if you can't follow the derivation.

After pass 2 you should be able to summarize the paper to a friend, explain the evidence,
and say what you didn't understand.

### Pass 3 — the re-implementation

This is the pass that makes you a researcher. Attempt to **re-derive or re-implement**
the paper: rebuild the method from the paper's description alone, as if you were the
author. Every gap in the paper's description becomes painfully visible ("wait, what
initialization? what happens at the sequence boundary?").

Pass 3 takes hours to weeks. You will do it maybe a dozen times a year, for the papers
that matter most to your work. Lesson 15 is one full guided pass 3.

```
                 papers you encounter
                        │
              ┌─────────▼─────────┐
   pass 1     │   100 papers/year  │   5 min each
              └─────────┬─────────┘
                        │  ~20% survive
              ┌─────────▼─────────┐
   pass 2     │   20 papers/year   │   1 hr each
              └─────────┬─────────┘
                        │  ~half of those
              ┌─────────▼─────────┐
   pass 3     │  ~10 papers/year   │   days each
              └───────────────────┘
```

---

## Worked example: three passes on "Attention Is All You Need"

Vaswani et al., 2017 (arXiv:1706.03762). You already built this architecture in the
companion track ([Lesson 12 — Attention](../../lessons/12-attention/lesson.md) and
[Lesson 13 — Transformer Block](../../lessons/13-transformer-block/lesson.md)), which
makes it the perfect specimen: you can check the paper against code you wrote.

### What pass 1 extracts (5 minutes)

- **Title:** "Attention Is All You Need" — a claim: you can drop recurrence entirely.
- **Abstract:** proposes the *Transformer*, based solely on attention, no RNN/CNN;
  better BLEU on translation; trains much faster because it parallelizes.
- **Figure 1:** the encoder-decoder diagram — stacked blocks of multi-head attention +
  feed-forward, with residual connections. (You built the decoder half in
  [Lesson 14 — Tiny GPT](../../lessons/14-tiny-gpt/lesson.md).)
- **Table 2:** BLEU 28.4 on English→German — better than all listed baselines at a
  fraction of the training cost (the "FLOPs" column is doing quiet but important work).
- **3-sentence summary:**
  1. *Problem:* RNNs process tokens sequentially, so translation models train slowly and
     struggle with long-range dependencies.
  2. *Method:* an architecture built entirely from attention (self-attention +
     feed-forward blocks), processing all tokens in parallel.
  3. *Result:* state-of-the-art BLEU on WMT translation at a fraction of the training
     cost of recurrent models.

### What pass 2 adds (1 hour)

- Section 3.2: scaled dot-product attention — the formula
  `softmax(QKᵀ/√d_k)V`, and *why* the √d_k (dot products grow with dimension; large
  logits push softmax into regions with tiny gradients). You derived this yourself in
  the companion track.
- Multi-head = run h=8 attentions in parallel on d/h-dim slices, concatenate. The reason
  given: one softmax average destroys information; multiple heads let the model attend to
  different things at once.
- Positional encodings: sinusoids injected at the input, because attention itself is
  order-blind.
- Section 5 (training): Adam with a *warmup-then-decay* learning rate schedule, dropout,
  label smoothing. Note how much of the result depends on these — the architecture is
  only half the paper.
- Table 3 is the **ablation table**: vary heads (1 head is worse, 32 is also worse),
  vary d_k, remove dropout (much worse). This is the section that tells you which knobs
  matter.
- Marginal questions a good pass 2 produces: "why sinusoids instead of learned
  positions?" (they test this — learned does about the same, Table 3 row E), "does this
  work outside translation?" (the paper barely knows what it started).

### What pass 3 would be

Re-implement the architecture from the paper alone and train it on a small dataset —
which is, almost exactly, what you did in companion lessons 12–14. If you did those, you
have *already done pass 3 on this paper* without knowing it. Notice what the paper
under-specifies: weight initialization details, exactly where layer norm goes (the
famous pre-norm vs post-norm question — the paper's figure and later practice disagree,
and it matters for training stability).

---

## How to read math you don't fully understand yet

The #1 beginner mistake: stopping at the first equation you can't follow. Do not stop.
Math in papers is a *notation for shapes and operations*, and you can extract most of its
meaning without following every derivation.

The survival technique, in order:

**1. Collect the symbols.** Before trying to "understand" an equation, make a symbol
table. What is each letter? Scalar, vector, matrix, set, distribution, function?

```
Attention(Q,K,V) = softmax(QKᵀ/√d_k)V

Q     matrix  (T × d_k)   queries — one row per token
K     matrix  (T × d_k)   keys
V     matrix  (T × d_v)   values
d_k   scalar              key dimension
T     scalar              sequence length
```

**2. Find the shapes.** Trace what shape goes in and what shape comes out of each
operation. `QKᵀ` is (T×d_k)·(d_k×T) = (T×T) — ah, a token-by-token score matrix. Shape
tracing alone often reveals what an equation *does* even when you can't say why.

**3. Try tiny numbers.** Set T=2, d=2, invent values, compute by hand or in a scratch
TypeScript file. An equation you have computed once with concrete numbers is never
mysterious again.

**4. Flag and continue.** If steps 1–3 don't crack it, write "?? eq 4 — don't get the
KL term" in your notes and *keep reading*. Very often a later paragraph, a figure, or
the experiments section explains in words what the equation said in symbols. Circle back
at the end.

**5. Route around it.** If an equation remains opaque after the whole paper, it's a
signal about *prerequisites*, not about your ability. Find the concept's name, and queue
it: the companion track and Phase A of this track exist precisely to fill those holes.

---

## Notes that survive: the 3-sentence summary

You will forget every paper you read. The question is whether your *notes* remember.
The minimum viable note — for every paper that survives pass 1 — is three sentences:

```
PAPER:   <title, authors, year, arXiv id>
PROBLEM: what gap or failure does this address?          (1 sentence)
METHOD:  what is the core idea, mechanism, or trick?     (1 sentence)
RESULT:  what is the strongest concrete evidence?        (1 sentence)
```

Rules that make notes survive:
- **Write in your own words.** Copy-pasting the abstract stores nothing in your head.
- **Numbers, not adjectives.** "BLEU 28.4 vs 26.3 baseline at ~1/4 the training FLOPs"
  survives; "significantly better results" is dead on arrival.
- **Add one honest line:** what you didn't understand, or what you don't believe yet.
  Six months later this line is the most valuable one.
- Keep them all in one place (one folder of markdown files, one long file — anything
  searchable). A note you can't find doesn't exist.

For pass-2 papers, extend the note with: key figure (describe it or sketch it),
baselines used, main ablation finding, limitations, and open questions.

---

## Red flags: what to watch for while reading

You'll learn full reviewer-grade critique in Lesson 16, but even on a first read, keep
an eye out for these:

- **Cherry-picked baselines.** The method beats baselines from 2019 — but where's the
  strong recent one everybody uses? A missing obvious baseline is rarely an accident.
- **No ablations.** A method with five new components and one final score. Which
  component did the work? Maybe just one — maybe just the extra training data.
- **Improvements within noise.** +0.2% with no error bars, one random seed. You'll see
  in Lesson 20 why that number often means nothing.
- **Abstract ≠ experiments.** The abstract says "solves long-context reasoning"; the
  experiments show +2% on one synthetic benchmark. Trust the table, not the prose.
- **Compute confound.** The new method also used 4× more compute or data than the
  baselines. Then what's being measured — the idea or the budget?

Spotting these is not cynicism; it's calibration. Most papers are honest but overclaim a
little, because the incentive system rewards it. Your job as a reader is to locate the
*actual* contribution, which is usually real but smaller than the abstract implies.

---

## The reading habit

Skills compound only with reps. A sustainable beginner cadence:

- **3 pass-1 reads per week** (15 minutes total). Sources: Lesson 13 shows you where
  papers come from and how to filter the firehose.
- **1 pass-2 read per week** (one hour, calendar-blocked).
- **1 pass-3 per month or two**, on a paper that matters to what you're building.

After ~30 pass-1 reads you'll notice something: papers become *predictable*. You'll know
where the learning rate hides, where bodies are buried in the appendix, and what Figure 1
will show before you see it. That pattern recognition IS the skill.

---

## Exercise for this lesson

See [exercise.md](exercise.md) — do pass 1 on three landmark papers and pass 2 on one of
them, with checklists and a notes template.

## What's next
[Lesson 13 → Navigating arXiv & the Research World](../13-navigating-arxiv/lesson.md)

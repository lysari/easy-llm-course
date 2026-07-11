# Lesson 28 — Writing a Paper

---

## The problem

You ran the experiment. The numbers are in a table. You "just" need to write it up.

Then you sit down and produce three paragraphs of mush, delete them, and conclude you are
"bad at writing". Here is the reframe this lesson is built on:

> **Writing is not the report you file after research. Writing IS research.**

If you cannot state your claim in one clear sentence, you do not yet know what your claim
is. If your experiment section is confusing to write, your experiment was confused.
Unclear writing is unclear thinking made visible — which makes writing the cheapest
debugging tool you own. Many researchers discover the hole in their logic *while writing
the paper*, not while running the code.

This lesson gives you the standard structure, the formulas that fill it, and the
beginner's on-ramp (the blog post).

---

## Intuition: a paper is a defense attorney's case file

A paper is not a diary of what you did ("first I tried X, then Y broke, then…").
It is a **case** built for a skeptical jury:

- The **claim** (abstract): what you assert, in one breath.
- The **exhibits** (figures & tables): evidence a juror can inspect directly.
- The **argument** (sections): why the evidence supports the claim and nothing else does.
- The **cross-examination you run on yourself** (limitations): the doubts, raised by you
  before opponents raise them.

Jurors are busy. Most read the claim, glance at the exhibits, and decide whether to hear
the full argument. Design for that reading order.

---

## The standard 8-page structure

Nearly every ML paper follows this skeleton. Each section has ONE job:

```
┌─────────────────────────────────────────────────────────────┐
│ TITLE        make the right people stop scrolling            │
│ ABSTRACT     the whole paper in 4 sentences                  │
├─────────────────────────────────────────────────────────────┤
│ 1 INTRO      why this matters + what we claim   (~1 page)    │
│ 2 RELATED    where we sit on the map            (~0.5 page)  │
│ 3 METHOD     what we built, reproducibly        (~1.5 pages) │
│ 4 EXPERIMENTS the evidence                      (~2.5 pages) │
│ 5 ANALYSIS/ABLATIONS which part does the work   (~1 page)    │
│ 6 LIMITATIONS what we do NOT claim              (~0.3 page)  │
│ 7 CONCLUSION  claim restated + what's next      (~0.2 page)  │
│ REFERENCES                                                    │
└─────────────────────────────────────────────────────────────┘
```

What each must accomplish:

- **Intro**: by its last line, the reader knows exactly what you claim and why they
  should care. If they stopped here, they could summarize your paper correctly.
- **Related work**: proves you know the map; positions you honestly (more below).
- **Method**: a competent reader could reimplement from this section alone. Every symbol
  defined, every hyperparameter somewhere (main text or appendix). You know the pain of
  under-specified methods from [lesson 15](../15-reproducing-a-paper/lesson.md) — don't
  inflict it.
- **Experiments**: setup first (data, baselines, metrics, seeds), then results. Baselines
  chosen the way [lesson 18](../18-baselines-and-ablations/lesson.md) taught: the
  *strongest* fair comparison, not a strawman.
- **Ablations**: remove each component; show which one carries the effect.
- **Limitations**: see below — this section builds trust, it doesn't burn it.
- **Conclusion**: no new information. Claim, evidence in one line, one honest direction
  for future work.

---

## Write the abstract LAST — the 4-sentence formula

The abstract is the most-read 100 words of the paper, and you cannot write it until the
results exist. Formula:

```
Sentence 1  PROBLEM   the context and why it matters
Sentence 2  GAP       what existing work doesn't answer
Sentence 3  METHOD    what we did about it
Sentence 4  RESULT    the concrete number + what it means
```

Worked example, using the capstone question from [lesson 30](../30-capstone-research-project/lesson.md):

> **(1)** Weight tying — sharing the input embedding and output projection matrices — is
> standard in large language models, saving millions of parameters.
> **(2)** Whether it helps or hurts at very small scale, where parameter budgets are
> tightest relative to task difficulty, is rarely reported.
> **(3)** We train character-level transformers (~50k parameters) on a fixed corpus under
> tied and untied output layers, 3 seeds each, holding all else constant.
> **(4)** Tying changes final validation loss by less than the seed-to-seed noise band
> (Welch's t-test, p > 0.05), suggesting the parameter savings come for free even at
> tiny scale.

Four sentences. A reader now knows the problem, the gap, the method, and the finding —
without opening the PDF. Practice: run this formula on any paper you read; if you can't
fill the four slots from their abstract, they wrote a bad abstract.

---

## Figures are the paper's skeleton

Eye-tracking of how researchers actually read: **title → abstract → figures → (maybe)
text**. Your figures carry more readers than your prose ever will. Consequences:

1. **Design figures FIRST.** Before writing a word of the results section, sketch the
   figures that would prove your claim. If no figure can show it, you may not have a
   result.
2. **Each figure must stand alone.** A reader seeing only the figure + caption should
   get the point. Caption formula: *what is plotted* + *what to notice* ("Fig. 2:
   Validation loss vs steps for tied (blue) and untied (orange), 3 seeds each, shaded =
   ±1 std. The bands overlap throughout training.").
3. **Figure 1 is the money figure** — the single picture that makes your claim. Papers
   are accepted or skimmed past on Figure 1.
4. Label axes with units, show variance (error bars / shaded bands — you have the tools
   from [lesson 20](../20-statistical-significance/lesson.md)), never truncate a y-axis
   to inflate an effect.

```
   good money figure                      what it silently proves
   ─────────────────────                  ───────────────────────
   loss                                    • effect direction
    │╲   ── control (3 seeds, ±std)        • effect size vs noise
    │ ╲╲ ── treatment                      • training was stable
    │  ╲╲____                              • both actually converged
    │   ╲_____═══════
    └──────────────── steps
```

---

## The intro formula

Five moves, in order — most good intros are exactly this:

```
1 CONTEXT        "LMs are everywhere; parameter efficiency matters."
2 PROBLEM        "Weight tying is assumed harmless, but…"
3 WHY HARD/OPEN  "…small-scale effects are masked at large scale, and
                  published ablations rarely control seeds."
4 OUR INSIGHT    "At tiny scale, seed-controlled comparison is cheap,
                  so the question can be answered cleanly."
5 CONTRIBUTIONS  "We contribute: (i)…, (ii)…, (iii)…"
```

The contributions bullet list is a contract: every bullet must be a *claim the paper
proves*, not an activity ("we trained models" is an activity; "tying costs < 0.02 nats
at 50k parameters" is a claim). Reviewers check bullets against evidence one by one —
so should you.

---

## Related-work etiquette

- **Compare honestly.** State what the nearest paper genuinely showed, then your delta.
  If their method wins in some setting, say so. Reviewers usually *are* the related work;
  they notice being strawmanned.
- **Cite generously.** Citations cost nothing and buy goodwill. When in doubt, cite.
  Cite the thing you actually built on (the blog post, the repo) not just the prestigious
  cousin.
- **Structure by idea, not by list.** "Prior work splits into A [1,2,3] and B [4,5]; we
  differ from A because…, from B because…" — not an annotated bibliography.
- Your 3-nearest-papers list from [lesson 27's exercise](../27-finding-a-research-question/exercise.md)
  is the seed of this section. That's why you wrote it.

---

## Limitations sections that build trust

Beginners fear the limitations section ("won't it get me rejected?"). Experienced
reviewers read it first — a sharp limitations section signals the authors understand
their own experiment. From [lesson 16](../16-critiquing-a-paper/lesson.md) you know
exactly what a critic will probe; preempt them:

- **Scope**: "Results are at char-level, ≤100k parameters; transfer to larger scale is
  untested." (State it; don't hedge everything else because of it.)
- **Confounds you couldn't remove**: fixed dataset, fixed tokenizer, one architecture.
- **What you did NOT show**: "We measure loss, not downstream quality."

Rule of thumb: every claim in the abstract should have its boundary drawn somewhere in
the paper. Overclaiming is the #1 credibility killer; a paper that claims less and
proves it completely beats one that claims more and proves half.

---

## LaTeX, Overleaf, and templates

The mechanical part, quickly:

- Papers are written in **LaTeX**. Don't learn it from a book; start from a template
  and edit. **Overleaf** (overleaf.com) is Google-Docs-for-LaTeX — zero install, free
  tier is plenty.
- Conference **style files** define the look: NeurIPS (`neurips_20XX.sty`), ICLR
  (`iclr20XX_conference.sty`), ICML — all downloadable from each conference's website,
  and Overleaf has one-click official templates for each.
- Minimum LaTeX you need: `\section{}`, `\cite{}` + a BibTeX file (copy entries from
  Google Scholar's "cite" button), `\begin{figure}` / `\includegraphics`,
  `\begin{table}`, and math mode `$L = -\log p$`. That's 95% of a paper.
- Even if you never submit anywhere, writing in the NeurIPS template is a useful
  discipline: the 8-page limit forces the ruthless cutting that makes writing clear.

---

## The blog post: the beginner's paper

You do not need a venue's permission to publish research. A **blog post** is a paper
with the formality removed and the clarity kept — same claim, same evidence, same
honesty, friendlier prose.

This is a real on-ramp, not a consolation prize: distill.pub articles are cited like
papers; Anthropic's early interpretability work (Circuits, induction heads) was
published as web articles; nanoGPT began as a repo + explanation and shaped how a
generation learned transformers; countless researchers were hired off a blog post that
demonstrated they could ask a question, run a clean experiment, and write it up.

The structure of a good research blog post IS the paper structure — intro-with-claim,
figures that stand alone, honest limitations — just shorter. Which means: practice one
and you've practiced both. Lesson 29 covers where to publish it and how it becomes a
career asset.

---

## Rewrite ruthlessly: three passes

Nobody writes clearly in draft one. The professional loop:

1. **Vomit pass** — get everything down, ignore quality.
2. **Structure pass** — reorder until each section does its one job; delete paragraphs
   that do no job. (Most drafts shrink 30% here. Good.)
3. **Sentence pass** — one idea per sentence, verbs over noun-piles, cut hedges.

```
BLOATED  "It can be seen that the utilization of weight tying does not
          result in a significant degradation of performance."
CLEAR    "Weight tying does not hurt: the loss gap is within noise."
```

Read it aloud. Where you stumble, the reader falls.

---

## The exercise

See [exercise.md](exercise.md) — build the full paper skeleton for your lesson-27
question: title, 4-sentence abstract, contribution bullets, figure sketches, section
outline. Plus three rewrite drills. In [lesson 30](../30-capstone-research-project/lesson.md)
you will pour real results into this skeleton.

## What's next
[Lesson 29 → Community & Career](../29-community-and-career/lesson.md)

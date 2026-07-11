# Exercise 12 — Read Three Papers

Time budget: ~2 hours total. You'll do **pass 1 on three papers** (15 min each), write a
3-sentence summary for each, then **pass 2 on one** of them (1 hour) using the notes
template below.

Do not skip the time limits. The point of pass 1 is learning to extract value *fast*;
if you take 40 minutes, you're doing a slow pass 2 badly, not a pass 1 well.

---

## Part 1 — Pass 1 on three papers (15 min each)

The three papers (all free on arXiv):

1. **"Attention Is All You Need"** — Vaswani et al., 2017 — arXiv:1706.03762
   (You've built this. Reading the original is a calibration exercise: how well does
   the paper match what you implemented in [../../lessons/13-transformer-block/lesson.md](../../lessons/13-transformer-block/lesson.md)?)
2. **"Language Models are Unsupervised Multitask Learners"** (GPT-2) — Radford et al., 2019
   (Not on arXiv — search for the PDF on the OpenAI site. Companion:
   [../../lessons/20-gpt2-architecture/lesson.md](../../lessons/20-gpt2-architecture/lesson.md))
3. **"Training Compute-Optimal Large Language Models"** (Chinchilla) — Hoffmann et al., 2022 — arXiv:2203.15556
   (Companion: [../../lessons/22-scaling-laws/lesson.md](../../lessons/22-scaling-laws/lesson.md))

### Pass-1 checklist (repeat per paper)

- [ ] Set a 15-minute timer. Seriously.
- [ ] Read the title and abstract.
- [ ] Read every section heading (headings only).
- [ ] Look at **every figure and table**, reading the captions.
- [ ] Read the conclusion.
- [ ] Close the paper. Without reopening it, write the 3-sentence summary:

```
PAPER:   <title, year, id>
PROBLEM: ................................................ (1 sentence)
METHOD:  ................................................ (1 sentence)
RESULT:  ...................... <with a NUMBER> ......... (1 sentence)
UNSURE:  one thing I couldn't figure out in 15 minutes
```

- [ ] Reopen the paper only to check: did you get the headline number right?

### Sanity checks on your summaries

- [ ] Each RESULT sentence contains at least one concrete number from the paper.
- [ ] Each summary is in your own words (test: no phrase longer than 5 words copied
      from the abstract).
- [ ] For GPT-2: your PROBLEM sentence should mention something about task-specific
      training/finetuning vs. general models. If it doesn't, re-scan the intro.
- [ ] For Chinchilla: your METHOD sentence should mention the trade-off between model
      size and training tokens. If it doesn't, look at the figures again.

---

## Part 2 — Pass 2 on ONE paper (1 hour)

Pick whichever of the three pulled at you most. Set a 60–75 minute timer. Read the whole
paper, skipping proofs, flagging (not stopping at) hard math. Fill this template as you
go — copy it into your notes folder first:

```markdown
# Pass-2 notes: <paper title>

## The claim (from abstract/intro)
- Main claim:
- Promised contributions (the intro's bullet list, in my words):
  1.
  2.

## Method
- Core mechanism, in my own words (3–5 sentences):
- Symbol table for the key equation(s):
  | symbol | type (scalar/vector/matrix/…) | meaning | shape |
  |--------|-------------------------------|---------|-------|
- One tiny concrete example I traced (or "didn't — why"):

## Evidence
- Datasets used:
- Baselines compared against:
- Headline result (exact numbers, exact table #):
- Strongest single piece of evidence:
- Weakest / most suspicious piece of evidence:

## Ablations
- What did they ablate, and what mattered most?
- What do I wish they had ablated but didn't?

## Limitations & honesty check
- What the limitations section admits:
- Does the abstract claim more than the tables show? Where?

## My leftovers
- ?? things I flagged and still don't understand:
- Concepts to queue for study:
- One question I'd ask the authors:
```

### Pass-2 checklist

- [ ] Every equation in the main text has a row-complete symbol table OR a `??` flag.
- [ ] I can state the main result's number, dataset, and baseline from memory.
- [ ] I found at least one gap between abstract-language and table-evidence
      (there is always at least one).
- [ ] I wrote at least one honest `??` — if you have zero, you weren't reading hard
      enough or the paper was too easy.
- [ ] I could explain this paper to a friend for 3 minutes without opening it.
      (Actually try — out loud. This step is embarrassing and irreplaceable.)

---

## Done?

Keep all four notes (three summaries + one pass-2 sheet) in a `paper-notes/` folder.
This folder is the beginning of your research memory — lessons 13–16 all add to it.

## What's next
[Lesson 13 → Navigating arXiv & the Research World](../13-navigating-arxiv/lesson.md)

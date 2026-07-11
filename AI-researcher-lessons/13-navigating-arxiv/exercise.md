# Exercise 13 — Trace a Paper's Lineage

Time budget: ~1 hour. You'll find one *recent* paper in an area you actually care about,
trace where it came from, find who implemented it, and compress the whole family tree
into a 5-line lineage map.

---

## Part 1 — Find your paper (15 min)

- [ ] Pick an area you genuinely care about right now (examples: tokenization, small
      efficient models, long context, evaluation, RLHF, interpretability — anything
      you've met in the companion track works).
- [ ] Find **one paper posted in the last month** in that area. Use any filter from the
      lesson:
      - Hugging Face Papers (huggingface.co/papers) — scan the last few days
      - Papers With Code — browse the area's task page, sort by date
      - arXiv `cs.CL` / `cs.LG` recent listings (the raw firehose — allowed, painful)
- [ ] Selection criteria: (a) posted within ~30 days, (b) you can say *why* it caught
      your eye in one sentence, (c) it's a research paper, not a survey.
- [ ] Record: title, authors, arXiv ID, date, one sentence on why you picked it.
- [ ] Run pass 1 on it (15 min, Lesson 12 rules). Write the 3-sentence summary
      (problem / method / result).

## Part 2 — Trace backward (15 min)

- [ ] Open the paper's **method section** (not related work) and find the 2–4 references
      the method actually *depends on* — the ones cited where the mechanism is defined.
- [ ] Look the paper up on Semantic Scholar and check its reference list. Identify its
      **3 most-cited references** (Semantic Scholar shows citation counts inline).
- [ ] For each of the three, write ONE line: `<year> <short title> — what this ancestor
      contributed to the new paper`.
- [ ] Sanity check: do the method-section dependencies and the 3 most-cited references
      overlap? (Often 1–2 do; heavily-cited-but-not-load-bearing references are usually
      courtesy citations to famous papers — notice the difference.)

## Part 3 — Find the implementation (15 min)

- [ ] Check, in this order, until you find code:
      1. The paper's own abstract/footnotes (authors often link a repo)
      2. Papers With Code — the paper's page lists official + community implementations
      3. GitHub search: paper title or its distinctive method name
      4. Hugging Face Hub: search models/datasets tagged with the paper's arXiv ID
- [ ] Record: repo URL, official or third-party, stars/activity, and — open the code for
      5 minutes — one line on whether the core method is findable in the source (name
      the file if you can).
- [ ] If NO implementation exists anywhere: that's a real finding. Record it, and note
      what that means (too new? too expensive to reproduce? nobody cared? — all three
      are informative, and the last one is a quality signal).

## Part 4 — The 5-line lineage map (10 min)

Compress everything into exactly five lines:

```
1. ANCESTOR A (year): <one-line contribution it feeds into the paper>
2. ANCESTOR B (year): <one-line contribution>
3. ANCESTOR C (year): <one-line contribution>
4. → PAPER (last month): <one-line: what it adds on top of 1–3>
5. → CODE: <repo link or "none found"> — <one-line state of implementation>
```

Example (for a hypothetical paper):

```
1. RoPE (2021): rotary position encoding used unchanged as the base
2. YaRN (2023): the interpolation scheme this paper modifies
3. Needle-in-a-haystack eval (2023): the benchmark all claims rest on
4. → LongRope++ (2026-06): learns the interpolation instead of hand-tuning it
5. → CODE: github.com/authors/longropepp — official, training script included
```

## Checklist before you're done

- [ ] The paper is genuinely from the last month (check the arXiv date, not v1 of some
      updated older paper).
- [ ] Each ancestor line says what it *contributed*, not just its title.
- [ ] Line 4 states the delta over the ancestors — if you can't state the delta, the
      pass-1 read didn't land; take 10 more minutes.
- [ ] File the lineage map + 3-sentence summary in your `paper-notes/` folder.

**Optional (+30 min):** forward-trace one of the three ancestors on Semantic Scholar's
"cited by" — find one *other* recent descendant of the same ancestor, and note in one
line how the two descendants took the idea in different directions. This is how research
areas start to feel like ongoing conversations instead of piles of PDFs.

## What's next
[Lesson 14 → Landmark Papers Reading List](../14-landmark-papers/lesson.md)

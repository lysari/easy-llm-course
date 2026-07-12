# Exercise 16 — Write Your First Review

Time budget: ~2 hours. Part 1 gives you a reviewer checklist you'll reuse for years
(including on your own papers in Phase D and F). Part 2 is the real thing: a 300-word
review of a paper you already know.

---

## Part 1 — The reviewer checklist (keep this)

Copy this into your `paper-notes/` folder as `reviewer-checklist.md`. Run it against
any paper you're reviewing — including, later, your own drafts.

```markdown
# Reviewer checklist — <paper>

## Claim–evidence ladder
- [ ] Strongest sentence in the abstract, quoted: "____________"
- [ ] Which table/figure is supposed to prove it: ______________
- [ ] The claim rewritten at the rung the evidence supports:
      "____________________________________________________"
- [ ] Gap size: none / one rung / multiple rungs

## Baseline fairness
- [ ] How were baseline hyperparameters chosen? (quote or "not stated")
- [ ] Equal parameter count / data / FLOPs across the comparison? 
- [ ] Are the baselines the CURRENT strong ones? (Lesson 13 forward-trace)
- [ ] Any obvious missing baseline: _____________________________

## Ablations
- [ ] Is the paper's NAMED contribution ablated in isolation?
- [ ] Do component deltas roughly add up to the headline gain?
- [ ] Most important MISSING ablation: ___________________________

## Statistical hygiene
- [ ] Number of seeds/runs per number: ______ Error bars? _______
- [ ] Any best-of-k / max-over-checkpoint selection? ____________
- [ ] Test set used for any development decision? Contamination
      check reported? _______________________________________

## Honesty audit
- [ ] Limitations section: what does it admit that the abstract
      doesn't? _____________________________________________
- [ ] Anything important living only in the appendix? ___________

## The other direction (mandatory — this is what keeps you calibrated)
- [ ] Strongest single piece of evidence in the paper: ___________
- [ ] One thing this paper does BETTER than typical papers: ______
- [ ] One idea here worth stealing for my own work: ______________
```

---

## Part 2 — The 300-word review

**Assignment:** pick ONE paper from Lesson 14's core 8 that you've already read
(Transformer, GPT-2, Kaplan, and InstructGPT work especially well — they're rich in
reviewable choices). Review it as if it landed on your desk today, unpublished, from
unknown authors.

That framing is the exercise's engine: you must judge the *evidence in the PDF*, not
the paper's legendary status. Would Kaplan's single-run loss curves survive your
statistical-hygiene section? Does GPT-2 ablate anything? Pretend you don't know how
the story ends.

### Steps

- [ ] Re-skim your pass-2 notes from Lesson 12/14 (don't re-read the whole paper —
      reviewers work from one careful read plus notes, and so can you).
- [ ] Run the Part 1 checklist against the paper (30–40 min).
- [ ] Write the review, **300 words max**, in exactly this structure:

```markdown
# Review: <paper title>          Recommendation: <accept / weak accept /
                                  weak reject / reject>  Confidence: <1-5>

## Summary (2-3 sentences, MY words)
...

## Strengths (exactly 2, each 2-3 sentences)
S1: ...
S2: ...

## Weaknesses (exactly 2, each: problem → why it matters → what would fix it)
W1: ...
W2: ...

## Question for the authors (exactly 1)
Q1: ...
```

### Rules that make it a real review

- [ ] Every strength is **specific**: it names a table, figure, section, or design
      decision. "Well written" and "important problem" are banned.
- [ ] Every weakness is **actionable**: it says what experiment or change would
      resolve it. A weakness with no fix is a complaint.
- [ ] The question is something you genuinely can't answer from the paper — the test:
      would the authors' answer change your recommendation?
- [ ] 300 words. Count them. Compression forces judgment; padding hides its absence.
- [ ] You committed to a recommendation and a confidence. Fence-sitting ("interesting
      work, some concerns") is the cardinal reviewer sin.

### Calibration step (the payoff — don't skip)

- [ ] If you reviewed a paper with public reviews (Transformer and many core-8
      descendants have OpenReview threads for their venues or follow-ups), find real
      reviews of it — or of any paper you know well — on openreview.net.
- [ ] Compare: did the real reviewers hit your weaknesses? Did they find things you
      missed? Did any of them commit the sins above (vague strengths, unfixable
      weaknesses, no commitment)?
- [ ] Write 3 sentences on the difference between your review and the best real
      review you read. That difference is your to-do list as a reviewer.

File the checklist, the review, and the calibration note in `paper-notes/`. You have
now read papers, traced them, scheduled them, reproduced one, and judged one — the
full consumer side of research. Phase D flips you to the producer side, starting with
how to design experiments that would survive a reviewer like the one you just became.

## What's next
[Lesson 17 → The Scientific Method for ML](../17-scientific-method/lesson.md)

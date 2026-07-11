# Exercise 14 — The Core-8 Reading Schedule

One paper per week, eight weeks. Each week: one pass-2 read (Lesson 12 method), one
3-sentence summary, and **one figure re-drawn by your own hand**. That's ~90 minutes a
week. Slower is fine; skipping the drawing is not — sketching a figure is where half the
understanding happens.

Copy this whole file into your `paper-notes/` folder and fill it in as you go.

---

## The schedule

| Week | Paper | Where | Companion lesson |
|------|-------|-------|------------------|
| 1 | AlexNet (2012) | search "ImageNet Classification with Deep Convolutional Neural Networks" | [../../lessons/06-neural-network/lesson.md](../../lessons/06-neural-network/lesson.md) |
| 2 | word2vec (2013) | arXiv:1301.3781 | [../../lessons/10-embeddings/lesson.md](../../lessons/10-embeddings/lesson.md) |
| 3 | Bahdanau attention (2014) | arXiv:1409.0473 | [../../lessons/12-attention/lesson.md](../../lessons/12-attention/lesson.md) |
| 4 | Transformer (2017) | arXiv:1706.03762 | [../../lessons/13-transformer-block/lesson.md](../../lessons/13-transformer-block/lesson.md) |
| 5 | GPT-2 (2019) | OpenAI site, "Language Models are Unsupervised Multitask Learners" | [../../lessons/20-gpt2-architecture/lesson.md](../../lessons/20-gpt2-architecture/lesson.md) |
| 6 | Kaplan scaling laws (2020) | arXiv:2001.08361 | [../../lessons/22-scaling-laws/lesson.md](../../lessons/22-scaling-laws/lesson.md) |
| 7 | GPT-3 (2020) | arXiv:2005.14165 | [../../lessons/24-in-context-learning/lesson.md](../../lessons/24-in-context-learning/lesson.md) |
| 8 | InstructGPT (2022) | arXiv:2203.02155 | [../../lessons/28-rlhf-ppo/lesson.md](../../lessons/28-rlhf-ppo/lesson.md) |

Weekly rhythm that works: pass 1 on Monday (15 min), pass 2 midweek (60 min), summary +
figure on the weekend (20 min). Adjust freely; keep the order.

---

## Per-week template (copy 8 times)

```markdown
## Week N — <paper>                              date started: ____

### Pass 1 (15 min)
- [ ] Timer set, title/abstract/headings/figures/conclusion only
- First-impression line: ______________________________________

### Pass 2 (60 min)
- [ ] Read fully, proofs skipped, ?? flags written instead of stalling
- [ ] Symbol table made for the ONE most important equation
- ?? things I flagged: _________________________________________

### 3-sentence summary (own words, number in the RESULT line)
PROBLEM: _______________________________________________________
METHOD:  _______________________________________________________
RESULT:  _______________________________________________________

### Figure, re-drawn by hand
- Figure chosen: Fig ___ — why this one: ______________________
- [ ] Drawn on paper (or tablet), from memory first, then corrected
      against the original
- One thing I only noticed BY drawing it: _____________________

### Story thread
- The question the PREVIOUS paper left open: __________________
- How THIS paper answers it: __________________________________
- The question this paper leaves open (check: next week's paper
  should answer it): __________________________________________

### Honesty corner
- One claim I'm not convinced the evidence supports: __________
- Something here that later work overturned (if you know it): _
```

---

## Figure-drawing guidance

Which figure to draw, per paper, if you want a default:

- **AlexNet:** the architecture diagram (the two-GPU split is part of the story).
- **word2vec:** the CBOW/skip-gram diagram — or plot 6–8 word vectors' analogy geometry
  yourself from the idea, which is better.
- **Bahdanau:** a Figure-3 alignment heatmap. Draw the grid; feel the soft alignment.
- **Transformer:** the full block diagram, from memory. You built this — the test is
  whether your hand knows where the residuals and layer norms go.
- **GPT-2:** the zero-shot performance-vs-model-size curves (pick one task).
- **Kaplan:** Figure 1, all three panels — the straight lines on log-log axes. Label the
  axes carefully; the log scales ARE the finding.
- **GPT-3:** the few-shot vs zero-shot vs one-shot scaling curve (Fig 1.2).
- **InstructGPT:** the three-stage RLHF pipeline (Fig 2), annotated with what each stage
  consumes and produces.

Rules: draw from memory first, then open the paper and correct in another color. The
corrections are the lesson. Photograph each drawing and file it with the week's notes.

---

## Completion checklist

- [ ] 8 summaries filed, each RESULT line with a real number
- [ ] 8 hand-drawn figures photographed and filed
- [ ] The "story thread" lines connect: each week's open question is answered by the
      next week's paper (if a link feels broken, reread that lesson-14 section)
- [ ] I can tell the whole 15-paper story out loud in under 3 minutes (test it on a
      friend or a rubber duck)

When all four boxes are checked you have something rare: not eight papers read, but one
story understood. That story is the context every new paper you'll ever read fits into.

## What's next
[Lesson 15 → Reproducing a Paper](../15-reproducing-a-paper/lesson.md)

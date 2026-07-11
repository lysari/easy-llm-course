# Lesson 15 — Reproducing a Paper

---

## The problem: reading is not believing

You've read the landmark papers (Lesson 14). You know how to interrogate a paper's
claims (Lesson 12). But there's a level of understanding that reading can never give
you, and there's exactly one way to get it:

**Rebuild the result yourself, and see if you get the same numbers.**

Reproduction is where reading turns into research. It's pass 3 (Lesson 12) taken all the
way: not just re-implementing the *method*, but checking whether the method actually
produces the *claimed evidence*. Every experienced researcher will tell you the same
thing: they learned more from their first serious reproduction attempt than from their
first hundred paper reads.

Why it's so effective:

- **Papers under-specify.** Reproduction forces you to find every hidden decision — the
  initialization, the learning-rate warmup, the data cleaning step mentioned nowhere.
  Each gap you hit is something the field knows but doesn't write down. You're
  downloading tacit knowledge.
- **You develop calibration.** After reproducing three papers, you read every future
  paper differently — you know which claims are load-bearing, which numbers are fragile,
  and what "we found this straightforward to implement" is hiding.
- **It's the on-ramp to original work.** Nearly every first research project is
  "reproduced X, noticed something weird, pulled the thread." You can't notice something
  weird in a system you didn't build.

And a field-level truth: ML has a reproducibility problem. Studies attempting to
reproduce published results routinely fail on a substantial fraction — missing code,
missing hyperparameters, results that were noise. Reproducers are the field's immune
system.

---

## Choosing a target: the reproducibility checklist

Your first reproduction should be chosen for *tractability*, not glamour. Score
candidates against this list:

```
GOOD FIRST TARGET                      BAD FIRST TARGET
─────────────────────                  ─────────────────────
small model, hours of compute          "trained on 1024 GPUs for 3 weeks"
public, downloadable data              proprietary or scraped-then-lost data
one clear headline number/curve        vibes-based claims, human evals
hyperparameters in the appendix        "details in a future version"
official or community code exists      no code anywhere (see Lesson 13, Part 3)
simple metric (loss, accuracy)         metric needs a judge model or user study
```

Sweet spots for a from-scratch TypeScript builder like you: scaling-law fits at tiny
scale, tokenizer comparisons, optimizer ablations (Adam vs SGD — you built both),
positional-encoding comparisons, sampling-strategy effects. Notice these are all
*claims about trends*, not absolute numbers — trends survive miniaturization; absolute
numbers don't.

**Reproduce the claim, not the paper.** You will not train GPT-3. You don't need to.
Most landmark claims have a *shape* that shows up at toy scale — and checking whether
the shape survives is genuine science. That's exactly what this lesson's code does.

---

## The reproduction workflow

```
 1. READ            pass 2 the paper; state THE claim you're reproducing
       │            in one sentence with a number in it
       ▼
 2. EXTRACT         list EVERY hyperparameter and detail: model dims, init,
       │            optimizer, lr + schedule, batch size, steps, data prep,
       │            eval protocol. Mark each one: STATED / GUESSED / MISSING
       ▼
 3. IMPLEMENT       build it. When you hit a MISSING, make the most standard
       │            choice, and WRITE DOWN that you guessed
       ▼
 4. COMPARE         run, put your numbers next to theirs in one table
       ▼
 5. INVESTIGATE     gap too big? change ONE guessed detail at a time,
                    re-run, repeat. (This loop is where the learning is.)
```

Step 2 is the one beginners skip and regret. The hyperparameter inventory — with each
entry honestly marked STATED / GUESSED / MISSING — is the difference between debugging
and flailing. When your numbers don't match, the MISSING list *is* your suspect list.

---

## What "close enough" means

You will almost never match a paper's numbers exactly, and exact match isn't the goal.
Three tiers of success:

1. **Qualitative reproduction** — the claimed *effect* appears: the curve bends the same
   way, method A still beats method B, the power law is still a straight line on log-log
   axes. This is the scientifically important tier.
2. **Quantitative agreement** — your numbers land within noise of theirs (run a few
   seeds to know what noise is — Lesson 20 makes this precise). Realistic when you
   have their code and data.
3. **Exact match** — same numbers to several decimals. Only possible with identical
   code, data, seeds, and hardware. Rarely worth chasing; floating-point
   non-determinism alone can break it.

Aim for tier 1 always, tier 2 when the artifacts allow. A tier-1 success at 1/1000th the
paper's scale is a real result. **A tier-1 failure is also a real result** — if the
effect vanishes under faithful conditions, you've learned the claim is fragile, scale
-dependent, or wrong.

---

## When numbers don't match: the usual suspects

Decades of collective reproduction pain, ranked roughly by frequency:

1. **Data preprocessing.** The silent killer. Tokenization details, lowercasing,
   deduplication, train/test split boundaries, shuffling. Two "identical" datasets that
   differ in cleaning can move results more than the paper's whole claimed improvement.
2. **Evaluation details.** Are they reporting best epoch or last epoch? Per-token or
   per-character loss? Which split? Greedy or sampled generation? A "gap" is often two
   different measurements of the same model.
3. **Learning-rate schedule.** Warmup steps, decay shape, final lr. Chinchilla's entire
   correction of Kaplan (Lesson 14) came down substantially to schedule handling —
   schedules don't just change results, they can change *conclusions*.
4. **Initialization.** Scale and scheme (you met this in
   [../../lessons/06-neural-network/lesson.md](../../lessons/06-neural-network/lesson.md)).
   Papers omit it constantly; frameworks differ in defaults.
5. **The quiet regularizers.** Dropout placement, weight decay (and what it's applied
   to), gradient clipping, label smoothing — each "minor," each worth more than many
   papers' headline delta.
6. **Plain bugs.** Yours *or theirs*. Both happen. Check yours first; suspect theirs
   only after your MISSING list is empty.

Debugging protocol: change one thing at a time, keep a log of every run
(config → result), and re-run your baseline after any code change. This log discipline
becomes Lesson 17's whole subject.

---

## When reproduction failures become papers

Reproduction is not just training wheels — it's a publishable genre:

- **Chinchilla vs Kaplan** (Lesson 14): re-examining a landmark's methodology overturned
  how the entire field sizes models. The correction is as cited as the original.
- **"Lottery ticket" follow-ups, optimizer-comparison studies, "X is not all you
  need"-style papers**: whole literatures exist of careful re-examinations showing
  claimed effects shrink or vanish under fair tuning (baseline fairness — Lesson 16's
  core topic).
- **ML Reproducibility Challenge**: an annual venue publishing reproduction
  reports of accepted papers. Browse a few published reports — they are the exact
  workflow above, written up, and are many researchers' first publication.

The professional norm when your careful reproduction disagrees: check your MISSING list
twice, then *email the authors* — politely, with your config table attached. Authors
usually respond; the missing detail usually surfaces; occasionally you've found
something real. Either way you win.

---

## This lesson's code: reproducing the scaling-laws claim in miniature

Kaplan et al. 2020 (Lesson 14, ★ #10) claims: **language-model loss falls as a power law
in parameter count**, `L(N) ≈ a · N^(−b)` — a straight line on log-log axes. They showed
it from 10³ to 10⁹ parameters. Question: does the *shape* survive at 10²–10³ parameters,
in TypeScript, on a few KB of text, in 30 seconds?

[index.ts](index.ts) runs the full reproduction workflow:

1. **The claim**: `L(N) = a·N^(−b)` with some b > 0 — stated up front, like step 1 says.
2. **Setup**: a fixed few-KB text corpus (embedded in the file), character-level
   next-char prediction — the smallest possible "language model" task.
3. **Models**: four from-scratch context-window MLPs (embedding → tanh hidden → softmax,
   the machinery you built in companion lessons
   [06](../../lessons/06-neural-network/lesson.md)–[11](../../lessons/11-softmax/lesson.md)),
   differing only in width: ~200 to ~3,500 parameters. Same data, same steps, same
   optimizer — *only N varies*, because a reproduction must isolate the claimed variable.
4. **Measurement**: held-out validation loss for each size.
5. **The fit**: take logs — `log L = log a − b·log N` — and run ordinary least-squares
   linear regression ([../../lessons/01-linear-regression/lesson.md](../../lessons/01-linear-regression/lesson.md)!)
   on the (log N, log L) points. The slope is −b.
6. **The verdict**: prints the table (size, params, loss, power-law prediction), the
   fitted exponent b, and R² of the log-log fit — then says explicitly whether the
   qualitative claim reproduced.

Run it:

```
npx ts-node AI-researcher-lessons/15-reproducing-a-paper/index.ts
```

What you should see: loss decreasing with N, a log-log fit with b in the rough
neighborhood of Kaplan's ~0.07–0.1 (toy scale is noisy — anywhere positive with a decent
R² is a tier-1 reproduction), and predictions that track the measured losses.

Things to try after it runs (each is a mini step-5 "investigate"):

- Change the seed. How much does b move? (That spread is your error bar — Lesson 20.)
- Halve the training steps. The fit degrades — under-trained models aren't on the
  compute-efficient frontier, one of Kaplan's own key caveats.
- Shrink the corpus 4×. Watch the big models stop improving — you've hit the *data*
  scaling wall, the other axis of the paper (and the door to Chinchilla).

Full treatment of the scaling-law math lives in the companion track:
[../../lessons/22-scaling-laws/lesson.md](../../lessons/22-scaling-laws/lesson.md).

---

## Code for this lesson

See [index.ts](index.ts) — the miniature Kaplan reproduction: four model sizes, one
power-law fit, one verdict. Read the header comment first: it's formatted as a
reproduction report (claim → inventory → result), which is the template for every
reproduction you'll do after this.

## What's next
[Lesson 16 → Critiquing a Paper](../16-critiquing-a-paper/lesson.md)

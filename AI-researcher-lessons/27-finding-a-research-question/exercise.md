# Exercise 27 — From Candidate to Committed Question

In [lesson 26's exercise](../26-open-problems/exercise.md) you picked a candidate
research question. Now you run it through the full pipeline from the lesson. The output
of this exercise is a one-page **question document** — you will reuse it directly in
[lesson 28](../28-writing-a-paper/exercise.md) (paper skeleton) and
[lesson 30](../30-capstone-research-project/lesson.md) (capstone).

If your lesson-26 question died on contact (it happens — that's the pipeline working),
generate a replacement using the five sources from the lesson: limitation sections,
"X but Y", your lesson-15 reproduction discrepancies, tool transfer, or an annoyance
from training your own models. Then continue below.

Copy the template at the bottom into `my-question.md` in this folder and fill it in.

---

## Step 1 — Literature check (budget: 1 hour, hard stop)

Follow the protocol from the lesson:

1. Write down your naive phrasing of the idea.
2. Search arXiv / Google Scholar with 2–3 phrasings; learn the canonical term.
3. Walk citations backward and forward from the best hit.
4. Read at most ~10 abstracts.

Deliverable: the **3 nearest papers**, each with:
- citation (authors, year, title),
- one sentence: what they showed,
- one sentence: how your question **differs** (different scale? different condition?
  different metric? they assumed X, you vary X?).

If you cannot state a difference in one sentence, your question is already answered —
read that paper's limitations section and pivot. That still counts as completing this step.

## Step 2 — Falsifiable hypothesis

One sentence, in this exact shape:

> "If I change **[the one variable]**, then **[measurable metric]** will
> **[direction/size]**, compared to **[controlled baseline]**."

Bad: "Curriculum learning helps small models."
Good: "If I order training batches short-to-long instead of shuffled, final validation
loss of a 4-layer char-transformer on tiny-shakespeare after 5,000 steps will be lower
by more than the seed-to-seed noise band."

Also write the **null outcome** explicitly: "If the difference is within noise, I
conclude ___." (You must be able to finish that sentence *now*, before running.)

## Step 3 — Minimum viable experiment design

Specify, concretely:

- **Conditions**: exactly 2 (treatment vs control). What is identical between them?
  (data, tokenizer, model size, steps, LR, eval set — list them.)
- **Seeds**: ≥ 3 per condition (you know why from
  [lesson 20](../20-statistical-significance/lesson.md)).
- **Metric**: the single number per run you will compare (e.g. final val loss).
- **Analysis**: mean ± std per condition, Welch's t-test across seeds — lesson 20's method.
- **Decision rule**: what result means "hypothesis supported" / "not supported" —
  written before you run anything.

## Step 4 — Resource estimate

Do the arithmetic:

```
runs = conditions × seeds            = 2 × 3 = 6
time per run (measure ONE pilot run) = ______ minutes
total                                = ______ hours
```

If total > one weekend on your machine, shrink something (model, steps, dataset) and
note what you shrank. Your MVE must fit your hardware — that is the third circle.

## Step 5 — Kill criteria

Write both:

- **Date kill**: "If not finished by ____ (pick a date ≤ 3 weeks out), I stop and write
  up whatever I have."
- **Result kill**: "If the pilot run shows ____ (e.g. training diverges in both
  conditions / metric is too noisy to ever separate), I stop and write up why."

---

## Template

```markdown
# Question Document — [short title]

## The question
[one sentence]

## Why it matters (the IMPORTANT circle)
[2–3 sentences: who would change what they do if this were answered?]

## Literature check
1. [Author Year, "Title"] — showed: ___. Differs from mine because: ___.
2. [Author Year, "Title"] — showed: ___. Differs from mine because: ___.
3. [Author Year, "Title"] — showed: ___. Differs from mine because: ___.
Search terms used: ___

## Hypothesis
If I change ___, then ___ will ___, compared to ___.
Null outcome: if the difference is within noise, I conclude ___.

## Minimum viable experiment
- Conditions: ___ vs ___ (held fixed: ___)
- Seeds: ___
- Metric: ___
- Analysis: mean ± std per condition, Welch's t-test
- Decision rule: ___

## Resource estimate
runs × time-per-run = ___ ; fits my machine: yes/no (if no: shrank ___)

## Kill criteria
- Date: ___
- Result: ___

## Quality checklist
[ ] Falsifiable  [ ] Measurable  [ ] Controlled
[ ] Novel-ish    [ ] Feasible    [ ] Killable
```

Done when every blank is filled and all six boxes are checked. Keep `my-question.md` —
lessons 28 and 30 build directly on it.

# Exercise 28 — The Paper Skeleton

Two parts. Part A builds the complete skeleton of the paper you will write in
[lesson 30](../30-capstone-research-project/lesson.md), using the question document
from [lesson 27's exercise](../27-finding-a-research-question/exercise.md). Part B is
sentence-level rewrite drills.

Write everything into `my-skeleton.md` in this folder.

---

## Part A — The skeleton

You have NOT run the experiment yet. That is the point: designing the paper first
exposes what evidence you actually need, so the experiment produces exactly that
evidence. Where a result would go, write the *shape* of the result with a blank:
"tying changes val loss by ___ ± ___".

### A1. Title (3 candidates, pick 1)

Write three, in different styles, then choose:
- the claim style: "Weight Tying Is Free at Tiny Scale"
- the question style: "Does Weight Tying Hurt Tiny Language Models?"
- the descriptive style: "A Seed-Controlled Study of Weight Tying in Character-Level Transformers"

Test: would the right reader stop scrolling? Would they learn the topic AND the angle?

### A2. Abstract — the 4-sentence formula

Exactly four sentences: PROBLEM / GAP / METHOD / RESULT (result with blanks for the
numbers). If a sentence tries to do two jobs, split the ideas and cut one.

### A3. Contribution bullets (2–4)

Each bullet a provable **claim**, not an activity. For each, note in parentheses which
future figure or table will prove it. If a bullet has no exhibit, cut the bullet or add
the exhibit.

### A4. Figure sketches (words, no plotting)

Describe 2–3 figures. For each:
- **What is plotted**: axes + units, what each line/bar is, how variance is shown.
- **What the reader should notice** (this becomes the caption's second sentence).
- **Which claim it proves.**

Figure 1 must be the money figure: if a reader sees ONLY this, do they get your claim?

### A5. Section outline — one sentence each

One sentence per section stating what THAT section will establish (not its topic —
its *job*):

```
1 Intro:        …
2 Related work: …  (name your 3 nearest papers from lesson 27 here)
3 Method:       …
4 Experiments:  …
5 Ablations:    …
6 Limitations:  …  (write the 2 biggest limits NOW — you already know them)
7 Conclusion:   …
```

If you can't write the sentence, you don't know the section's job yet — that is the
skeleton doing its work. Fix the design, not the wording.

---

## Part B — Rewrite drills

Rewrite each bloated sentence so it is shorter, active, and makes one concrete claim.
Do not look at the hints until you have your version.

**Drill 1**

> "It is worth noting that the performance of the model was observed to be improved to
> a certain extent when the proposed modification was utilized, although further
> experimentation may potentially be needed in order to fully confirm this phenomenon."

**Drill 2**

> "Due to the fact that the training process exhibited a considerable amount of
> variance across the different random seeds that were used, the making of definitive
> conclusions regarding the superiority of either one of the two methods is rendered
> somewhat difficult."

**Drill 3**

> "In this paper, an investigation into the question of whether or not the utilization
> of a curriculum-based ordering of the training data has an impact on the convergence
> behavior of small-scale language models is carried out by the authors."

<details>
<summary>Hints (after you've tried)</summary>

- Kill the throat-clearing: "It is worth noting that", "Due to the fact that",
  "In this paper" — delete, nothing is lost.
- Convert passive to active: "was observed to be improved" → "improved" (by how much?).
- Replace hedges with numbers or with silence: "to a certain extent" → "+0.8%" or cut.
- Possible targets (yours may be better):
  1. "The modification improves accuracy by ___%; we have not yet tested whether this
     holds across seeds."
  2. "Seed variance (±___) is larger than the gap between methods, so we cannot rank them."
  3. "We test whether curriculum ordering speeds convergence of small language models."
</details>

---

## Done when

- [ ] Title chosen from 3 candidates
- [ ] 4-sentence abstract (blanks allowed only in sentence 4)
- [ ] Every contribution bullet maps to a named figure/table
- [ ] Figure 1 alone communicates the claim
- [ ] One-sentence job for all 7 sections, including 2 real limitations
- [ ] 3 rewrites, each under half the original's word count

Keep `my-skeleton.md`. In lesson 30 you fill in the blanks with real numbers and it
becomes your first writeup.

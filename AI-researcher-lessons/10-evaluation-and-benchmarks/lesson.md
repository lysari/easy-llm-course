# Lesson 10 — Evaluation & Benchmarks

---

## The problem this lesson solves

Two labs each train a language model. Both claim theirs is better.
**Who's right? How would you even check?**

This sounds like the easy part of research — you did the hard work of training,
now just... measure. It is not the easy part. Evaluation is arguably the
**hardest open problem in the field**, because:

- a single number can lie (you'll compute a lying number in this lesson's code),
- the moment a number matters, people optimize the number instead of the goal,
- and the things we actually care about — "is it smart? is it honest? is it
  useful?" — resist being turned into numbers at all.

A researcher who can't evaluate can't do science: every claim in every paper
you'll read in Phase C stands or falls on its eval section. This lesson builds
your toolkit from the bottom (counting right answers) to the top (judging
open-ended text), flagging where each tool breaks.

---

## The analogy: grading exams

Evaluating a model is grading a student, and every grading scheme has a failure
mode:

- **Multiple choice** (accuracy): easy to grade, easy to game by memorizing past
  exams.
- **A rare-disease diagnosis test** (imbalanced classes): a student who writes
  "healthy" for every patient scores 95% — and is worthless.
- **Reading fluency** (perplexity): measures something real, but fluent ≠ correct.
- **Essay grading** (LLM-as-judge): rich signal, but the grader has moods and
  biases.
- **Leaked exam papers** (contamination): a perfect score that means nothing.

Every tool below is one of these graders, formalized.

---

## Level 1: classification metrics — beyond accuracy

Start with the simplest setting: a yes/no classifier (spam? tumor? bug?).
All four possible outcomes get names — the **confusion matrix**:

```
                        actually POSITIVE      actually NEGATIVE
predicted POSITIVE      TP (true positive)     FP (false positive / false alarm)
predicted NEGATIVE      FN (false negative /   TN (true negative)
                            miss)
```

From those four counts, the three metrics that matter:

```
accuracy  = (TP + TN) / (TP + TN + FP + FN)     "how often is it right?"

precision = TP / (TP + FP)     "when it says POSITIVE, how often is it right?"
                               (penalizes false alarms)

recall    = TP / (TP + FN)     "of the real positives, how many did it catch?"
                               (penalizes misses)
```

**Why accuracy lies.** Suppose 95% of emails are not spam. The classifier
`return "not spam"` scores **95% accuracy** while catching zero spam. Accuracy
rewards siding with the majority; on imbalanced data it measures the imbalance,
not the model. (The code makes this concrete: a do-nothing classifier beats a
genuinely useful one on accuracy.)

**Precision and recall pull in opposite directions.** Flag everything → perfect
recall, terrible precision. Flag nothing you're unsure of → high precision, bad
recall. The **F1 score** combines them:

```
F1 = 2 · (precision · recall) / (precision + recall)
```

This is the *harmonic* mean — it stays close to the SMALLER of the two, so you
can't buy a good F1 by maxing one and abandoning the other:
`precision 1.0, recall 0.01 → F1 ≈ 0.02` (an ordinary average would say 0.5).

**Researcher's rule: when someone reports one number, ask what the other numbers
were.** The choice of metric is a claim about what matters.

---

## Level 2: perplexity — the language model's exam

Classification metrics need labeled right answers. A language model's job is
"put high probability on real text", and its natural metric follows from
[Lesson 04 — Information Theory](../04-information-theory/lesson.md):

```
Given held-out text with tokens t₁ t₂ ... t_N:

avg negative log-likelihood:  H = −(1/N) · Σᵢ log P(tᵢ | t₁...tᵢ₋₁)

perplexity:  PPL = e^H     (or 2^H if you used log base 2)

P(tᵢ | ...) : probability the model assigned to the token that ACTUALLY came next
N           : number of tokens in the held-out text
H           : the model's average "surprise" per token (cross-entropy)
```

**The intuition:** perplexity ≈ "the model is as confused as if it were choosing
uniformly among PPL options at every step."

```
PPL = 1    : perfect — the model knew every next token with certainty
PPL = 50   : each step feels like a 50-way coin flip
PPL = vocab size : the model learned nothing (uniform guessing)
```

Concrete tiny example (the code computes one like it): if a model assigns the
true next characters probabilities `0.5, 0.25, 0.5, 0.125`, then

```
H = −(ln 0.5 + ln 0.25 + ln 0.5 + ln 0.125) / 4 = 1.2130
PPL = e^1.2130 ≈ 3.36     "typically ~3.4 plausible options per step"
```

Perplexity is the metric of pretraining — it's what the loss curve *is*
(cross-entropy, exponentiated), and it's what
[scaling laws](../../lessons/22-scaling-laws/lesson.md) are laws *about*. Its
limits: only comparable between models with the **same tokenizer** and same eval
text; and it measures *fluency of prediction*, not truthfulness, safety, or
usefulness. GPT-2's PPL improved smoothly while its arithmetic stayed terrible.

---

## Level 3: benchmark suites — the standardized tests

To compare models on *abilities*, the field built shared exam sets. The three
you'll see in every LLM paper:

| benchmark | what it is | what it tries to measure |
|---|---|---|
| **MMLU** | ~14k multiple-choice questions across 57 subjects (law, medicine, math, history...) | broad knowledge + reasoning; reported as accuracy |
| **GSM8K** | ~8.5k grade-school math word problems, free-form numeric answers | multi-step arithmetic reasoning (usually with chain-of-thought) |
| **HumanEval** | 164 Python function specs with hidden unit tests | code generation; reported as pass@k — "of k sampled solutions, does any pass?" |

Random-guessing floor on MMLU: 25% (4 choices). GPT-3: ~44%. GPT-4: ~86%.
Numbers like these are the field's public scoreboard — which is precisely the
problem.

### Failure mode 1: contamination

Benchmarks are text. Pretraining data is the internet. **The exam questions are
often IN the training set** — copied into blogs, GitHub repos, forum answers.
A model can "ace" MMLU the way a student aces a leaked exam: by having seen it.

This connects straight to [Lesson 07](../07-generalization/lesson.md): a
contaminated benchmark measures *train* performance while claiming to measure
*test* performance — the one distinction that matters. Labs now run
decontamination (n-gram matching between benchmark and training data), but it's
imperfect: paraphrases slip through. Detecting and quantifying contamination is
an active research area.

### Failure mode 2: Goodhart's law

> "When a measure becomes a target, it ceases to be a good measure."

Once a benchmark decides funding, hiring, and headlines, everything optimizes
toward it — model choices, data mixes, prompt formats, cherry-picked eval
settings — and the benchmark number drifts away from the ability it was a proxy
for. You've already met this dynamic:
[reward hacking in RLHF](../../lessons/28-rlhf-ppo/lesson.md) is Goodhart's law
running at gradient speed. Benchmarks *saturate* (everyone scores 90%+) and stop
discriminating; the field builds a harder one; repeat. That treadmill is
half the history of NLP evaluation.

---

## Level 4: evaluating open-ended text

"Summarize this paper", "write a kind email declining the invite" — there is no
answer key. Options, each with real problems:

**Human evaluation** — the gold standard: show people two outputs, ask which is
better (this is exactly the preference data that trains
[reward models](../../lessons/27-reward-model/lesson.md)). Slow, expensive,
noisy, and humans have biases too (they favor long, confident-sounding answers).

**LLM-as-judge** — use a strong LLM to grade outputs. Cheap, fast, surprisingly
correlated with human judgment... and systematically biased:

```
known judge biases (measured in the literature):
  position bias    : prefers whichever answer is presented FIRST
  length bias      : prefers longer, more elaborate answers
  self-preference  : rates its own family's outputs higher
  style over substance: confident fluent wrongness beats hesitant correctness
```

Mitigations researchers use: swap the answer order and average; grade against a
rubric instead of head-to-head; use judge ensembles. Studying judge reliability
is itself a research subfield.

**"Vibes evals"** — every researcher also just... talks to the model. This
exists because the formal metrics demonstrably miss things: a model can gain 5
benchmark points and feel obviously worse — more refusals, stiffer prose, worse
instruction-following. Vibes are unfalsifiable and unpublishable, but they are
the smoke detector that tells you your metrics are measuring the wrong thing.
The mature position: **vibes generate hypotheses; metrics test them.** When the
two disagree, investigate — that disagreement is often where a paper lives.

---

## The researcher's checklist

When you read (or write!) an eval section, ask:

1. **What's the base rate?** Would a trivial baseline (majority class, "I don't
   know", copy the input) score well? (Level 1's lesson)
2. **One number or the full picture?** Precision AND recall? Variance across
   seeds ([Lesson 11](../11-reproducibility/lesson.md))?
3. **Could the model have seen the test set?** What decontamination was done?
4. **Is the metric still meaningful, or Goodharted/saturated?**
5. **Who judged, and what are the judge's biases?**
6. **Do the vibes agree?** If the demo feels worse than the table looks, someone
   is measuring the wrong thing.

---

## Code for this lesson

See [index.ts](index.ts) — three demos:

1. precision / recall / F1 computed from a confusion matrix,
2. the 95/5 imbalanced dataset where a useless classifier beats a useful one on
   accuracy (and F1 tells the truth),
3. perplexity of two toy character-level language models on a held-out string —
   picking the genuinely better model.

Run it:
```
npx ts-node index.ts
```

## What's next

Your eval says model A beats model B by 1.3 points. You rerun training with a
different random seed and B wins. Which result is real? That question — and the
discipline it forces — is next.
[Lesson 11 → Reproducibility](../11-reproducibility/lesson.md)

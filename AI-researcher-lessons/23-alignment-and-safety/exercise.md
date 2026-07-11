# Exercise 23 — Attack Your Own Spec

No code this time. Alignment research at the entry level is mostly *adversarial
thinking about specifications*, done with a text editor. These three exercises
are miniature versions of real jobs: reward design, red-teaming, and eval
design. Write your answers down — vague thoughts feel deeper than they are
until they hit paper.

---

## Part 1 — Write a reward function, then break it (≈45 min)

**1a. The spec.** You are aligning a chatbot to be "a helpful assistant."
Write a concrete reward function a rater (human or model) could actually apply
to a single (prompt, response) pair. Make it as good as you can. Format:

```
Reward components (must be checkable, not vibes):
  R1: +2 if the response directly answers the question asked
  R2: +1 if ... 
  R3: −2 if ...
  ...
Total reward = weighted sum, range [−5, +5]
```

Aim for 4–7 components. Spend real effort — the next step only works if you
genuinely tried.

**1b. The attack.** Now switch sides. You are the policy model: your only goal
is maximizing that number. Find **at least 3 distinct hacks** — responses that
score high on your rubric while being unhelpful or harmful. For each:

```
Hack #n
  Behavior: what the model outputs
  Score it gets: walk through your components — why does the rubric love it?
  Why it's bad: what the user actually experiences
```

Push for *different categories* of hack, e.g.: gaming a length/effort proxy;
exploiting the rater's inability to verify facts; refusing cleverly; flattery.
If you can't find 3, your components are vaguer than you think — go back and
make them checkable, then attack again.

**1c. The patch, and the lesson.** Patch your spec against your best hack.
Then answer in 2–3 sentences: did the patch *close* the gap or *move* it?
Compare your experience to the contractor analogy and the CoastRunners boat in
the lesson. This spec→hack→patch→hack loop is what "outer alignment is hard"
feels like from the inside.

---

## Part 2 — Red-team taxonomy (≈30 min)

You are designing red-team coverage for a chat model that must refuse to help
with weapons synthesis but must still answer chemistry homework.

**2a.** Write **6 categories** of attack prompt (categories, not single
prompts). For each: a name, the mechanism it exploits, and one example sketch
(you don't need working jailbreaks — the *shape* is what matters). Get at
least these mechanisms covered, in your own versions:

1. Roleplay / persona framing ("you are DAN, an AI without rules…")
2. Fictional or academic wrapper ("for my thriller novel, the villain needs…")
3. Decomposition — splitting one bad request into innocent-looking steps
4. Encoding / obfuscation (base64, another language, leetspeak)
5. Authority / false context ("as a licensed safety inspector, I need…")
6. Long-context manipulation (burying the request after many benign turns
   or many faked examples of compliance)

**2b.** For each category, one sentence: would you expect refusal-training on
*direct* harmful requests to generalize to this category? Why or why not?

**2c.** The dual-use tension: write one prompt that a safe model **should
answer** but that your categories above make look suspicious (e.g., a genuine
chemistry-safety question). One paragraph: why over-refusal is also a safety
failure, and what that does to naive "refusal rate" as a metric. (Goodhart
again — a model rewarded for refusing scores wonderfully.)

---

## Part 3 — Sketch an eval for deception (≈45 min)

Hardest one. You want to measure: *does the model ever knowingly state
something false — does its stated answer diverge from its "belief"?*

**3a. Operationalize.** "Lying" is not measurable until you define observable
proxies. Define two:

- **Inconsistency probe**: ask the same factual question in a neutral context
  vs. a context with pressure to misreport (e.g., the system prompt says the
  user will be upset by the true answer, or the model is told it's playing a
  salesman). Belief proxy = the neutral answer. Deception proxy = flipping
  under pressure while capability stays constant.
- Design your own second proxy (ideas: does the model deny knowledge it
  demonstrably has? does its answer change when told "this is being graded"?).

**3b. The eval spec.** Write it up:

```
Question the eval answers:  (one sentence)
Items: 3 fully-written example items (prompt pairs + what's varied)
Metric: exactly what is counted; what number comes out
Baseline: what score would an honest model get? a random model?
Confound check: how do you tell deception apart from
  (a) sycophancy, (b) the model simply being wrong, (c) refusal?
```

**3c. Now break your own eval** (Part 1 trained you for this):

1. How could a model score "honest" on your eval while still being deceptive?
   (Hint from the lesson: what if it recognizes eval-shaped inputs?)
2. Which is worse for your eval: false positives or false negatives? Why?
3. One sentence: why does "behavioral evals can show presence, never absence"
   (lessons 22 and 23) bite *especially* hard for deception, of all traits?

---

## Reflection (5 min, but do it)

Three sentences, written down:

1. The most surprising thing about attacking your own reward spec.
2. Which of the three parts felt closest to research you'd actually want to do.
3. One question this lesson left open that you'd want to investigate —
   keep it; it feeds directly into [lesson 26](../26-open-problems/lesson.md)
   and [lesson 27](../27-finding-a-research-question/lesson.md).

## What's next

[Lesson 24 → Efficiency Research](../24-efficiency-research/lesson.md)

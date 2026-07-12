# Exercise 26 — From Map to Target

The lesson gave you a map of open problems. This exercise walks you from
"interesting map" to "one question I could actually start on Monday." The
output of Part 3 feeds directly into
[lesson 27](../27-finding-a-research-question/lesson.md) — do not skip it.

Total time: ~3 hours, spread over a few days is fine. Everything is paper,
browser, and honesty. No GPU.

---

## Part 1 — Two problems, two papers, two triples (≈90 min)

**1a.** Pick **two** problems from the lesson's map — one that excites you,
and one from a *different* root cause (check the "what the problems have in
common" section: don't pick two Goodhart problems).

**1b.** For each, find **one paper from the last ~2 years** that attacks it.
Use your [lesson 13](../13-navigating-arxiv/lesson.md) skills: arXiv search, Papers With Code, citation
chasing from the surveys you know. Starting keywords if stuck:

| Problem | Try searching |
|---|---|
| reasoning | "process reward model", "chain-of-thought faithfulness" |
| hallucination | "abstention", "LLM calibration", "knowledge boundary" |
| long context | "lost in the middle", "RULER benchmark", "effective context" |
| forgetting | "catastrophic forgetting LLM", "model editing", "MEMIT" |
| sample efficiency | "BabyLM", "data curriculum", "sample-efficient pretraining" |
| agents | "agent benchmark", "task horizon", "web agent evaluation" |
| synthetic data | "model collapse", "self-improvement LLM", "verifier" |
| evaluation | "benchmark contamination", "LLM-as-judge bias" |
| energy | "inference energy", "carbon footprint LLM", "joules per token" |

**1c.** For each paper, read it at [lesson 12](../12-how-to-read-a-paper/lesson.md) pass-2 depth and write the
**problem–attempt–gap triple**:

```
PAPER: title, authors, year, arXiv ID

PROBLEM  (2–3 sentences): the specific sub-problem attacked — not
  "hallucination" but e.g. "models don't abstain even when internal
  representations distinguish known from unknown facts."

ATTEMPT  (3–4 sentences): the method, in your own words, plus the
  headline result WITH its number ("reduces X by 31% on Y").

GAP      (2–3 sentences): what it does NOT solve. Look in: the limitations
  section, the evaluation's blind spots (what did they NOT test?), scale
  assumptions, and anything the abstract promises that the tables don't
  deliver ([lesson 16](../16-critiquing-a-paper/lesson.md) skills).
```

The GAP is the valuable part. Gaps are where new research questions live.

---

## Part 2 — Five questions you could run on your own machine (≈45 min)

Brainstorm **five research questions** you could plausibly investigate with:
your laptop, free-tier or cheap API access, small open models (≤7B), and the
skills you already have from both tracks (you've built GPTs, run ablations,
done significance tests).

Rules for a valid entry:

1. It is a **question**, with a question mark, not a topic. ("Long context" is
   a topic. "Does accuracy on multi-hop questions drop when the second hop's
   evidence sits in the middle third of a 32K context?" is a question.)
2. You can name the **measurement** that answers it (what number, computed how).
3. You can name the **comparison** ([lesson 18](../18-baselines-and-ablations/lesson.md): result = number + baseline).
4. A first result is reachable in **≤ 2 weeks of evenings**.
5. At least two of the five must attach to a GAP you wrote in Part 1.

Format for each:

```
Q#: the question
  Measures:   what you'd compute
  Baseline:   what you'd compare against
  Resources:  model(s), data, rough runtime
  Risk:       the most likely way this dies (can't get data? effect too
              small to detect with your n? already done?)
```

Struggling to get five? Reliable generators: "X is claimed for big models —
does it hold at 1B?", "benchmark Y — is it contaminated / label-noisy?",
"technique Z helps task A — does it hurt task B?" (forgetting!), "everyone
assumes W — has anyone measured W?"

---

## Part 3 — Commit (≈20 min, the part that matters)

**3a.** Score each of your five questions 1–5 on:

- **Feasibility** — could *you*, with *your* machine and time, finish it?
- **Novelty** — 15 minutes of searching finds no paper already doing exactly this (do the 15 minutes; log your search terms)
- **Interest** — would you still care in week 3, when everything is broken?

**3b.** Pick the winner. Feasibility is the tiebreaker — a finished small
answer beats an abandoned ambitious one, every time. First projects die of
scope, not of smallness.

**3c.** Write the commitment page — one page, exactly this:

```
MY QUESTION: (one sentence, question mark)
WHY IT MATTERS: (2–3 sentences: which open problem, which gap)
FIRST EXPERIMENT: (what you would run in the FIRST WEEK — be concrete:
  model, data, metric, baseline)
WHAT WOULD CHANGE MY MIND: (what result would make you drop or flip
  your hypothesis — decide this NOW, before you've seen any numbers,
  while you're still honest)
```

Save it as `my-question.md` next to this exercise. Lesson 27 starts from this
page: you'll stress-test the question, scope it, and turn it into a real
mini-project ([lesson 30](../30-capstone-research-project/lesson.md) is where it ships).

---

## Sanity checklist before moving on

- [ ] Two triples written, each GAP specific enough that a stranger could invent a follow-up project from it
- [ ] Five questions, every one with a measurement, a baseline, and a risk
- [ ] 15-minute novelty search done for the winner, search terms logged
- [ ] `my-question.md` exists and the FIRST EXPERIMENT names a model, a dataset, a metric, and a baseline

## What's next

[Lesson 27 → Finding a Research Question](../27-finding-a-research-question/lesson.md)

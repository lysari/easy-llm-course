# Lesson 13 — Navigating arXiv & the Research World

---

## The problem: 500 new ML papers. Per day.

You've learned *how* to read a paper (Lesson 12). Now: *which* paper? arXiv receives on
the order of hundreds of new machine learning submissions **every day**. If you read one
per day, at the end of a year you'd be roughly 180,000 papers behind — and further behind
than when you started.

Nobody reads everything. Not professors, not lab leads, not the people whose job title is
literally "researcher". Everyone operates a **filter**, and the quality of your filter
matters more than your reading speed.

This lesson is a map of the territory: what arXiv actually is, where peer review happens,
how ideas *really* spread in the 2020s, the tools that act as filters, and how to trace a
paper's family tree.

---

## What arXiv is (and is not)

**arXiv** (pronounced "archive" — the X is a Greek chi) is a free repository of paper
*preprints*: papers uploaded directly by their authors, **before or without peer review**.

That last part is the key fact. What "not peer-reviewed" means in practice:

- Nobody checked the math, the experiments, or even the honesty before it went online.
  (There's a light moderation check that it looks like a paper in the right category —
  that's all.)
- Brilliant, field-defining work appears on arXiv months or years before formal
  publication. Some of the most cited papers in ML were *never* peer-reviewed at all
  (GPT-2's report, GPT-3-era tech reports, many industry papers).
- Total nonsense also appears on arXiv, formatted in beautiful LaTeX, indistinguishable
  at a glance from the brilliant work.

So arXiv gives you **speed without a quality signal**. ML moves so fast that waiting for
peer review means being ~6–12 months behind, so the field's norm is: post on arXiv the
moment work is done, submit to a conference in parallel. You must supply your own quality
filter — that's what Lessons 12 and 16 train.

### arXiv IDs

Every paper gets a permanent ID: `YYMM.NNNNN`.

```
arXiv:1706.03762
      │ │  │
      │ │  └── 3762nd submission that month
      │ └───── June
      └─────── 2017
                → "Attention Is All You Need", June 2017
```

The ID is the paper's true name. Titles get abbreviated, authors get forgotten, but
`1706.03762` always resolves at `https://arxiv.org/abs/1706.03762`. Versions are suffixed
(`v1`, `v2`, …) — authors can and do revise; if a claim mysteriously disappeared, compare
versions.

### Categories

Papers are tagged with subject categories. The ones you'll live in:

| Category | Name | What's there |
|----------|------|--------------|
| `cs.CL`  | Computation and Language | NLP, LLMs, tokenization, evaluation of language models |
| `cs.LG`  | Machine Learning | training methods, architectures, theory-flavored ML |
| `cs.AI`  | Artificial Intelligence | agents, reasoning, broader AI (a grab-bag) |
| `cs.CV`  | Computer Vision | images, video, multimodal |
| `stat.ML`| Machine Learning (statistics) | the same field, statistics-flavored |

Most LLM papers are cross-listed `cs.CL` + `cs.LG`. Browsing `https://arxiv.org/list/cs.CL/recent`
shows you the raw firehose — visit once to feel its pressure, then never browse raw again.

---

## Where peer review happens: the venues

Peer review in ML happens at **conferences**, not journals (unusual among sciences —
journals are too slow for this field). The ones that matter for you:

```
GENERAL ML                    NLP / LANGUAGE
──────────────                ────────────────
NeurIPS  (December)           ACL     (summer)
ICML     (July)               EMNLP   (fall)
ICLR     (spring)             NAACL   (varies)
```

- **NeurIPS, ICML, ICLR** — the big-three general ML venues. A paper accepted at any of
  these has survived review by ~3–4 researchers. ICLR is notable for **open review**:
  every submission's reviews, scores, and author rebuttals are public at
  openreview.net — reading a few review threads there is a free masterclass (you'll use
  this in Lesson 16).
- **ACL, EMNLP** — the top venues for NLP/language work specifically.

### The conference cycle

Each conference has a yearly deadline; the field's work rhythm orbits these:

```
write paper → post to arXiv → submit to conference → reviews (2–3 months)
     ↑                                                    │
     │                              rebuttal → accept (~20–30%) or reject
     │                                                    │
     └──────────── revise and resubmit to the next one ◄──┘ 
```

Practical consequences for you as a reader:
- The arXiv version usually appears **6–12 months before** the "official" version.
- A paper "under review" is anonymized on OpenReview but often findable on arXiv —
  an open secret.
- Acceptance is noisy. The famous NeurIPS 2014 experiment sent papers to two independent
  committees; they disagreed on a large fraction of accepts. Venue acceptance is *a*
  signal, not *the* signal.

### What acceptance actually tells you

- Accepted at NeurIPS/ICML/ICLR/ACL/EMNLP → several experts found it plausible and
  interesting. Errors still slip through routinely.
- arXiv-only → tells you nothing either way. Chinchilla and the GPT reports were
  arXiv/tech-report only. Judge on content.
- "Workshop paper" → lighter review, early-stage ideas. Good for spotting trends early.

---

## How ideas actually spread today

The official pipeline (arXiv → conference → citation) is slow. The *real* pipeline in
modern ML looks like this:

```
  day 0    arXiv post
  day 0-1  X/Twitter threads — authors + early readers summarize, hype, or debunk
  week 1   blog posts, YouTube explainers, discussion on HN/Reddit/Discord
  week 2-8 implementation repos appear on GitHub ("annotated X", "minimal X",
           "X in 400 lines"), Papers With Code links them
  month 2+ follow-up papers citing it appear on arXiv
  month 6+ the conference version is "published" — by then old news
```

Things worth knowing about this pipeline:

- **The commentary layer is a filter, not truth.** A paper going viral on X means it's
  *interesting or well-marketed*, not correct. Several high-profile "breakthroughs"
  were debunked within days — often *in the replies*. Read the skeptical replies; the
  best critique of a paper is frequently a tweet from someone who tried to reproduce it.
- **Implementation repos are the field's real peer review.** When three independent
  groups reimplement a method and get the claimed numbers, that's stronger evidence than
  conference acceptance. When nobody can, that tells you something too (Lesson 15).
- **Blogs matter.** Many important ideas debut or become understandable via blogs
  (distill.pub historically, individual researchers' blogs, lab blogs). Some influential
  work never becomes a paper at all.

---

## Your toolkit: filters for the firehose

You cannot read everything, so you subscribe to filters. The current standard set:

| Tool | What it does | Use it for |
|------|--------------|------------|
| **Hugging Face Papers** (huggingface.co/papers) | daily community-upvoted list of new ML papers | your daily 2-minute scan — the closest thing to a front page of ML |
| **Papers With Code** | links papers ↔ code ↔ benchmark leaderboards | "has anyone implemented this?" and "what's SOTA on X?" |
| **Semantic Scholar** | search engine + citation graph + TL;DRs | tracing lineage (below), finding related work |
| **arxiv-sanity** (arxiv-sanity-lite) | Karpathy's personal-recommender over arXiv | training a filter on *your* taste — rate papers, get similar ones |
| **alphaXiv** | public discussion/comments layered on arXiv papers | seeing questions and author answers on a specific paper |
| **Google Scholar** | citation counts + alerts | "cited by" hopping; email alerts on authors you follow |
| **OpenReview** | public reviews for ICLR and others | reading real peer reviews (Lesson 16) |

A sane beginner setup — total cost ~15 minutes a day:

1. **Daily (2 min):** scan Hugging Face Papers titles. Click nothing unless it pulls.
2. **Weekly (10 min):** pick 2–3 papers from the week, run pass 1 on each (Lesson 12).
3. **Follow 5–10 researchers** whose work you admire (X, blogs, or Scholar alerts) —
   people are a better filter than keywords.
4. **When a paper matters to you:** check Papers With Code for an implementation, and
   alphaXiv/OpenReview for discussion, *before* investing in a pass 3.

Filter principle: **optimize for recall on what matters to you, not coverage of
everything.** You care about, say, tokenization and small-model training? Missing a
diffusion-model breakthrough for three weeks costs you nothing. The important papers in
your own area will reach you through several channels at once — that redundancy is the
signal.

---

## Tracing a paper's lineage

Every paper is a node in a citation graph. Reading one paper properly means glancing at
its neighborhood:

```
            BACKWARD (what it builds on)
   Bahdanau 2014        Sutskever 2014        Kim 2016
   (attention)          (seq2seq)             (convs for text)
        ╲                    │                    ╱
         ╲                   │                   ╱
          ▼                  ▼                  ▼
        ┌─────────────────────────────────────────┐
        │   "Attention Is All You Need" (2017)    │
        └─────────────────────────────────────────┘
          ╱                  │                  ╲
         ▼                   ▼                   ▼
   BERT (2018)          GPT-1 (2018)         ViT (2020)…
            FORWARD (what built on it)
```

**Backward tracing — "what does this build on?"**
Look at which references appear in the *method* section (not related work — those are
courtesy citations). The 2–5 references the method actually depends on are the paper's
true parents. When a paper is over your head, its parents are your prerequisite list.

**Forward tracing — "what happened next?"**
On Semantic Scholar or Google Scholar, open "cited by" and sort by citations. The top
forward-citations tell you: Did the idea survive? What replaced or fixed it? Was it
debunked? A 2017 paper's forward trace *is* the field's verdict on it.

Forward tracing is also the best way to find out that the paper you're excited about was
improved, simplified, or refuted two years ago — *before* you spend a month building
on the original.

---

## The mindset: gardener, not completionist

The firehose induces two failure modes in beginners:

1. **FOMO paralysis** — trying to keep up with everything, retaining nothing, building
   nothing. (Reading becomes a substitute for doing.)
2. **Hermit mode** — reading nothing, reinventing wheels that were published, evaluated,
   and superseded years ago.

The working posture is in between: a small, well-tended garden. A handful of daily-scan
filters, one focus area you go *deep* on, a notes folder that grows (Lesson 12's habit),
and the citation graph whenever you need context. Researchers with 20 years of experience
operate exactly this way — they just have better-trained taste on which titles to click.

That taste is trainable. Every pass-1 read tunes it.

---

## Exercise for this lesson

See [exercise.md](exercise.md) — find one recent paper in an area you care about, trace
its lineage, find its implementation, and draw its family tree in 5 lines.

## What's next
[Lesson 14 → Landmark Papers Reading List](../14-landmark-papers/lesson.md)

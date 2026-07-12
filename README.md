# LLM From Scratch

Learn how large language models work by building every piece yourself in TypeScript — no ML frameworks, no black boxes.

---

## Three ways to learn

This course has three tracks. Take the first two in either order (or together); the researcher track runs alongside them.

### Understanding AI — top down
Start from what AI produces (meaningful text) and strip back every layer until nothing is left but floating-point math.

```
"AI understands language"
      ↓  text is just integers
      ↓  integers are just coordinates
      ↓  relationships are just dot products
      ↓  "knowing" is just highest probability
      ↓  "learning" is just gradient descent
relu(Wx + b) all the way down
```

Good if you want to know **what a language model actually is** — and isn't.
→ Start at [REVERSE-LESSON.md](REVERSE-LESSON.md)

---

### Normal lessons — bottom up
Start from zero and build toward GPT-4.

```
Numbers → Neural Nets → Words → Attention → GPT-1 → GPT-2 → GPT-3 → GPT-4
  00–03      04–07       08–11     12–13      14–17    18–21    22–25    26–30
```

Good if you want to know **how to build** a language model.
→ Start at [LESSON.md](LESSON.md)

---

### AI Researcher track — beyond building
Building models is engineering. This track teaches the other half: how researchers think and work.

```
Math foundations → DL science → Papers → Experiments → Frontiers → Your own research
     00–05           06–11      12–16      17–21        22–26          27–30
```

Good if you want to **become an AI researcher** — read papers, run real experiments, and produce your own research. Designed to interleave with the normal lessons (the roadmap tells you when).
→ Start at [AI-researcher-lessons/README.md](AI-researcher-lessons/README.md)

---

## Prerequisites

No ML background needed. You do need:

- Basic programming (variables, functions, loops)
- Comfortable reading TypeScript (or JavaScript)
- High school math — the course re-teaches anything beyond that

---

## Setup

```bash
npm install
```

Run any lesson:

```bash
npx ts-node lessons/00-what-is-a-model/index.ts
npx ts-node reverse-lessons/RL-00-the-illusion/demo.ts
```

---

## How the "Understanding AI" lessons work

Each folder under `reverse-lessons/` contains:

- `lesson.md` — starts with what looks like understanding, peels back one layer
- `demo.ts` — proves the point with runnable code

| Lesson | What gets stripped |
|--------|--------------------|
| RL-00: The Illusion | The assumption that AI understands |
| RL-01: Text Is Numbers | Words become integer IDs — meaning gone |
| RL-02: Meaning Is Position | IDs become coordinate vectors — not definitions |
| RL-03: Attention Is Arithmetic | "Relationships" are dot products and weighted sums |
| RL-04: Prediction, Not Knowing | "Knowing" is the highest-probability token |
| RL-05: Weights, Not Wisdom | "Learning" is gradient descent minimizing a number |
| RL-06: Ground Zero | Everything is `relu(Wx + b)` repeated |

---

## How the normal lessons work

Each folder under `lessons/` contains:

- `lesson.md` — the concept, the math, worked examples
- `index.ts` — runnable TypeScript that implements it from scratch

Read the lesson, then run (and edit) the code.

| Phase | Lessons | What you build |
|-------|---------|----------------|
| Foundations | 00–07 | Linear regression → neural network → backprop |
| Words | 08–11 | Tokenization → embeddings → next-token prediction |
| Attention | 12–13 | Self-attention → transformer block |
| GPT-1 | 14–17 | Tiny GPT → FFN → Adam optimizer |
| GPT-2 | 18–21 | BPE → multi-head attention → KV cache |
| GPT-3 | 22–25 | Scaling laws → sampling → in-context learning |
| GPT-4 | 26–30 | Fine-tuning → RLHF → mixture of experts |

---

## Suggested paths

**I want to build a GPT from scratch:**
→ Normal lessons 00–14, in order. Each one depends on the previous.

**I want to understand AI critically (non-technical):**
→ Understanding AI lessons only. RL-00 → RL-06. No prior ML needed.

**I'm technical and want both perspectives:**
→ Read each Understanding AI lesson first (top-down intuition), then the matching normal lesson (bottom-up implementation).

Rough pairings:

| Understanding AI lesson | Normal lessons |
|----------------|----------------|
| RL-01 (Text Is Numbers) | 09 (Tokenization), 18 (BPE) |
| RL-02 (Meaning Is Position) | 10 (Embeddings) |
| RL-03 (Attention Is Arithmetic) | 12 (Attention), 19 (Multi-Head Attention) |
| RL-04 (Prediction, Not Knowing) | 11 (Softmax), 23 (Sampling Strategies) |
| RL-05 (Weights, Not Wisdom) | 03 (Gradient Descent), 07 (Backpropagation) |
| RL-06 (Ground Zero) | 14 (Tiny GPT) |

---

## Project structure

```
LLM/
├── README.md               ← you are here
├── LESSON.md               ← normal lessons progress tracker
├── REVERSE-LESSON.md       ← Understanding AI lessons index
├── lessons/
│   ├── 00-what-is-a-model/
│   │   ├── lesson.md
│   │   └── index.ts
│   ├── 01-linear-regression/
│   └── ...
├── reverse-lessons/
│   ├── RL-00-the-illusion/
│   │   ├── lesson.md
│   │   └── demo.ts
│   ├── RL-01-text-is-numbers/
│   └── ...
└── AI-researcher-lessons/
    ├── README.md           ← researcher track roadmap
    ├── 00-what-ai-research-is/
    │   └── lesson.md
    ├── 01-linear-algebra/
    │   ├── lesson.md
    │   └── index.ts        (some lessons have exercise.md instead)
    └── ...
```

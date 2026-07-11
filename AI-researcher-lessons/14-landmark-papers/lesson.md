# Lesson 14 — Landmark Papers: the Story of Deep Learning in 15 Papers

---

## The problem: the literature has no front door

You know how to read a paper (Lesson 12) and where papers live (Lesson 13). But the
field is a million papers deep. Where do you *start*?

Here's the thing beginners miss: the important papers are not a pile — they are a
**story**. Each landmark answered a question the previous one raised. Read in order,
they form a narrative with characters, plot twists, and a running argument about what
intelligence-from-data even means. Read out of order, they're just PDFs.

This lesson is that story, told through 15 papers. For each: what it showed, why it
mattered, and what to actually read (many are skimmable — knowing *what to skip* is half
the value of a guide).

**The core 8** — marked ★ — are the ones a beginner should genuinely read (pass 2,
Lesson 12 style). The other seven you should know *about*; pass 1 is enough.

```
1986 ──── backprop ──────────────────┐  "networks can learn"
1998 ──── LeNet                      │  "…and see digits"
2012 ──── AlexNet ★ ─────────────────┤  THE BIG BANG
2013 ──── word2vec ★                 │  "words are vectors"
2014 ──── seq2seq                    │  "sequences in, sequences out"
2014 ──── attention (Bahdanau) ★     │  "stop cramming into one vector"
2017 ──── Transformer ★ ─────────────┤  THE ARCHITECTURE
2018 ──── BERT / GPT-1               │  two paths diverge
2019 ──── GPT-2 ★                    │  "one model, many tasks, no finetuning"
2020 ──── scaling laws (Kaplan) ★    │  "loss is predictable"
2020 ──── GPT-3 ★                    │  "prompting is programming"
2022 ──── Chinchilla                 │  "we sized models wrong"
2022 ──── InstructGPT ★              │  "align it with human feedback"
2023 ──── LLaMA                      │  open weights change everything
2023 ──── GPT-4 report               │  capabilities + the eval era
```

---

## Act I — Learning is possible (1986–2012)

### 1. Backpropagation — Rumelhart, Hinton & Williams, 1986
*"Learning representations by back-propagating errors" (Nature)*

**What it showed:** multi-layer neural networks can be trained by propagating the error
gradient backward through the layers — and the hidden layers learn useful internal
representations on their own.

**Why it mattered:** before this, nobody had a practical recipe for training networks
with hidden layers; the field had largely written them off. Backprop is still, today,
how every model in this curriculum learns. You implemented it by hand in
[../../lessons/07-backpropagation/lesson.md](../../lessons/07-backpropagation/lesson.md) —
which means you understand this paper better than most people who cite it.

**What to read:** skim. It's 4 pages; read it for the historical voice ("we describe a
new learning procedure…") and Figure 1. The math you already own. **Skimmable.**

### 2. LeNet — LeCun et al., 1998
*"Gradient-based learning applied to document recognition"*

**What it showed:** a convolutional network (convolution + pooling + backprop) reading
handwritten digits well enough to be deployed on real bank checks.

**Why it mattered:** proof that gradient-trained networks could do real, deployed
perception — a decade before anyone believed it scaled. Also the origin of MNIST, the
fruit fly of ML. The architecture pattern (local features composed into global
structure) prefigures everything.

**What to read:** Sections 1–2 and the architecture figure. It's 46 pages — do not read
it all; the rest is a document-processing system of purely historical interest.
**Skimmable.**

### 3. ★ AlexNet — Krizhevsky, Sutskever & Hinton, 2012
*"ImageNet Classification with Deep Convolutional Neural Networks"*

**What it showed:** a big conv net, trained on two consumer GPUs, crushed the ImageNet
competition — 15.3% top-5 error vs 26.2% for the best non-neural method. Not an edge:
a chasm.

**Why it mattered:** this is the big bang of modern deep learning. Overnight, the recipe
became clear: *big model + big data + GPUs + a few tricks (ReLU, dropout, augmentation)*.
Every hyperscale training run today is this paper's descendant. It also set the template
for the modern ML paper: architecture diagram, benchmark table, ablations.

**What to read for:** the *recipe-ness* of it. Notice how little theory there is — it's
engineering choices, each justified by "this made the error go down". That empirical
style IS modern ML research. Read fully; it's short and clear.

---

## Act II — Language becomes learnable (2013–2016)

### 4. ★ word2vec — Mikolov et al., 2013
*"Efficient Estimation of Word Representations in Vector Space"*

**What it showed:** train a tiny model to predict nearby words, and the learned vectors
encode meaning as geometry — famously `king − man + woman ≈ queen`.

**Why it mattered:** the first mass demonstration that *meaning is learnable as
structure in a vector space* from raw text, no labels. Every embedding you built in
[../../lessons/10-embeddings/lesson.md](../../lessons/10-embeddings/lesson.md) descends
from this. It converted a generation of skeptics to the idea that "predict the context"
is a path to semantics — the LLM thesis in miniature.

**What to read for:** the analogy results (Table 8 region), and the sheer smallness of
the models. Meaning didn't need a giant model — it needed the right objective.

### 5. seq2seq — Sutskever, Vinyals & Le, 2014
*"Sequence to Sequence Learning with Neural Networks"*

**What it showed:** two LSTMs — one encoding the input sentence into a vector, one
decoding the output sentence from it — can do machine translation end-to-end.

**Why it mattered:** established the encoder-decoder pattern and the framing "everything
is sequence-in, sequence-out" that still underlies chat models (your prompt is the
source sequence; the reply is the target). Also introduced tricks (reversing the input!)
that reveal how fragile the one-vector bottleneck was — which sets up the next paper.

**What to read:** abstract, model figure, and Section 3.3's input-reversal hack. The
LSTM machinery is obsolete; skim it. **Skimmable.**

### 6. ★ Attention — Bahdanau, Cho & Bengio, 2014
*"Neural Machine Translation by Jointly Learning to Align and Translate"*

**What it showed:** instead of forcing the whole input sentence through one fixed
vector, let the decoder *look back* at all encoder states and learn where to look —
attention. Translation quality on long sentences stopped collapsing.

**Why it mattered:** this is attention's true birthplace, three years before the
Transformer. The diagnosis (a bottleneck) and the cure (learned, content-based lookup)
is one of the cleanest problem→mechanism stories in the field — and a model for how to
motivate an architecture change in a paper.

**What to read for:** Figure 3's alignment heatmaps — you can *see* the model learning
soft word alignments nobody programmed. Compare its additive attention with the dot
-product version you built in
[../../lessons/12-attention/lesson.md](../../lessons/12-attention/lesson.md).

---

## Act III — The architecture (2017–2019)

### 7. ★ Transformer — Vaswani et al., 2017
*"Attention Is All You Need"*

**What it showed:** delete the recurrence entirely; keep only attention + feed-forward
blocks. Better translation quality, and — the deeper point — every token processed in
parallel, so training scales with hardware.

**Why it mattered:** this is the architecture of essentially everything since: BERT,
every GPT, LLaMA, vision transformers, protein models. Arguably the most consequential
ML paper of its decade. You did pass 1 and 2 on it in Lesson 12, and you *built it* in
[../../lessons/13-transformer-block/lesson.md](../../lessons/13-transformer-block/lesson.md)
and [../../lessons/14-tiny-gpt/lesson.md](../../lessons/14-tiny-gpt/lesson.md).

**What to read for:** on this pass, read it as a *scaling* paper: the parallelism
argument (Table 1's path-length and complexity comparison) is the real thesis. The BLEU
scores are the least important numbers in it, in hindsight.

### 8. BERT and GPT-1 — Devlin et al. 2018 / Radford et al. 2018
*"BERT: Pre-training of Deep Bidirectional Transformers…" / "Improving Language
Understanding by Generative Pre-Training"*

**What they showed:** take the Transformer, pretrain it on raw text at scale, then
finetune on each downstream task. Two paths: BERT uses the *encoder* (sees both
directions, trained by masking words) and dominated understanding benchmarks; GPT-1 uses
the *decoder* (left-to-right, trained by next-token prediction) and could also generate.

**Why they mattered:** together they established **pretrain-then-finetune** as the
paradigm — stop training task models from scratch; adapt one big pretrained model.
For a few years BERT looked like the winner (it swept the benchmarks). The plot twist:
GPT's humbler objective — just predict the next token — is the one that scaled into
everything that followed. Worth sitting with: the "losing" branch won.

**What to read:** BERT's Figure 1 (the two pretraining tasks) and GPT-1's Section 3.
Skim both — the ideas matter; the details have been superseded. **Skimmable.**

### 9. ★ GPT-2 — Radford et al., 2019
*"Language Models are Unsupervised Multitask Learners"*

**What it showed:** scale GPT to 1.5B parameters on 40GB of curated web text, and it
performs tasks — translation, summarization, QA — **zero-shot**, with no finetuning,
just by conditioning on the right text.

**Why it mattered:** the conceptual pivot of the modern era: *tasks are just patterns
in text, and a good enough next-token predictor absorbs them all*. Also historically
notable for the staged release over misuse concerns — the beginning of "model release
policy" as a topic. You rebuilt its architecture in
[../../lessons/20-gpt2-architecture/lesson.md](../../lessons/20-gpt2-architecture/lesson.md).

**What to read for:** Section 3's zero-shot framing and the results tables — note how
*mediocre* many zero-shot numbers are in absolute terms. The claim wasn't "it's great at
translation"; it was "nobody trained it to translate *at all*". Reading claims at the
right altitude is a Lesson 16 skill; practice it here.

---

## Act IV — Scale becomes a science (2020–2022)

### 10. ★ Scaling laws — Kaplan et al., 2020
*"Scaling Laws for Neural Language Models"*

**What it showed:** language-model loss falls as a smooth **power law** in model size,
dataset size, and compute — straight lines on log-log plots, holding across seven orders
of magnitude. Performance became *predictable before you train*.

**Why it mattered:** turned "make it bigger" from a hunch into an equation, and directly
justified spending millions on GPT-3. It also drew a conclusion — prefer bigger models
over more data — that the field acted on for two years and that turned out to be wrong
(see Chinchilla). A landmark can be both foundational *and* mistaken; hold both.

**What to read for:** Figure 1 — the most influential single figure in modern ML — and
the fitted equations. Full treatment in
[../../lessons/22-scaling-laws/lesson.md](../../lessons/22-scaling-laws/lesson.md), and
you will *reproduce this paper's core claim in miniature* in
[Lesson 15](../15-reproducing-a-paper/lesson.md).

### 11. ★ GPT-3 — Brown et al., 2020
*"Language Models are Few-Shot Learners"*

**What it showed:** at 175B parameters, a language model does new tasks from a few
examples *placed in the prompt* — in-context learning. No gradient updates; the "training
examples" are just text the model reads.

**Why it mattered:** prompting replaced finetuning as the interface to models — the
reason "prompt" is now an everyday word. It's also the cleanest large-scale demo of an
ability *emerging* with scale: GPT-2 could barely do this; GPT-3 could. Companion:
[../../lessons/24-in-context-learning/lesson.md](../../lessons/24-in-context-learning/lesson.md).

**What to read:** it's 75 pages — do not read it linearly. Read Sections 1–2 (the
zero/one/few-shot framing and Figure 1.2), then skim results, then the broader-impacts
section (an early template for what became standard). The aggregate curves matter more
than any single benchmark.

### 12. Chinchilla — Hoffmann et al., 2022
*"Training Compute-Optimal Large Language Models"*

**What it showed:** rerun the scaling-law fits more carefully (varying the learning-rate
schedule with training length) and the optimum moves: for a fixed compute budget, models
should be **much smaller and trained on far more tokens** — roughly 20 tokens per
parameter. Their 70B model beat the 280B Gopher trained on the same compute.

**Why it mattered:** the field had been sizing models wrong for two years. Every serious
post-2022 model follows (or knowingly exceeds — training even longer for cheap inference)
Chinchilla-style budgeting. It's also the great modern example of a *correction paper*:
same question as Kaplan, better methodology, different answer. Science working.

**What to read:** the three-approaches section and Table 3. You did pass 1 on it in
Lesson 12's exercise; a full pass 2 is optional here. **Skimmable — but know its number
(~20 tokens/param) cold.**

---

## Act V — Alignment and the open era (2022–2023)

### 13. ★ InstructGPT — Ouyang et al., 2022
*"Training language models to follow instructions with human feedback"*

**What it showed:** raw GPT-3 completes text; it doesn't *help*. Pipeline: (1) finetune
on human demonstrations, (2) train a reward model on human preference comparisons,
(3) optimize the model against that reward with PPO. Labelers preferred the 1.3B
InstructGPT over the 175B GPT-3 — alignment beat 100× scale on helpfulness.

**Why it mattered:** this is the recipe that turned language models into assistants —
ChatGPT is this paper productized. It made "RLHF" and "alignment" central engineering
terms, and introduced the "alignment tax" framing. You built each stage from scratch in
companion lessons [26](../../lessons/26-instruction-finetuning/lesson.md),
[27](../../lessons/27-reward-model/lesson.md), and
[28 — RLHF/PPO](../../lessons/28-rlhf-ppo/lesson.md).

**What to read for:** Figure 2 (the three-stage pipeline — the most copied diagram since
the Transformer) and the human-preference win-rate plots. Note what the metric is:
*human preference*, not benchmark accuracy. That shift in what counts as "better" is the
real content.

### 14. LLaMA — Touvron et al., 2023
*"LLaMA: Open and Efficient Foundation Language Models"*

**What it showed:** train Chinchilla-style (and beyond — far more tokens than "optimal")
on only public data, and a 13B model competes with GPT-3 at 175B. Then: release the
weights to researchers.

**Why it mattered:** the weights (which promptly leaked, then were openly released as
LLaMA-2) ignited the open-model ecosystem — llama.cpp, LoRA finetunes, local inference,
and thousands of derivative models. It moved frontier-adjacent research from five labs
to anyone with a laptop. Architecturally it's a curated best-of collection you already
know: RoPE ([../../lessons/25-rope-positional-encoding/lesson.md](../../lessons/25-rope-positional-encoding/lesson.md)),
pre-norm, SwiGLU.

**What to read:** Sections 1–2 (data + architecture choices, each with its citation —
a model of engineering honesty). **Skimmable.**

### 15. GPT-4 — OpenAI, 2023
*"GPT-4 Technical Report"*

**What it showed:** a frontier model evaluated primarily on *human* exams (bar exam,
AP tests) and a broad eval suite; loss-before-training predicted from small-scale runs
(scaling laws, industrialized); and — pointedly — **no** architecture, size, or data
details disclosed.

**Why it mattered:** twice. First, capabilities: it made "the model passes the bar exam"
a normal sentence. Second, sociologically: the closing of frontier research. The report
marks the start of the **eval era** — when the binding research question shifted from
"how do we train better models?" to "how do we even measure what they can do?" (see
[../../lessons/30-gpt4-capstone/lesson.md](../../lessons/30-gpt4-capstone/lesson.md) and
this track's Lesson 10). Read it also as a document *about* research culture: what does
it mean when a landmark "paper" contains no method section?

**What to read:** the eval tables, the predictable-scaling figure, and the system card's
red-teaming examples. **Skimmable** — there is, by design, not much method to read.

---

## The story in one paragraph

Backprop said networks can learn; AlexNet proved learning + scale + GPUs beats
hand-engineering; word2vec showed meaning is geometry learnable from raw text; attention
(Bahdanau) removed the bottleneck; the Transformer made attention the whole architecture
and unlocked parallel scale; GPT-2/GPT-3 showed next-token prediction at scale absorbs
tasks, and prompting replaced finetuning; Kaplan and Chinchilla turned scale into
arithmetic; InstructGPT aligned the raw predictor into an assistant; LLaMA handed the
recipe to everyone; and GPT-4 closed the curtain while opening the era of evaluation.
Fifteen papers, one argument: **general capability = simple objective + scale +
alignment** — and every step was doubted until it worked.

---

## The core 8, and how to use this list

For a beginner: ★ **AlexNet, word2vec, Bahdanau attention, Transformer, GPT-2, Kaplan
scaling laws, GPT-3, InstructGPT.** These eight carry the full narrative; do a real
pass 2 (Lesson 12) on each. The remaining seven: pass 1, know their role in the story.

Rules of engagement:
- **In order.** The order is the point — each paper answers the previous one's question.
- **One per week.** Eight weeks. Racing through them in three days produces nothing
  durable; the exercise gives you the schedule.
- **Re-draw one figure per paper by hand.** Sketching Figure 1 forces you to notice what
  it actually shows. This is the single highest-value note-taking trick known.
- These are period pieces. Some numbers are quaint, some conclusions were overturned —
  you now know which. Reading them *knowing the ending* is precisely what makes the
  reasoning visible.

---

## Exercise for this lesson

See [exercise.md](exercise.md) — the 8-week core-8 reading schedule, with a summary +
hand-drawn-figure log per paper.

## What's next
[Lesson 15 → Reproducing a Paper](../15-reproducing-a-paper/lesson.md)

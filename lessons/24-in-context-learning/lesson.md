# Lesson 24 — In-Context Learning: GPT-3's Surprising Ability

## What Was GPT-3's Big Discovery?

In May 2020, OpenAI published "Language Models are Few-Shot Learners" (Brown et
al., 2020). The paper described GPT-3, a 175-billion-parameter language model.
But the scale itself was not the headline. The headline was what the model could
do with its context window.

The discovery: a large enough language model can perform new tasks it was never
explicitly trained on, just by reading a few examples in the prompt. No gradient
updates. No fine-tuning. No weight changes of any kind. You just write the
examples as text, and the model figures out what you want.

This became known as **in-context learning (ICL)**.

---

## The Three Modes

Brown et al. defined three variants that are now standard vocabulary:

### Zero-shot

You describe the task in natural language and give no examples. The model must
infer what you want from the instruction alone.

```
Translate English to French:
cheese →
```

The model outputs `fromage`. It was never trained to translate — it was trained
to predict the next token. But during pretraining it saw countless English-French
pairs, so it recognizes the pattern from the instruction.

### One-shot

You give exactly one example before the new query.

```
Translate English to French:
sea otter → loutre de mer
cheese →
```

The single example shows the model the expected format (word → word) and
confirms what "translate" means in this context.

### Few-shot

You give 3–10 examples before the new query.

```
Translate English to French:
sea otter → loutre de mer
peppermint → menthe poivrée
plush girafe → girafe en peluche
cheese →
```

More examples let the model pin down the task more precisely — not just "what
kind of task" but "what style of output, what level of formality, what format."

---

## Why Does This Work? The Mechanistic Explanation

At first glance this looks almost magical. The model sees text in its context
and suddenly "knows" how to do a new task without any learning in the
traditional sense. What is actually happening?

### The model has learned meta-patterns during pretraining

During pretraining on hundreds of billions of tokens, the model saw not just
facts and language, but the structure of demonstrations. It saw tutorials,
textbooks, worked examples, Q&A forums, documentation with examples, code
with comments, math solutions with steps. The model did not memorize tasks — it
learned the shape of "showing how something works."

### A context example is just a sequence of tokens

From the model's perspective, there is no fundamental difference between tokens
it was trained on and tokens in the context window. An example like
`sea otter → loutre de mer` is a token sequence. The model processes it the same
way it processes everything else: each token attends to every previous token.

### Attention can detect the input→output pattern

Here is the key insight. When the model processes the query token `cheese`, the
attention mechanism lets it look back at every earlier token. It attends to the
arrow `→` tokens, the example pairs, and the task description. The attention
weights encode "what is relevant to predicting the next token here." Crucially,
the pattern `[word] → [French word]` repeated several times creates a very
strong signal in the attention mechanism. The model recognizes it and uses it.

### No weight updates, but implicit belief updates

This is what makes ICL strange and interesting. The model's weights do not
change. But as tokens accumulate in the context, the attention scores shift.
Each new token changes what the model attends to when it generates the next one.
You can think of this as the model updating its "working hypothesis" about what
task it is doing — not by adjusting parameters, but by adjusting which parts of
its context it looks at. The context itself is the memory.

---

## An Analogy: The Self-Updating Dictionary

Imagine a very large dictionary that has entries for millions of words. You open
to a blank page and write:

```
grumble (verb): to complain quietly
grouse (verb): to complain irritably
kvetch (verb): to complain persistently
gripe (verb): to complain about trivial matters
```

Now you write just the word `bellyache (verb):` and push the dictionary toward
someone who has read every dictionary ever printed. They look at your four
entries, recognize the pattern (synonyms for complaining), and complete the
definition correctly — even if they have never seen that exact word in a
definition before.

That is in-context learning. The dictionary (the model) did not learn anything
new. But the four entries you wrote showed it what you were looking for. It used
pattern recognition across its entire pretraining experience to complete your
entry.

---

## Emergent Abilities: Chain-of-Thought Reasoning

As models scaled toward and past 100 billion parameters, something unexpected
appeared: **emergent abilities** — capabilities that were not present at smaller
scales and were not explicitly trained.

The most famous is **chain-of-thought (CoT) reasoning**, discovered by Wei et
al. (2022).

The finding was almost absurdly simple: if you add the phrase "Let's think step
by step:" to a problem, the model dramatically improves on multi-step reasoning
tasks like arithmetic, logic puzzles, and word problems.

Example without CoT:

```
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each.
   How many tennis balls does he have now?
A: 11
```

Example with CoT:

```
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each.
   How many tennis balls does he have now?
A: Let's think step by step:
   Roger starts with 5 balls.
   He buys 2 cans. Each can has 3 balls. That is 2 × 3 = 6 new balls.
   5 + 6 = 11.
   The answer is 11.
```

The model was not trained to reason step by step on word problems. The chain of
thought emerged from pretraining on text where humans reason step by step (code
comments, math textbooks, worked examples). The phrase "Let's think step by
step:" acts as a trigger — it shifts the model into a mode where it generates
intermediate reasoning tokens, and those tokens condition the final answer.

This is ICL at its deepest: a prompt fragment changing the model's generative
behavior without touching its weights.

---

## Prompt Engineering Basics

Because format is just tokens, how you write the prompt matters enormously.

### Format your examples clearly

```
Q: What is the capital of France?
A: Paris

Q: What is the capital of Japan?
A: Tokyo

Q: What is the capital of Brazil?
A:
```

The `Q:` and `A:` labels tell the model exactly where inputs end and outputs
begin. Raw text like "France — Paris, Japan — Tokyo, Brazil —" is noisier and
performs worse.

### More examples help, up to a point

GPT-3 showed consistent improvement from 0 to 1 to 3 to 10 examples for most
tasks. Beyond 10–20, gains plateau and you run up against the context window
limit. The optimal number depends on the task complexity.

### Order matters

Models are sensitive to the order of examples. Examples closer to the query tend
to have more influence (recency bias in attention). Shuffling examples can change
accuracy by several percentage points. For important tasks, try multiple orderings.

### The system prompt sets the context

Many API interfaces prepend a system prompt before user content. This is just
more tokens — but they appear first, so they influence all subsequent attention.
A system prompt like "You are a precise, formal translator" shifts the model's
behavior for the entire conversation without any examples.

---

## Why Fine-Tuning Is Still Useful

If a large model can perform any task via ICL, why fine-tune at all?

Three reasons:

**Efficiency.** A fine-tuned model does not need examples in the prompt. A
few-shot prompt with 5 examples might be 500 tokens. Running inference on 500
extra tokens for every single query is expensive at scale. A fine-tuned model
needs only the query itself.

**Consistency.** ICL performance varies with example choice, order, and phrasing.
Fine-tuning bakes a behavior into the weights directly, making it robust to
prompt variation. A customer-facing product cannot tolerate random performance
swings based on how a user phrases their request.

**Cost.** Fewer tokens in = cheaper API calls, lower latency, higher throughput.
At millions of queries per day, eliminating 500 tokens per query can mean a
substantial reduction in compute cost.

The practical pattern: prototype with few-shot ICL to validate the task, then
fine-tune once the task is well-defined. ICL is research and iteration; fine-tuning
is production.

---

## Summary

| Concept | What it means |
|---|---|
| In-context learning | Model adapts to a task from examples in the prompt, no weight updates |
| Zero-shot | Task description only, no examples |
| One-shot | One example before the query |
| Few-shot | 3–10 examples before the query |
| Chain-of-thought | Triggering step-by-step reasoning with a prompt phrase |
| Emergent ability | Capability not present at small scale, appears at large scale |
| Prompt engineering | Crafting the token sequence to elicit desired model behavior |

The key insight to carry forward: **the context window is a soft, temporary
memory that reshapes what the model attends to, without changing what the model
knows.** In-context learning is not learning in the weight-update sense — it is
pattern retrieval guided by attention.

---

## What's Next

Lesson 25 introduces RoPE (Rotary Position Encoding), a positional encoding
scheme that extends context length cleanly and is used in most modern models
including LLaMA and Mistral. Longer context windows mean more examples in
context, which means more powerful in-context learning — the two ideas scale
together.

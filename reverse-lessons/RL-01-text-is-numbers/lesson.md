# Reverse Lesson 01 — Text Is Numbers

---

## Layer to peel: words → integers

The previous lesson showed you the illusion. Now we peel back the first layer.

You type: `"Hello, how are you?"`

Before any "thinking" happens, the model does one thing first:

**It converts your text into a list of integers.**

```
"Hello, how are you?"
     ↓
[15496, 11, 703, 389, 345, 30]
```

These integers are called **tokens**. The process of converting text to tokens is called **tokenization**.

After this step, **the model never sees your words again.** It only ever works with numbers.

---

## What is a token?

A token is a chunk of text — usually a word, part of a word, or a punctuation mark. Each token has an ID number in a fixed vocabulary table.

Example (GPT-2 vocabulary):

```
"the"    →  262
"cat"    →  3797
"sat"    →  3332
"on"     →  319
"mat"    →  2603
"."      →  13
```

The model has a vocabulary of ~50,000 tokens. Every possible token has an ID from 0 to 49,999.

---

## Why tokens and not letters?

Tokens are bigger than single characters but smaller than full words. This is a practical engineering choice:

- Letters only: too many steps to build meaning, too slow
- Full words: too many unique words (100,000+), rare words get no data
- Subword tokens: common words are single tokens, rare words split into pieces

```
"unhappiness"  →  ["un", "happiness"]  →  [403, 25763]
"tokenization" →  ["token", "ization"] →  [30001, 1634]
```

The word `"tokenization"` literally does not exist as a single entry. The model sees it as two separate pieces.

---

## The critical point

At the moment of tokenization, **all linguistic meaning is stripped away.**

The model receives: `[262, 3797, 3332, 319, 2603, 13]`

Those six numbers carry no semantic content. They are just positions in a lookup table. The number `3797` doesn't mean "cat." It means "the 3797th row in the vocabulary table."

```
Vocabulary table (simplified):
  Row 0:     "!"
  Row 1:     " "
  ...
  Row 262:   "the"
  ...
  Row 3797:  "cat"    ← this is all "cat" is to the model: row 3797
  ...
  Row 49999: <last token>
```

The model has no idea that cats are animals, that animals are living things, or that living things are different from inanimate objects.

---

## Tokenization is lossy

Tokenization also throws away:

- **Capitalization** (sometimes — depends on the tokenizer)
- **Whitespace details** (tabs, multiple spaces)
- **The original spelling** of rare words (they get split)

```
"GPT-4"   →  ["G", "PT", "-", "4"]  →  [38, 2767, 12, 19]
```

The model sees four numbers when you write "GPT-4". The connection between those four numbers and the concept "a large language model by OpenAI" exists nowhere in the tokenization step.

---

## Tokenization has no understanding

Here is a question that feels simple: **which of these is more positive?**

```
"happy"
"unhappy"
```

A human answers instantly: "happy" is positive, "unhappy" is negative. The prefix "un-" negates.

Now look at what the tokenizer does:

```
"happy"   →  [14774]
"unhappy" →  [403, 14774]    ← "un" + "happy"
```

The model receives two tokens for "unhappy." It does not receive any signal that these tokens negate each other. The relationship between "un" (negation) and "happy" (positive emotion) is something the model has to learn purely from patterns in text — it is not built into the tokenization.

---

## What the model gets as input

After tokenization, your full prompt becomes a list of integers:

```
"The cat sat on the mat" → [464, 3797, 3332, 319, 262, 2603]
```

This is the ONLY thing that enters the model. From this point on, there are no words, no sentences, no grammar rules, no dictionary definitions.

Just six integers.

Everything the model "knows" about language — grammar, facts, style, logic — must emerge from patterns it learned while looking at billions of such integer sequences.

---

## Run the demo

See [demo.ts](demo.ts) — implements a minimal character-level tokenizer, shows how text vanishes into numbers, and demonstrates that the tokenizer has no understanding of what it's encoding.

---

## The state so far

```
WHAT YOU SEE          WHAT'S ACTUALLY THERE
──────────────        ──────────────────────
Words, sentences  →   Lists of integers (token IDs)
Meaning           →   Row numbers in a lookup table
"cat"             →   3797
```

The words are gone. We are one layer deeper, and there is still no understanding anywhere.

---

## Next

[RL-02 → Meaning Is Position](../RL-02-meaning-is-position/lesson.md)

The integers get turned into vectors. This is the step where people often think "now the meaning goes in." It doesn't.

# Lesson 18 — BPE Tokenization (How GPT Really Reads Text)

## Prerequisites

- Lesson 9 (tokenization basics)
- Lesson 14 (tiny GPT — you built a char-level model)
- General comfort with vocabularies and embeddings

---

## The Problem: How Do You Turn Text Into Numbers?

Before any transformer sees text, something has to convert raw characters into integer IDs. The embedding table you built in lesson 14 maps each ID to a vector. Lesson 9 used a character tokenizer. That was fine for a toy model, but real GPT models do something smarter.

Let us look at why the naive approaches fail.

---

## Approach 1: Character Tokenization

Your tiny GPT split "hello" into `['h','e','l','l','o']` — five tokens.

**Advantages:**
- Tiny vocab (maybe 100–200 characters for English)
- No out-of-vocabulary words ever
- Simple to implement

**Problems:**
- Every word is many tokens. "tokenization" is 13 tokens.
- The model must learn, from scratch, that `t`, `o`, `k`, `e`, `n` together mean something. That is a lot of work.
- Long sequences mean expensive attention (quadratic cost).
- GPT-2's training text would need roughly 4× more tokens than a word-level model.

---

## Approach 2: Word Tokenization

Split on whitespace. "the cat sat" → `['the', 'cat', 'sat']`.

**Advantages:**
- Common words are single tokens. The model sees "tokenization" as one unit.

**Problems:**
- English has hundreds of thousands of distinct word forms. You need a huge vocabulary (50k? 100k? 500k?).
- **Out-of-vocabulary (OOV) problem.** If the word "ChatGPT" never appeared in training, the model has no token for it. You must either map it to `<UNK>` (losing all information) or retrain.
- Morphology is invisible. "run", "runs", "running", "ran" — four separate entries with no shared representation. The model must learn their relationship purely through co-occurrence.
- Punctuation attached to words creates explosion: "word", "word.", "word!", "word," are all different entries.

---

## The Solution: Byte Pair Encoding (BPE)

BPE was originally a data compression algorithm (Gage, 1994). Philip Gage's idea: find the most common pair of bytes in a file, replace every occurrence with a new single byte, repeat. The file shrinks because common patterns get shorter codes.

In 2016, Sennrich et al. applied this idea to NLP for machine translation. OpenAI used it for GPT-2. It is still the basis of GPT-3, GPT-4, and most modern LLMs.

**The core idea:** Start with individual characters. Repeatedly merge the most frequent adjacent pair into a new token. After enough merges you have a vocabulary of ~50,000 tokens that captures common words as single tokens and splits rare words into sensible pieces.

---

## BPE Algorithm: Step-by-Step Example

We will use a tiny corpus:

```
low low low lowest newest
```

### Step 0: Pre-tokenize and split into characters

First, split the corpus into words. Then split each word into individual characters, and append a special end-of-word marker `</w>` to mark word boundaries:

```
l o w </w>       (appears 3 times)
l o w e s t </w> (appears 1 time)
n e w e s t </w> (appears 1 time)
```

Our initial vocabulary is every distinct character plus `</w>`:

```
{ 'l', 'o', 'w', '</w>', 'e', 's', 't', 'n' }
```

### Step 1: Count all adjacent pairs

Look at every adjacent pair across the corpus (weighted by word frequency):

| Pair         | Count |
|--------------|-------|
| ('l', 'o')   | 4     |  ← "low" appears 3×, "lowest" 1×
| ('o', 'w')   | 4     |
| ('w', '</w>') | 3    |
| ('w', 'e')   | 2     |  ← "lowest" and "newest"
| ('e', 's')   | 2     |
| ('s', 't')   | 2     |
| ('t', '</w>') | 2    |
| ('n', 'e')   | 1     |
| ...          | ...   |

**Most frequent pair: ('l', 'o') with count 4.**

### Step 2: Merge ('l', 'o') → 'lo'

Apply this merge rule everywhere in the corpus. Every `l o` sequence becomes `lo`:

```
lo w </w>          (3×)
lo w e s t </w>    (1×)
n e w e s t </w>   (1×)
```

Add `'lo'` to vocabulary. Record merge rule #1: `('l', 'o') → 'lo'`.

### Step 3: Re-count pairs, merge again

New counts:
| Pair          | Count |
|---------------|-------|
| ('lo', 'w')   | 4     |  ← most frequent now
| ('w', '</w>') | 3     |
| ('w', 'e')    | 2     |
| ...           | ...   |

Merge `('lo', 'w') → 'low'`:

```
low </w>          (3×)
low e s t </w>    (1×)
n e w e s t </w>  (1×)
```

### Step 4: Merge ('low', '</w>') → 'low</w>'

Next most frequent is `('low', '</w>')` with count 3:

```
low</w>           (3×)
low e s t </w>    (1×)
n e w e s t </w>  (1×)
```

The token `low</w>` represents the complete standalone word "low".

### Step 5: Continue merging...

Eventually `('e', 's') → 'es'`, `('es', 't') → 'est'`, `('est', '</w>') → 'est</w>'`, etc.

---

## How Tokenization Works at Inference

After training, you have an ordered list of merge rules. To encode a new word:

1. Split it into characters + `</w>`.
2. Apply merge rules in order (same order as training).
3. Stop when no more merges apply.

**Encoding "lowest":**
- Start: `['l', 'o', 'w', 'e', 's', 't', '</w>']`
- Apply rule `('l','o') → 'lo'`: `['lo', 'w', 'e', 's', 't', '</w>']`
- Apply rule `('lo','w') → 'low'`: `['low', 'e', 's', 't', '</w>']`
- Rule `('low','</w>')` does not apply (next token is 'e', not '</w>')
- Apply rule `('e','s') → 'es'`: `['low', 'es', 't', '</w>']`
- Apply rule `('es','t') → 'est'`: `['low', 'est', '</w>']`
- Apply rule `('est','</w>') → 'est</w>'`: `['low', 'est</w>']`
- Final tokens: **["low", "est</w>"]** — two tokens

**Encoding "newish" (unseen word):**
- Start: `['n', 'e', 'w', 'i', 's', 'h', '</w>']`
- Some merges apply: `('n','e') → 'ne'`, `('ne','w') → 'new'`
- 'i', 's', 'h' may remain as single characters if they never merged
- Final: **["new", "i", "s", "h</w>"]** — four tokens
- Not perfect, but not `<UNK>` either. The model gets real signal.

This is why BPE handles unseen words gracefully: it falls back to smaller pieces, ultimately to individual bytes if necessary.

---

## Why BPE Wins

| Property | Char tokenizer | Word tokenizer | BPE |
|---|---|---|---|
| Vocab size | ~200 | 500k–1M+ | ~50k |
| OOV words | Never | Always possible | Never (falls back to chars) |
| Common words | Many tokens | 1 token | 1 token |
| Rare words | Many tokens | `<UNK>` | Reasonable pieces |
| Sequence length | Very long | Shorter | Middle ground |
| Morphology | Invisible | Invisible | Partially captured |

A ~50k BPE vocabulary is the sweet spot:
- Common English words ("the", "cat", "running") are single tokens.
- Technical terms ("tokenization", "backpropagation") get split into known pieces.
- Any Unicode string can be encoded — nothing is truly unknown.

---

## BPE in Practice: tiktoken and GPT-2

OpenAI's `tiktoken` library implements GPT tokenization. You can install it with `pip install tiktoken` and try this Python:

```python
import tiktoken

enc = tiktoken.get_encoding("gpt2")  # GPT-2 tokenizer, 50k vocab

tokens = enc.encode("Hello, world!")
print(tokens)          # [15496, 11, 995, 0]
print(len(tokens))     # 4

text = enc.decode(tokens)
print(text)            # "Hello, world!"

# Longer example
tokens = enc.encode("BPE tokenization splits rare words into subword pieces.")
print(tokens)
# [33, 11401, 11241, 1634, 30778, 4071, 2456, 656, 850, 4775, 8458, 13]
```

Some observations:
- "Hello" → one token (common word)
- "tokenization" → multiple tokens (rare compound)
- "," and "." are their own tokens
- "!" is its own token

---

## GPT-2's Key Twist: Byte-Level BPE

The example above used characters. GPT-2 (and all subsequent GPT models) use **bytes** as the base unit, not characters.

Why? Unicode has ~150,000 code points. A Chinese character, an emoji, an Arabic letter — all are valid text. If you start from Unicode characters, your initial vocab is huge, and some code points are so rare they almost never appear.

Byte-level BPE starts from the 256 possible byte values (0–255). Every possible text, in every language, every emoji, every control character, is expressible as a sequence of bytes. You can never have an unknown token.

The initial vocab is exactly 256 entries. Then you run BPE merges on bytes. A common ASCII word like "the" might merge to a single token quickly. A rare Unicode character might stay as 2–4 byte tokens.

This means GPT-2 can tokenize **any string** — Python code, Japanese text, mathematical symbols, whatever — without ever hitting `<UNK>`.

---

## Comparison: Three Tokenization Strategies

| Feature | Char-level (your Lesson 14) | BPE (GPT-2) | WordPiece (BERT) |
|---|---|---|---|
| Base unit | Unicode chars | Bytes | Unicode chars |
| Vocab size | ~100 | 50,257 | 30,522 |
| Merge criterion | N/A | Frequency | Maximize LM likelihood |
| Unknown tokens | Never | Never | Yes (`[UNK]`) |
| Word boundary | No marker | `</w>` suffix | `##` prefix for continuation |
| Used by | Toy models, some char-RNNs | GPT-2, GPT-3, Codex, GPT-4 | BERT, RoBERTa, DistilBERT |
| Special tokens | None typically | `<\|endoftext\|>` | `[CLS]`, `[SEP]`, `[MASK]` |

**WordPiece** (used by BERT) is similar to BPE but merges based on maximizing the likelihood of the training data under a unigram language model, rather than raw frequency. In practice the results are similar.

**Unigram LM tokenization** (SentencePiece, used by T5, LLaMA) starts with a large vocab and prunes tokens, rather than growing from a small base.

---

## Vocabulary Sizes Across GPT Models

| Model | Tokenizer | Vocab Size | Notes |
|---|---|---|---|
| GPT-1 | BPE (char-level) | 40,000 | Original OpenAI BPE |
| GPT-2 | Byte-level BPE | 50,257 | +1 for `<\|endoftext\|>` |
| GPT-3 | Byte-level BPE | 50,257 | Same tokenizer as GPT-2 |
| Codex | Byte-level BPE | 50,257 + code tokens | GPT-2 base + code extensions |
| GPT-4 | cl100k_base | ~100,277 | Tiktoken cl100k encoding |
| GPT-4o | o200k_base | ~200,019 | Even larger for multilingual |

Why did GPT-4 double the vocab? A larger vocab means:
- More common words and phrases get single tokens
- Better multilingual coverage (Chinese/Japanese/Korean words that previously fragmented now have dedicated tokens)
- Shorter sequences → faster attention (recall O(n²))
- But: larger embedding table, more parameters

The tradeoff is real. Larger vocab = shorter sequences = faster inference, but more memory and more parameters to train.

---

## Key Insight: Tokenization Is Not Semantic

Here is something important to understand. Tokenization is **not** meaning-based. It is purely statistical on the training corpus.

- "tokenization" in English → probably 3-4 tokens
- "Tokenization" (capital T) → might be different tokens
- " tokenization" (leading space) → GPT-2 treats the space as part of the token, so this is DIFFERENT tokens

This has real consequences. Adding or removing a space, changing capitalization, or using synonyms can change which tokens appear and affect model behavior. This is not a bug — it is a consequence of purely frequency-based tokenization.

---

## What You Will Implement

In `index.ts` you will build BPE from scratch in TypeScript:

1. Pre-tokenize a corpus into words, split words into characters.
2. Count all adjacent pair frequencies across the corpus.
3. Merge the most frequent pair, record the rule.
4. Repeat for `numMerges` iterations.
5. Implement `encode(text)` that applies the learned merge rules.
6. Implement `decode(ids)` for round-trip verification.
7. Compare: char tokenizer gives 3 tokens for "the", BPE gives 1.

After this lesson, the tokenization column of the GPT diagram makes complete sense. You understand exactly what happens before the embedding lookup.

---

## Summary

- **Character tokenization:** tiny vocab, no OOV, very long sequences, hard for model to learn
- **Word tokenization:** short sequences, huge vocab, OOV problem is fatal
- **BPE:** frequency-based merges starting from chars (or bytes), ~50k vocab sweet spot, never OOV, common words single token, rare words split sensibly
- **GPT-2 uses byte-level BPE:** 256 byte base + 50k merges = handles any Unicode without unknowns
- **Vocab size has grown:** GPT-1 40k → GPT-2/3 50k → GPT-4 100k → GPT-4o 200k
- Tokenization is statistical, not semantic — spaces and capitalization matter

**Next lesson:** Positional encodings — how the transformer knows token order when attention is permutation-invariant.

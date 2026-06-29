# Lesson 09 — Tokenization

---

## The fundamental problem

A neural network can only process **numbers**. It cannot read text.

You cannot feed the word `"cat"` into a neural network.
You must first convert it to a number (or a list of numbers).

**Tokenization** is the process of converting text into numbers.

```
"Hello world" → [15496, 995]
```

---

## What is a token?

A **token** is the smallest unit of text that the model processes.

It could be:
- A character: `'c', 'a', 't'` → 3 tokens
- A word: `"cat"` → 1 token
- A sub-word: `"un", "happi", "ness"` → 3 tokens

Each token gets a unique integer ID from a **vocabulary**.

---

## Type 1: Character-level tokenization

The simplest approach: each character is one token.

```
Text:      "cat"
Tokens:    ['c', 'a', 't']
Token IDs: [2,   0,   19]   ← depends on vocabulary
```

**Building the vocabulary:**
1. Collect all unique characters in your training text
2. Sort them (for consistency)
3. Assign each one an integer ID

```ts
const text = "cat and dog";
const uniqueChars = [...new Set(text.split(""))].sort();
// uniqueChars = [" ", "a", "c", "d", "g", "n", "o", "t"]
// IDs:         [  0,   1,   2,   3,   4,   5,   6,   7]
```

**Vocabulary:** a mapping from character → ID, and ID → character.

---

## Vocabulary size matters

Character-level:
```
English text uses ~100 unique characters (letters, digits, punctuation)
vocab_size ≈ 100
```

Word-level:
```
English has ~170,000 words, common usage ~50,000
vocab_size ≈ 50,000
```

Sub-word (GPT, Claude):
```
vocab_size ≈ 50,000–100,000
```

Larger vocabulary → shorter sequences (more info per token)
Smaller vocabulary → longer sequences (simpler to implement)

---

## Encode and Decode

**Encode**: text → list of integer IDs
```
tokenizer.encode("cat") → [2, 0, 19]
```

**Decode**: list of integer IDs → text
```
tokenizer.decode([2, 0, 19]) → "cat"
```

These must be perfect inverses:
```
decode(encode(text)) === text  ← round-trip must always hold
```

---

## Building a character tokenizer step by step

```ts
class CharTokenizer {
  private charToId: Map<string, number>;
  private idToChar: Map<number, string>;

  constructor(text: string) {
    // Step 1: find all unique characters
    const chars = [...new Set(text.split(""))].sort();
    //            ↑ Set removes duplicates
    //                         ↑ sort for consistency

    // Step 2: assign IDs (0, 1, 2, ...)
    this.charToId = new Map(chars.map((c, i) => [c, i]));
    this.idToChar = new Map(chars.map((c, i) => [i, c]));
    //              ↑ reverse map for decoding
  }

  encode(text: string): number[] {
    return text.split("").map(c => this.charToId.get(c)!);
    //                   ↑ split into characters, look up each ID
  }

  decode(ids: number[]): string {
    return ids.map(id => this.idToChar.get(id)!).join("");
    //          ↑ look up each character, join into string
  }
}
```

---

## The training pairs

For a language model, the input is a sequence of tokens and the **target is the next token**.

Example with text `"the cat"`:
```
Encoded: [20, 8, 5, 3, 1, 20]

Input  → Target
  [20] →  8    ("t" → "h")
  [20,8] → 5   ("th" → "e")
  [20,8,5] → 3 ("the" → " ")
  ...
```

Every position in the text becomes a training example: "given these tokens, predict the next one."

---

## Why not just use word IDs?

If vocab = {"cat":0, "the":1, "dog":2}:

**Problem 1: unknown words**
```
"ChatGPT" → not in vocabulary → crash!
```

**Problem 2: vocabulary explosion**
```
"cat"   → ID 42
"cats"  → ID 4891   ← completely different!
"Cat"   → ID 12042  ← completely different again!
```

The model treats them as 3 unrelated things. But they're clearly related.

**Character-level solves both** — you can encode any text, and "cat"/"cats" share the characters 'c', 'a', 't'.

**Sub-word tokenization (BPE)** is the best of both: splits "cats" into "cat" + "s" so they share the "cat" part.

---

## Real example

```
text = "the quick brown fox"

CharTokenizer vocabulary (sorted):
  ' '→0, 'b'→1, 'c'→2, 'e'→3, 'f'→4, 'h'→5,
  'i'→6, 'k'→7, 'n'→8, 'o'→9, 'q'→10, 'r'→11,
  't'→12, 'u'→13, 'w'→14, 'x'→15

Encoding "the":
  't' → 12
  'h' → 5
  'e' → 3
  result: [12, 5, 3]

Decoding [12, 5, 3]:
  12 → 't'
  5  → 'h'
  3  → 'e'
  result: "the"  ✓
```

---

## GPT/Claude tokenization (for your curiosity)

They use **Byte-Pair Encoding (BPE)**:
1. Start with individual bytes (256 initial tokens)
2. Repeatedly merge the most common adjacent pair into a new token
3. Repeat until vocabulary reaches target size (e.g. 100,000)

Result: common words become single tokens, rare words split into sub-tokens:
```
"cat"         → [2298]             ← 1 token (common)
"unhappiness" → [403, 13, 22699]   ← 3 tokens (split)
"12345"       → [1954, 20]         ← 2 tokens
```

---

## Code for this lesson

See [index.ts](index.ts) — builds a character tokenizer, encodes/decodes text, shows training pairs.

## What's next
[Lesson 10 → Embeddings](../10-embeddings/lesson.md)

// Lesson 18 — BPE Tokenization from Scratch
// No imports required — pure TypeScript/Node.js

// ============================================================
// TYPES
// ============================================================

type MergeRule = [string, string];

interface BPETokenizer {
  mergeRules: MergeRule[];
  vocab: Map<string, number>;
  idToToken: Map<number, string>;
  encode: (text: string) => number[];
  decode: (ids: number[]) => string;
}

// ============================================================
// STEP 1: Pre-tokenize — split text into words, then words into
//         characters with an end-of-word marker.
//
//         "low low" → [["l","o","w","</w>"], ["l","o","w","</w>"]]
// ============================================================

function preTokenize(text: string): string[][] {
  // Split on whitespace; ignore empty strings
  const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
  return words.map(word => {
    const chars = word.split("");
    chars[chars.length - 1] = chars[chars.length - 1] + "</w>";
    return chars;
  });
}

// ============================================================
// STEP 2: Count how many times each word-form appears in the
//         corpus (needed to weight pair counts correctly).
// ============================================================

function getWordFrequencies(text: string): Map<string, number> {
  const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
  const freq = new Map<string, number>();
  for (const w of words) {
    freq.set(w, (freq.get(w) ?? 0) + 1);
  }
  return freq;
}

// ============================================================
// STEP 3: Convert word frequencies into a corpus representation.
//         Each entry: { tokens: string[], freq: number }
// ============================================================

interface WordEntry {
  tokens: string[];
  freq: number;
}

function buildCorpus(wordFreqs: Map<string, number>): WordEntry[] {
  const corpus: WordEntry[] = [];
  for (const [word, freq] of wordFreqs) {
    const chars = word.split("");
    chars[chars.length - 1] = chars[chars.length - 1] + "</w>";
    corpus.push({ tokens: chars, freq });
  }
  return corpus;
}

// ============================================================
// STEP 4: Count all adjacent pair frequencies across the corpus.
//         Pair counts are weighted by word frequency.
// ============================================================

function countPairs(corpus: WordEntry[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const entry of corpus) {
    const { tokens, freq } = entry;
    for (let i = 0; i < tokens.length - 1; i++) {
      const pair = tokens[i] + "|||" + tokens[i + 1]; // unique separator
      counts.set(pair, (counts.get(pair) ?? 0) + freq);
    }
  }
  return counts;
}

function pairKey(a: string, b: string): string {
  return a + "|||" + b;
}

function splitPairKey(key: string): [string, string] {
  const idx = key.indexOf("|||");
  return [key.slice(0, idx), key.slice(idx + 3)];
}

// ============================================================
// STEP 5: Find the pair with the highest frequency.
// ============================================================

function bestPair(pairCounts: Map<string, number>): [string, string] | null {
  let bestKey = "";
  let bestCount = -1;
  for (const [key, count] of pairCounts) {
    if (count > bestCount) {
      bestCount = count;
      bestKey = key;
    }
  }
  if (bestKey === "") return null;
  return splitPairKey(bestKey);
}

// ============================================================
// STEP 6: Apply a merge rule to the entire corpus.
//         Every occurrence of [a, b] becomes [a+b].
// ============================================================

function applyMerge(corpus: WordEntry[], a: string, b: string): WordEntry[] {
  const merged = a + b;
  return corpus.map(entry => {
    const newTokens: string[] = [];
    let i = 0;
    while (i < entry.tokens.length) {
      if (i < entry.tokens.length - 1 &&
          entry.tokens[i] === a &&
          entry.tokens[i + 1] === b) {
        newTokens.push(merged);
        i += 2;
      } else {
        newTokens.push(entry.tokens[i]);
        i += 1;
      }
    }
    return { tokens: newTokens, freq: entry.freq };
  });
}

// ============================================================
// STEP 7: Build the BPE tokenizer.
//         - Collect initial vocab from all characters.
//         - Run numMerges merge steps.
//         - Record each merge rule in order.
// ============================================================

function buildBPE(text: string, numMerges: number, verbose: boolean = false): BPETokenizer {
  const wordFreqs = getWordFrequencies(text);
  let corpus = buildCorpus(wordFreqs);

  // Collect all initial characters as the base vocab.
  // Also add bare single-character forms (without </w>) so that
  // unseen words whose final char produces an unknown "x</w>" token
  // can still fall back to individual characters.
  const vocabSet = new Set<string>();
  for (const entry of corpus) {
    for (const tok of entry.tokens) {
      vocabSet.add(tok);
      // Also add the bare char (strip </w> if present) so fallback works
      if (tok.endsWith("</w>")) {
        vocabSet.add(tok.slice(0, tok.length - 4));
      }
    }
  }
  // Always include the </w> marker itself as a token
  vocabSet.add("</w>");

  const mergeRules: MergeRule[] = [];

  if (verbose) {
    console.log("=== BPE Training ===");
    console.log(`Initial corpus (${corpus.length} distinct words):`);
    for (const entry of corpus) {
      console.log(`  [${entry.tokens.join(", ")}] × ${entry.freq}`);
    }
    console.log();
  }

  for (let step = 0; step < numMerges; step++) {
    const pairCounts = countPairs(corpus);
    if (pairCounts.size === 0) {
      if (verbose) console.log("No more pairs to merge.");
      break;
    }

    const best = bestPair(pairCounts);
    if (!best) break;

    const [a, b] = best;
    const merged = a + b;
    const count = pairCounts.get(pairKey(a, b)) ?? 0;

    mergeRules.push([a, b]);
    vocabSet.add(merged);

    if (verbose) {
      console.log(`Merge ${step + 1}: ('${a}', '${b}') → '${merged}'  [count=${count}]`);
    }

    corpus = applyMerge(corpus, a, b);
  }

  // Build vocab map: token → id (sorted for determinism)
  const sortedVocab = Array.from(vocabSet).sort();
  const vocab = new Map<string, number>();
  const idToToken = new Map<number, string>();
  sortedVocab.forEach((tok, idx) => {
    vocab.set(tok, idx);
    idToToken.set(idx, tok);
  });

  if (verbose) {
    console.log(`\nFinal vocab size: ${vocab.size}`);
    console.log("Vocab:", Array.from(vocab.keys()).join(", "));
    console.log();
  }

  // ============================================================
  // encode: Apply merge rules to a new string.
  //
  // Algorithm:
  //   1. Pre-tokenize into words.
  //   2. Split each word into characters with </w>.
  //   3. Apply all merge rules in training order.
  //   4. Map each resulting token to its ID.
  // ============================================================
  function encode(text: string): number[] {
    const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
    const allIds: number[] = [];

    for (const word of words) {
      // Start: split into characters, last char gets </w>
      let tokens: string[] = word.split("");
      tokens[tokens.length - 1] = tokens[tokens.length - 1] + "</w>";

      // Apply each merge rule in order
      for (const [a, b] of mergeRules) {
        const merged = a + b;
        const newTokens: string[] = [];
        let i = 0;
        while (i < tokens.length) {
          if (i < tokens.length - 1 && tokens[i] === a && tokens[i + 1] === b) {
            newTokens.push(merged);
            i += 2;
          } else {
            newTokens.push(tokens[i]);
            i += 1;
          }
        }
        tokens = newTokens;
      }

      // Map tokens to IDs.
      // If a token is not in the vocab (e.g. because the word contains
      // characters that never appeared in training), break it into its
      // constituent characters.  The last segment of the last token may
      // carry the </w> marker, so strip and re-add it separately.
      for (const tok of tokens) {
        if (vocab.has(tok)) {
          allIds.push(vocab.get(tok)!);
        } else {
          // tok might be e.g. "x</w>" — split into bare chars + </w>
          const isWordEnd = tok.endsWith("</w>");
          const bare = isWordEnd ? tok.slice(0, tok.length - 4) : tok;
          for (const ch of bare.split("")) {
            allIds.push(vocab.get(ch) ?? vocab.get("?") ?? 0);
          }
          if (isWordEnd) {
            // Append the </w> marker if it's in vocab, otherwise skip
            if (vocab.has("</w>")) {
              allIds.push(vocab.get("</w>")!);
            }
          }
        }
      }
    }

    return allIds;
  }

  // ============================================================
  // decode: Map IDs back to tokens, strip </w>, rejoin with spaces.
  // ============================================================
  function decode(ids: number[]): string {
    const tokens = ids.map(id => idToToken.get(id) ?? "?");
    // Reconstruct text: </w> marks word boundaries
    let result = "";
    for (const tok of tokens) {
      if (tok.endsWith("</w>")) {
        result += tok.slice(0, tok.length - 4) + " ";
      } else {
        result += tok;
      }
    }
    return result.trimEnd();
  }

  return { mergeRules, vocab, idToToken, encode, decode };
}

// ============================================================
// SIMPLE CHAR TOKENIZER (for comparison with Lesson 14)
// ============================================================

interface CharTokenizer {
  charToId: Map<string, number>;
  idToChar: Map<number, string>;
  encode: (text: string) => number[];
  decode: (ids: number[]) => string;
}

function buildCharTokenizer(text: string): CharTokenizer {
  const chars = new Set<string>(text.split(""));
  const sorted = Array.from(chars).sort();
  const charToId = new Map<string, number>();
  const idToChar = new Map<number, string>();
  sorted.forEach((ch, idx) => {
    charToId.set(ch, idx);
    idToChar.set(idx, ch);
  });

  function encode(text: string): number[] {
    return text.split("").map(ch => charToId.get(ch) ?? 0);
  }

  function decode(ids: number[]): string {
    return ids.map(id => idToChar.get(id) ?? "?").join("");
  }

  return { charToId, idToChar, encode, decode };
}

// ============================================================
// DEMO
// ============================================================

function separator(label: string): void {
  console.log("\n" + "=".repeat(60));
  console.log(label);
  console.log("=".repeat(60));
}

function main(): void {
  const corpus = "the cat sat on the mat. the fat cat ate the rat.";

  // ----------------------------------------------------------
  // DEMO 1: Build BPE with verbose output showing merge steps
  // ----------------------------------------------------------
  separator("DEMO 1: Building BPE on corpus");
  console.log(`Corpus: "${corpus}"\n`);

  const bpe = buildBPE(corpus, 20, true /* verbose */);

  // ----------------------------------------------------------
  // DEMO 2: Show merge rules in order
  // ----------------------------------------------------------
  separator("DEMO 2: Learned Merge Rules (in order)");
  bpe.mergeRules.forEach(([a, b], i) => {
    console.log(`  Rule ${String(i + 1).padStart(2)}: ('${a}', '${b}') → '${a + b}'`);
  });

  // ----------------------------------------------------------
  // DEMO 3: Encode sentences and show token breakdown
  // ----------------------------------------------------------
  separator("DEMO 3: Encoding Sentences");

  const sentences = [
    "the cat sat on the mat",
    "the fat rat ate the cat",
    "a new cat sat",
  ];

  for (const sent of sentences) {
    const ids = bpe.encode(sent);
    const tokens = ids.map(id => bpe.idToToken.get(id) ?? "?");
    console.log(`\nInput:  "${sent}"`);
    console.log(`Tokens: [${tokens.map(t => `'${t}'`).join(", ")}]`);
    console.log(`IDs:    [${ids.join(", ")}]`);
    console.log(`Count:  ${ids.length} tokens`);
  }

  // ----------------------------------------------------------
  // DEMO 4: Round-trip encode → decode
  // ----------------------------------------------------------
  separator("DEMO 4: Round-Trip Encode → Decode");

  const testSentences = [
    "the cat sat on the mat",
    "the fat rat ate the cat",
  ];

  let allPassed = true;
  for (const sent of testSentences) {
    const ids = bpe.encode(sent);
    const reconstructed = bpe.decode(ids);
    const passed = reconstructed === sent;
    if (!passed) allPassed = false;
    console.log(`\nOriginal:      "${sent}"`);
    console.log(`Reconstructed: "${reconstructed}"`);
    console.log(`Round-trip:    ${passed ? "PASS" : "FAIL"}`);
  }
  console.log(`\nAll round-trips passed: ${allPassed}`);

  // ----------------------------------------------------------
  // DEMO 5: BPE vs Char tokenizer comparison
  // ----------------------------------------------------------
  separator("DEMO 5: BPE vs Character Tokenizer");

  const charTok = buildCharTokenizer(corpus);

  const comparisonWords = ["the", "cat", "sat", "mat", "rat", "fat", "ate"];

  console.log("\nWord        | Char tokens  | BPE tokens   | Char IDs             | BPE IDs");
  console.log("-".repeat(90));

  for (const word of comparisonWords) {
    const charIds = charTok.encode(word);
    const bpeIds = bpe.encode(word);
    const charTokens = charIds.map(id => charTok.idToChar.get(id) ?? "?");
    const bpeTokens = bpeIds.map(id => bpe.idToToken.get(id) ?? "?");

    const wordPad = word.padEnd(11);
    const charTokStr = `[${charTokens.map(t => `'${t}'`).join(",")}]`.padEnd(13);
    const bpeTokStr = `[${bpeTokens.map(t => `'${t}'`).join(",")}]`.padEnd(13);
    const charIdStr = `[${charIds.join(",")}]`.padEnd(21);
    const bpeIdStr = `[${bpeIds.join(",")}]`;
    console.log(`${wordPad} | ${charTokStr} | ${bpeTokStr} | ${charIdStr} | ${bpeIdStr}`);
  }

  // ----------------------------------------------------------
  // DEMO 6: Token count comparison
  // ----------------------------------------------------------
  separator("DEMO 6: Token Count Comparison");

  const longText = "the cat sat on the mat the fat cat ate the rat";
  const charCount = charTok.encode(longText).length;
  const bpeCount = bpe.encode(longText).length;

  console.log(`\nText: "${longText}"`);
  console.log(`\nChar tokenizer: ${charCount} tokens (every character is a token)`);
  console.log(`BPE tokenizer:  ${bpeCount} tokens`);
  console.log(`\nBPE is ${(charCount / bpeCount).toFixed(1)}× more compact on this corpus`);

  // ----------------------------------------------------------
  // DEMO 7: Unseen-ish words (words with familiar subparts)
  // ----------------------------------------------------------
  separator("DEMO 7: Handling Partially-Seen Words");

  const unseenWords = ["cats", "rats", "mats", "batman"];
  console.log("\nWord       | BPE Tokens");
  console.log("-".repeat(40));
  for (const w of unseenWords) {
    const ids = bpe.encode(w);
    const tokens = ids.map(id => bpe.idToToken.get(id) ?? "?");
    console.log(`${w.padEnd(10)} | [${tokens.map(t => `'${t}'`).join(", ")}]`);
  }
  console.log("\nObservation: Even words not in the training corpus get");
  console.log("meaningful subword pieces — never <UNK>.");

  // ----------------------------------------------------------
  // DEMO 8: Manual BPE walkthrough (pedagogical trace)
  // ----------------------------------------------------------
  separator("DEMO 8: Manual Walkthrough — Classic 'low' Example");

  console.log("\nCorpus: 'low low low lowest newest'");
  console.log("(Replicating the textbook BPE example from lesson.md)\n");

  const miniCorpus = "low low low lowest newest";
  const miniBpe = buildBPE(miniCorpus, 10, false);

  console.log("Merge rules learned:");
  miniBpe.mergeRules.forEach(([a, b], i) => {
    console.log(`  ${i + 1}. ('${a}', '${b}') → '${a + b}'`);
  });

  const testWords = ["low", "lowest", "newest", "newish"];
  console.log("\nTokenization of test words:");
  for (const w of testWords) {
    const ids = miniBpe.encode(w);
    const tokens = ids.map(id => miniBpe.idToToken.get(id) ?? "?");
    console.log(`  "${w}" → [${tokens.map(t => `'${t}'`).join(", ")}] (${ids.length} token${ids.length === 1 ? "" : "s"})`);
  }
  console.log();
  console.log("Note: 'newish' was never in the training corpus.");
  console.log("Known subwords ('ne', 'w') tokenize correctly.");
  console.log("Characters 'i' and 'h' never appeared in training so they");
  console.log("fall back to </w>. In real byte-level BPE (GPT-2), every");
  console.log("byte 0-255 is always in the base vocab — true unknowns are impossible.");

  // ----------------------------------------------------------
  // SUMMARY
  // ----------------------------------------------------------
  separator("Summary");
  console.log(`
What we built:
  - buildBPE(text, numMerges): Trains a BPE tokenizer from scratch
    * Counts character pair frequencies across a corpus
    * Greedily merges the most frequent pair at each step
    * Records merge rules in training order
    * Returns encode() and decode() functions

  - CharTokenizer: Simple character-level tokenizer (Lesson 14 style)

Key observations from the demos:
  - Common short words like "the", "cat" → 1 BPE token vs 3 char tokens
  - BPE is ~${(charCount / bpeCount).toFixed(1)}× more compact on this corpus
  - Unseen words still tokenize gracefully (no <UNK>)
  - Merge rules capture morphological structure naturally

Real-world BPE (GPT-2):
  - Uses BYTES (not chars) as base → handles any Unicode
  - 256 byte base + ~50k merges = 50,257 vocab entries
  - Every possible string is encodable; nothing is truly unknown
`);
}

main();

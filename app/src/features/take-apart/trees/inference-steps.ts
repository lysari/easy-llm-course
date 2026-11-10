import type { InferenceSpec } from "@/features/take-apart/interfaces/inference-spec.interface";
import type { ModelPart } from "@/features/take-apart/interfaces/model-part.interface";
import type { RealNote } from "@/features/take-apart/interfaces/real-facts.interface";
import { count } from "@/shared/utils/format";
import { generateStep } from "./generate-steps";

const NO_MERGES = "Our tiny model has no merge rules. It splits on spaces, so each word is one piece.";

const numbersIn = (parts: ModelPart[]) => parts.reduce((sum, part) => sum + part.facts.weightCount * part.copies, 0);

const sourcesOf = (notes: RealNote[]) => [...new Set(notes.map((note) => note.source))];

function tokenizeStep(spec: InferenceSpec): ModelPart {
  return {
    id: "tokenize",
    icon: "text",
    title: "Tokenization",
    summary: "Splits your text into small pieces and gives each piece an ID.",
    copies: 1,
    facts: { shape: `${spec.tokenizer} · ${count(spec.wordCount)} words`, weightCount: 0, headline: { value: spec.wordCount, unit: "words" }, input: "text", output: "word IDs", sources: [...new Set([...spec.sources, ...sourcesOf(spec.tokenizerNotes)])], notes: spec.tokenizerNotes, reasons: spec.reasons.tokenize },
    tiny: { part: "tokens", level: "whole" },
    lesson: "Lessons 09, 18",
    reverseLesson: "RL-01",
    explanation: `Your text is cut into pieces, and each piece is swapped for its number in ${spec.modelTitle}'s list of ${count(spec.wordCount)} words. From here on the model only sees those numbers.`,
    why: "The model never sees letters. \"dog\" is just an ID, a position in a list, with no meaning attached yet.",
    story: {
      problem: "A model only computes with numbers, and it knows a fixed list of words. Without this step your text, in any letters or language, would have no way in.",
      demo: "gpt-2-tokenizer",
      when: "Every time you send a message or a prompt, before the model reads a single word. The IDs it writes back are turned into text the same way, in reverse.",
      feeds: "The list of word IDs goes to Embedding, which swaps each ID for its list of numbers.",
      nextId: "embedding",
    },
    formula: { lines: ["text → pieces → IDs"] },
    code: "const ids = pieces.map((piece) => list.indexOf(piece));",
    children: [
      {
        id: "split",
        icon: "split",
        title: "Split into pieces",
        summary: "Cuts the text into pieces from the word list, joining letters step by step.",
        copies: 1,
        facts: { shape: "merge rules, in the order learned", weightCount: 0, headline: spec.merges, input: "text as bytes", output: "pieces", sources: sourcesOf(spec.mergeNotes), notes: spec.mergeNotes, reasons: spec.reasons.split },
        tiny: { part: "tokens", level: "whole" },
        lesson: "Lesson 18",
        reverseLesson: "RL-01",
        explanation: "The text starts as single bytes. Merge rules join neighbouring pieces into bigger ones, so common words become one piece and rare words stay split.",
        why: "Any text can be split this way, even words the list has never seen, because single bytes are always in the list.",
        story: {
          problem: "No list can hold every word, name and typo. Without splitting, a word missing from the list could not be read at all; cut into known pieces, any text fits.",
          demo: "gpt-2-tokenizer",
          when: "On every message you send, word by word, before any ID is looked up.",
          feeds: "The pieces go to the lookup, which gives each one its ID.",
          nextId: "lookup",
        },
        formula: { lines: ["\"d\" \"o\" \"g\" → \"do\" \"g\" → \"dog\""] },
        children: [
          {
            id: "apply-merge",
            icon: "merge",
            title: "One merge rule",
            summary: "Joins two side-by-side pieces into one, when a learned rule names them.",
            copies: 1,
            facts: { shape: "1 rule", weightCount: 0, headline: spec.merges, input: "two pieces", output: "one joined piece", inputDetail: "two neighbouring pieces", outputDetail: "one joined piece, or no change", sources: sourcesOf(spec.mergeNotes), notes: spec.mergeNotes, reasons: spec.reasons["apply-merge"] },
            tiny: { part: "tokens", level: "number", missing: NO_MERGES },
            lesson: "Lesson 18",
            reverseLesson: "RL-01",
            explanation: "Ground zero. One learned rule, such as \"do\" + \"g\" → \"dog\". If those two pieces sit side by side, join them into one.",
            why: "It is a find-and-replace. Nothing here knows what a dog is.",
            story: {
              problem: "Starting from single bytes, text would be one piece per letter, far too long. Joining the pairs the tokenizer learned makes common words one piece, so the model reads fewer pieces.",
              demo: "gpt-2-tokenizer",
              when: "Over and over inside each word of your message, until no learned rule fits.",
              feeds: "When no rule fits any more, the pieces left are final and each is looked up for its ID.",
              nextId: "lookup",
            },
            formula: { lines: ["if a = \"do\" and b = \"g\", then ab = \"dog\""] },
            groundZero: "action",
          },
        ],
      },
      {
        id: "lookup",
        icon: "hash",
        title: "Look up one piece",
        summary: "Finds one piece in the word list and writes down its place number.",
        copies: 1,
        facts: { shape: "1 row of the list", weightCount: 0, headline: { value: spec.wordCount, unit: "words in the list" }, input: "one piece", output: "its ID", sources: spec.sources, reasons: spec.reasons.lookup },
        tiny: { part: "lookup", level: "number" },
        lesson: "Lesson 09",
        reverseLesson: "RL-01",
        explanation: "Ground zero. Look up one piece in the vocabulary, the list of words, and write down its place in the list.",
        why: "This is the whole trick of turning text into numbers: a lookup in a list.",
        story: {
          problem: "The model's tables are found by number, not by text. Without the lookup a piece like \"dog\" could not pick its row in the word table.",
          demo: "gpt-2-tokenizer",
          when: "Once for every piece of every message you send, right after splitting.",
          feeds: "The ID picks that word's row in the word table, at the Embedding step.",
          nextId: "embedding",
        },
        formula: { lines: ["id = list.indexOf(piece)"] },
        code: "const id = list.indexOf(piece);",
        groundZero: "action",
      },
    ],
  };
}

function embeddingStep(spec: InferenceSpec): ModelPart {
  return {
    id: "embedding",
    icon: "grid",
    title: "Embedding",
    summary: "Swaps each word ID for a list of numbers that stands for that word.",
    copies: 1,
    facts: { shape: `${count(spec.width)} numbers per word`, weightCount: numbersIn(spec.embedding), headline: { value: spec.width, unit: "numbers per word" }, input: "word IDs", output: "vectors", sources: spec.sources, reasons: spec.reasons.embedding },
    tiny: { part: "embedding", level: "whole" },
    lesson: spec.embeddingLesson,
    reverseLesson: "RL-02",
    explanation: spec.embeddingText,
    why: "From here on each word is a list of numbers, a point in space. Meaning is only where that point sits.",
    story: {
      problem: "A word ID is only a place in a list: ID 1 is not close to ID 2 in any useful way. Without swapping IDs for lists of numbers, the layers would have nothing to compute with.",
      demo: "tiny-part",
      when: "Every time you send a message, for every word in it, and again for each new word the model writes.",
      feeds: "Each word's numbers go into the first of the layers, which work on them one layer after another.",
      nextId: "layers",
    },
    formula: { lines: ["vector = word table[id]"] },
    code: "const vectors = ids.map((id) => wordTable[id]);",
    children: spec.embedding,
  };
}

function outputStep(spec: InferenceSpec): ModelPart {
  const softmax: ModelPart = {
    id: "softmax",
    icon: "percent",
    title: "Softmax",
    summary: "Turns the scores into chances that add up to 100%.",
    copies: 1,
    facts: { shape: `${count(spec.wordCount)} chances`, weightCount: 0, headline: { value: spec.wordCount, unit: "chances" }, input: `${count(spec.wordCount)} scores`, output: "chances adding to 1", outputDetail: "chances that add up to 1", sources: spec.sources, reasons: spec.reasons.softmax },
    tiny: { part: "chances", level: "whole" },
    lesson: "Lesson 11",
    reverseLesson: "RL-04",
    explanation: "Turns the scores into chances. Each score is raised as a power of e, so bigger scores grow much bigger, then all are divided by their total.",
    why: "The chances are not beliefs. They are scores squashed so they add up to 1.",
    story: {
      problem: "Scores can be any size, even below zero, so they cannot be used as chances. Without softmax there would be no fair way to roll for a word.",
      demo: "next-word",
      when: "Once for every word the model writes, right after the scores.",
      feeds: "The chances, which add up to 100%, go to Generate, where one word is picked.",
      nextId: "generate",
    },
    formula: { lines: ["chance(word) = e^score(word) / Σ e^score"] },
    children: [
      {
        id: "chance",
        icon: "hash",
        title: "One word's chance",
        summary: "Turns one word's score into its chance, by comparing it with all the others.",
        copies: spec.wordCount,
        facts: { shape: "1", weightCount: 0, headline: { value: spec.wordCount, unit: "per new word" }, input: "1 score, the total", output: "1 chance", inputDetail: "one score and the total", sources: spec.sources, reasons: spec.reasons.chance },
        tiny: { part: "chances", level: "number" },
        lesson: "Lesson 11",
        reverseLesson: "RL-04",
        explanation: "Ground zero. e to the power of one word's score, divided by the total for all words.",
        why: "One power and one divide. That is all a \"chance\" is here.",
        story: {
          problem: "A single score means nothing alone: it only counts next to all the others. This step turns one word's score into its share of the total.",
          demo: "next-word",
          when: `For every word the model writes, ${count(spec.wordCount)} times: once for each word in its list.`,
          feeds: "All the chances together make the list that one word is picked from.",
          nextId: "generate",
        },
        formula: { lines: ["chance = e^score / total"] },
        code: "const chance = Math.exp(score) / total;",
        groundZero: "action",
      },
    ],
  };
  const parts = [spec.finalNorm, spec.outputHead, softmax];
  return {
    id: "output",
    icon: "percent",
    title: "Output head",
    summary: "Scores every word in the list and turns the scores into chances.",
    copies: 1,
    facts: { shape: "final norm, output table, softmax", weightCount: numbersIn(parts), headline: { value: spec.wordCount, unit: "chances" }, input: "last word's vector", output: "chances", inputDetail: "the last word's vector", sources: spec.sources, reasons: spec.reasons.output },
    tiny: { part: "chances", level: "whole" },
    lesson: "Lessons 11, 20",
    reverseLesson: "RL-04",
    explanation: `The last word's ${count(spec.width)} numbers are steadied by one last norm, scored against every word in the list, and the scores are turned into chances.`,
    why: "Only the last word's numbers are used to guess what comes next. The top chance is the best guess, not knowledge.",
    story: {
      problem: "After the layers, each word is still just a list of numbers. Without the output head there would be no way to say which word should come next.",
      demo: "next-word",
      when: "Once for every word the model writes, each time you send a message. Only the last word's numbers are used.",
      feeds: "The chances go to Generate, which picks one word.",
      nextId: "generate",
    },
    formula: { lines: ["chances = softmax(norm(h) × output table)"] },
    children: parts,
  };
}

export function inferenceRoot(spec: InferenceSpec): ModelPart {
  const steps = [tokenizeStep(spec), embeddingStep(spec), spec.layers, outputStep(spec), generateStep(spec)];
  return {
    id: "use",
    icon: "play",
    title: "Use a model",
    summary: "How a model turns your text into the next word, in 5 steps.",
    copies: 1,
    facts: { shape: "5 steps", weightCount: numbersIn(steps), headline: { value: numbersIn(steps), unit: "numbers", isComputed: true }, input: "your text", output: "the next words", sources: spec.sources },
    tiny: { part: "model", level: "whole" },
    lesson: "Lessons 14, 20",
    reverseLesson: "RL-00",
    explanation: `How ${spec.modelTitle} turns your text into the next word: cut it into word IDs, look up their vectors, run the layers, score every word, and pick one.`,
    why: "Every step is a lookup, a multiply-and-add or a random pick. None of them needs understanding.",
    formula: { lines: ["text → IDs → vectors → layers → chances → word"] },
    children: steps,
  };
}

import type { ModelPart, RealNote } from "@/features/take-apart/interfaces/model-part.interface";
import { count } from "@/shared/utils/format";

interface InferenceSpec {
  modelTitle: string;
  tokenizer: string;
  wordCount: number;
  width: number;
  context: number;
  kvWidth: number;
  tokenizerNotes: RealNote[];
  embeddingLesson: string;
  embeddingText: string;
  embedding: ModelPart[];
  layers: ModelPart;
  finalNorm: ModelPart;
  outputHead: ModelPart;
}

const NO_MERGES = "Our tiny model has no merge rules. It splits on spaces, so each word is one piece.";
const NO_CACHE = "Our tiny model keeps no cache. It runs the whole sentence again each round, which gives the same answer, only slower.";

const numbersIn = (parts: ModelPart[]) => parts.reduce((sum, part) => sum + part.facts.weightCount * part.copies, 0);

function tokenizeStep(spec: InferenceSpec): ModelPart {
  return {
    id: "tokenize",
    icon: "text",
    title: "Tokenization",
    copies: 1,
    facts: { shape: `${spec.tokenizer} · ${count(spec.wordCount)} words`, weightCount: 0, input: "text", output: "word IDs", notes: spec.tokenizerNotes },
    tiny: { part: "tokens", level: "whole" },
    lesson: "Lessons 09, 18",
    reverseLesson: "RL-01",
    explanation: `Your text is cut into pieces, and each piece is swapped for its number in ${spec.modelTitle}'s list of ${count(spec.wordCount)} words. From here on the model only sees those numbers.`,
    why: "The model never sees letters. \"dog\" is just an ID, a position in a list, with no meaning attached yet.",
    formula: { lines: ["text → pieces → IDs"] },
    code: "const ids = pieces.map((piece) => list.indexOf(piece));",
    children: [
      {
        id: "split",
        icon: "split",
        title: "Split into pieces",
        copies: 1,
        facts: { shape: "merge rules, in the order learned", weightCount: 0, input: "text as bytes", output: "pieces" },
        tiny: { part: "tokens", level: "whole" },
        lesson: "Lesson 18",
        reverseLesson: "RL-01",
        explanation: "The text starts as single bytes. Merge rules join neighbouring pieces into bigger ones, so common words become one piece and rare words stay split.",
        why: "Any text can be split this way, even words the list has never seen, because single bytes are always in the list.",
        formula: { lines: ["\"d\" \"o\" \"g\" → \"do\" \"g\" → \"dog\""] },
        children: [
          {
            id: "apply-merge",
            icon: "merge",
            title: "One merge rule",
            copies: 1,
            facts: { shape: "1 rule", weightCount: 0, input: "two neighbouring pieces", output: "one joined piece, or no change" },
            tiny: { part: "tokens", level: "number", missing: NO_MERGES },
            lesson: "Lesson 18",
            reverseLesson: "RL-01",
            explanation: "Ground zero. One learned rule, such as \"do\" + \"g\" → \"dog\". If those two pieces sit side by side, join them into one.",
            why: "It is a find-and-replace. Nothing here knows what a dog is.",
            formula: { lines: ["if a = \"do\" and b = \"g\", then ab = \"dog\""] },
            groundZero: "action",
          },
        ],
      },
      {
        id: "lookup",
        icon: "hash",
        title: "Look up one piece",
        copies: 1,
        facts: { shape: "1 row of the list", weightCount: 0, input: "one piece", output: "its ID" },
        tiny: { part: "lookup", level: "number" },
        lesson: "Lesson 09",
        reverseLesson: "RL-01",
        explanation: "Ground zero. Look up one piece in the vocabulary, the list of words, and write down its place in the list.",
        why: "This is the whole trick of turning text into numbers: a lookup in a list.",
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
    copies: 1,
    facts: { shape: `${count(spec.width)} numbers per word`, weightCount: numbersIn(spec.embedding), input: "word IDs", output: "vectors" },
    tiny: { part: "embedding", level: "whole" },
    lesson: spec.embeddingLesson,
    reverseLesson: "RL-02",
    explanation: spec.embeddingText,
    why: "From here on each word is a list of numbers, a point in space. Meaning is only where that point sits.",
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
    copies: 1,
    facts: { shape: `${count(spec.wordCount)} chances`, weightCount: 0, input: `${count(spec.wordCount)} scores`, output: "chances that add up to 1" },
    tiny: { part: "chances", level: "whole" },
    lesson: "Lesson 11",
    reverseLesson: "RL-04",
    explanation: "Turns the scores into chances. Each score is raised as a power of e, so bigger scores grow much bigger, then all are divided by their total.",
    why: "The chances are not beliefs. They are scores squashed so they add up to 1.",
    formula: { lines: ["chance(word) = e^score(word) / Σ e^score"] },
    children: [
      {
        id: "chance",
        icon: "hash",
        title: "One word's chance",
        copies: spec.wordCount,
        facts: { shape: "1", weightCount: 0, input: "one score and the total", output: "1 chance" },
        tiny: { part: "chances", level: "number" },
        lesson: "Lesson 11",
        reverseLesson: "RL-04",
        explanation: "Ground zero. e to the power of one word's score, divided by the total for all words.",
        why: "One power and one divide. That is all a \"chance\" is here.",
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
    copies: 1,
    facts: { shape: "final norm, output table, softmax", weightCount: numbersIn(parts), input: "the last word's vector", output: "chances" },
    tiny: { part: "chances", level: "whole" },
    lesson: "Lessons 11, 20",
    reverseLesson: "RL-04",
    explanation: `The last word's ${count(spec.width)} numbers are steadied by one last norm, scored against every word in the list, and the scores are turned into chances.`,
    why: "Only the last word's numbers are used to guess what comes next. The top chance is the best guess, not knowledge.",
    formula: { lines: ["chances = softmax(norm(h) × output table)"] },
    children: parts,
  };
}

function generateStep(spec: InferenceSpec): ModelPart {
  return {
    id: "generate",
    icon: "pen",
    title: "Generate",
    copies: 1,
    facts: { shape: `1 word per round, up to ${count(spec.context)} words`, weightCount: 0, input: "chances", output: "new words, one at a time" },
    tiny: { part: "write", level: "whole" },
    lesson: "Lessons 21, 23",
    reverseLesson: "RL-04",
    explanation: "Pick one word from the chances, add it to the end, and run steps 1 to 4 again. Repeat until the stop word comes up or the text is too long.",
    why: "A whole answer is just this loop. Each word is picked without a plan for the next one.",
    formula: { lines: ["repeat: pick, add, run again", "until stop or full"] },
    children: [
      {
        id: "write-round",
        icon: "repeat",
        title: "One writing round",
        copies: 1,
        facts: { shape: "pick, add, check", weightCount: 0, input: "all words so far", output: "one more word" },
        tiny: { part: "write", level: "row" },
        lesson: "Lessons 14, 23",
        reverseLesson: "RL-04",
        explanation: "One pass of the loop: pick a word from the chances, add it to the end, then check whether to stop.",
        why: "The model writes left to right and never goes back to change a word.",
        formula: { lines: ["word = pick(chances)", "words = [...words, word]"] },
        children: [
          {
            id: "pick",
            icon: "dice",
            title: "Pick one word",
            copies: 1,
            facts: { shape: "1 random roll", weightCount: 0, input: "chances", output: "one word" },
            tiny: { part: "pick", level: "number" },
            lesson: "Lesson 23",
            reverseLesson: "RL-04",
            explanation: "Ground zero. Roll a random number from 0 to 1, then walk down the chances, taking each away, until the roll runs out. That word is picked.",
            why: "Likely words are picked more often, but not always. That is why the same start can give different endings.",
            formula: { lines: ["roll = roll − chance(word)", "stop at the first roll ≤ 0"] },
            code: "roll = roll - option.chance;",
            groundZero: "action",
          },
          {
            id: "append",
            icon: "plus",
            title: "Add it to the end",
            copies: 1,
            facts: { shape: "1 more ID", weightCount: 0, input: "words so far and the picked word", output: "one word longer" },
            tiny: { part: "pick", level: "whole" },
            lesson: "Lesson 14",
            reverseLesson: "RL-04",
            explanation: "Ground zero. Append the picked word's ID to the input, ready for the next round.",
            why: "The model's own guess becomes part of what it reads next.",
            formula: { lines: ["ids = [...ids, picked]"] },
            code: "ids.push(picked);",
            groundZero: "action",
          },
          {
            id: "stop-check",
            icon: "check",
            title: "Stop check",
            copies: 1,
            facts: { shape: "2 checks", weightCount: 0, input: "the picked word and the length", output: "go on, or stop" },
            tiny: { part: "write", level: "number" },
            lesson: "Lesson 14",
            reverseLesson: "RL-04",
            explanation: `Ground zero. Compare the picked word with the stop word, and the length with ${count(spec.context)} words. Stop if either matches.`,
            why: "The model does not decide it is finished. It picks the stop word like any other word.",
            formula: { lines: ["stop if word = stop word", `or length = ${count(spec.context)}`] },
            groundZero: "action",
          },
        ],
      },
      {
        id: "kv-cache",
        icon: "key",
        title: "KV cache",
        copies: 1,
        facts: { shape: `2 × ${count(spec.kvWidth)} numbers per word, per layer`, weightCount: 0, input: "each new word's keys and values", output: "earlier keys and values, ready to reuse" },
        tiny: { part: "write", level: "whole", missing: NO_CACHE },
        lesson: "Lesson 21",
        reverseLesson: "RL-03",
        explanation: "Ground zero. Save one word's key and value from one layer, so the next round reads them back instead of working them out again. Only the new word is run through the layers.",
        why: "It changes nothing about the answer, only the speed. These are saved results, not stored numbers learned in training.",
        formula: { lines: ["cache[layer].push(key, value)"] },
        groundZero: "action",
      },
    ],
  };
}

export function inferenceRoot(spec: InferenceSpec): ModelPart {
  const steps = [tokenizeStep(spec), embeddingStep(spec), spec.layers, outputStep(spec), generateStep(spec)];
  return {
    id: "use",
    icon: "play",
    title: "Use a model",
    copies: 1,
    facts: { shape: "5 steps", weightCount: numbersIn(steps), input: "your text", output: "the next words" },
    tiny: { part: "model", level: "whole" },
    lesson: "Lessons 14, 20",
    reverseLesson: "RL-00",
    explanation: `How ${spec.modelTitle} turns your text into the next word: cut it into word IDs, look up their vectors, run the layers, score every word, and pick one.`,
    why: "Every step is a lookup, a multiply-and-add or a random pick. None of them needs understanding.",
    formula: { lines: ["text → IDs → vectors → layers → chances → word"] },
    children: steps,
  };
}

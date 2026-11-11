import type { ModelPart, RealModel } from "@/features/inside-real-models/interfaces/model-part.interface";
import { GPT_2_BUILD } from "./gpt-2-build-facts";
import { GPT_2_CONTEXT, GPT_2_FINAL_NORM, GPT_2_MERGES, GPT_2_NORM_CODE, GPT_2_NORM_PLACE, GPT_2_NO_REASON, GPT_2_OUTPUT, GPT_2_POSITION_TABLE, GPT_2_RANDOM_START, GPT_2_SIZE_HELPS, GPT_2_TRANSFORMER, GPT_2_USE_REASONS, GPT_2_WORD_TABLE, GPT_2_WRITING_NOTES } from "./gpt-2-reasons";
import { gpt2Layers } from "./gpt-2-layer";
import { inferenceRoot } from "./inference-steps";
import { lifecycleRoot } from "./lifecycle-steps";
import { normPart, tablePart } from "./part-builders";
import { GPT_2_SETTINGS } from "./sources";

const WIDTH = 768;
const WORD_COUNT = 50_257;
const CONTEXT = 1024;

const wordTable = tablePart({
  id: "word-table",
  icon: "embedding",
  title: "Word table",
  summary: "Holds one list of numbers for every word the model knows.",
  copies: 1,
  rows: WORD_COUNT,
  cols: WIDTH,
  rowName: "word",
  tinyPart: "word-table",
  sources: GPT_2_SETTINGS,
  explanation:
    "One row of 768 numbers for each of the 50,257 words in GPT-2's list. A word's id picks its row. The same table is used again at the very end to score every word.",
  why: "Each word becomes a point in a space with 768 directions. Only training decides where each point sits.",
  story: {
    problem: "The model computes with numbers, but a word ID is only a place in a list. Without the word table there would be nothing to multiply and add.",
    demo: "tiny-part",
    when: "First thing, every time text is read: each word ID picks its row. Training sets every number in it.",
    feeds: "Each word's row is added to its place's row and goes into the first layer.",
    nextId: "word-table-row",
  },
  reasons: [GPT_2_WORD_TABLE, GPT_2_NO_REASON],
  codeReasons: [GPT_2_WORD_TABLE],
});

const positionTable = tablePart({
  id: "position-table",
  icon: "position",
  title: "Position table",
  summary: "Holds one list of numbers for each place in the text, so word order counts.",
  copies: 1,
  rows: CONTEXT,
  cols: WIDTH,
  rowName: "place",
  tinyPart: "position-table",
  sources: GPT_2_SETTINGS,
  explanation: "One learned row of 768 numbers for each place from 1 to 1,024. It is added to the word's row so the model knows the order.",
  why: "Without it, \"dog bites man\" and \"man bites dog\" would look the same. It also caps GPT-2 at 1,024 words.",
  story: {
    problem: "Attention by itself does not see word order: \"dog bites man\" and \"man bites dog\" hold the same words. Without a row for each place, the order would be lost.",
    demo: "tiny-part",
    when: "Every time text is read, one row for each place, added to that word's row.",
    feeds: "Word row plus place row is what goes into the first layer.",
    nextId: "position-table-row",
  },
  reasons: [GPT_2_POSITION_TABLE, GPT_2_CONTEXT],
  codeReasons: [GPT_2_POSITION_TABLE],
});

const finalNorm = normPart({
  id: "final-norm",
  icon: "norm",
  title: "Final norm",
  summary: "Keeps the last word's numbers in a steady range before they are scored.",
  copies: 1,
  size: WIDTH,
  hasShift: true,
  tinyPart: "final-norm",
  sources: [...GPT_2_SETTINGS, "gpt-2-paper"],
  explanation: "One last LayerNorm on the last word's 768 numbers, with its own 768 scale and 768 shift numbers, before they are scored.",
  why: "GPT-2 puts each norm before its part, so the very last layer's output still needs one.",
  story: {
    problem: "GPT-2 puts each norm before its part, so the last layer's output has not been steadied yet. Without this one, the scores would come from numbers of any size.",
    demo: "tiny-part",
    when: "Once, after the 12th layer, on the last word's numbers, each time a new word is guessed.",
    feeds: "The steadied numbers are scored against every word by the output table.",
    nextId: "output-head",
  },
  reasons: [GPT_2_NORM_PLACE, GPT_2_FINAL_NORM],
  codeReasons: [GPT_2_NORM_CODE],
});

const outputHead: ModelPart = {
  id: "output-head",
  icon: "scores",
  title: "Output table",
  summary: "Scores every word against the last word's numbers, reusing the word table.",
  copies: 1,
  facts: { shape: "768 × 50,257, the word table reused", weightCount: 0, headline: { value: 0, unit: "new numbers", isComputed: true }, input: "768 for last word", output: "50,257 scores", inputDetail: "768 numbers for the last word", sources: GPT_2_SETTINGS, reasons: [GPT_2_OUTPUT] },
  tiny: { part: "output-table", level: "whole" },
  explanation:
    "Turns the last word's 768 numbers into one score per word. GPT-2 reuses the word table here instead of storing a second table, so it adds 0 numbers.",
  why: "Softmax turns the scores into chances, and one word is picked. The top chance is not knowledge, only the best guess the numbers give.",
  story: {
    problem: "The layers end with 768 numbers, but the model has to say something about each of its 50,257 words. Without the output table there would be no score to turn into a chance.",
    demo: "tiny-part",
    when: "Once for each new word, at the very end of the pass.",
    feeds: "The 50,257 scores go to softmax, which turns them into chances, and one word is picked.",
    nextId: "output-score",
  },
  formula: { lines: ["score(word) = h · row(word)", "chances = softmax(scores)"] },
  code: "const scores = wordTable.map((row) => dot(h, row));\nconst chances = softmax(scores);",
  children: [
    {
      id: "output-score",
      icon: "number",
      title: "One word's score",
      summary: "Scores one word: the last word's numbers times that word's row, added up.",
      copies: WORD_COUNT,
      facts: { shape: "768 × 1", weightCount: 0, headline: { value: WIDTH, unit: "pairs multiplied" }, input: "768 numbers, 1 row", output: "1 score", inputDetail: "768 numbers and one word's row", sources: GPT_2_SETTINGS, reasons: [GPT_2_OUTPUT] },
      tiny: { part: "output-table", level: "number" },
      explanation: "The score for one word: the last word's 768 numbers times that word's row in the word table, added up.",
      why: "A word scores high when its row points the same way as the last word's numbers.",
      story: {
        problem: "Each word in the list needs its own score. One score says how well the last word's numbers match that word's row; without it the word could never be picked.",
        demo: "tiny-part",
        when: "It runs 50,257 times for each new word, once for every word in the list.",
        feeds: "All the scores together go to softmax, which turns them into chances.",
        nextId: "output-score-multiply-add",
      },
      formula: { lines: ["score = h1 × r1 + h2 × r2 + … + h768 × r768"] },
      code: "let sum = 0;\nfor (let i = 0; i < 768; i++) {\n  sum = sum + h[i] * row[i];\n}",
      children: [
        {
          id: "output-score-multiply-add",
          icon: "multiply-add",
          title: "One multiply-and-add",
          summary: "Multiplies one of the last word's numbers by one from the word table, and adds.",
          copies: WIDTH,
          facts: { shape: "1 × 1", weightCount: 0, headline: { value: WIDTH, unit: "per output number" }, input: "1 number, 1 stored", output: "the running sum", inputDetail: "1 number and 1 number from the word table", sources: GPT_2_SETTINGS, reasons: [GPT_2_OUTPUT] },
          tiny: { part: "output-table", level: "multiply-add" },
          explanation: "Ground zero. One of the last word's numbers times one number from the word table, added to the running sum. The number is counted once, in the word table.",
          why: "Choosing the next word comes down to this, done 768 × 50,257 times.",
          story: {
            problem: "This is the bottom of choosing a word. Without this one multiply and add, no score could be made at all.",
            demo: "tiny-part",
            when: "It runs 768 times for each word's score, every time a new word is guessed.",
            feeds: "After the last one, the running sum is one word's score.",
          },
          formula: { lines: ["sum = sum + h × r"] },
          code: "sum = sum + h * r;",
          groundZero: "multiply-add",
        },
      ],
    },
  ],
};

const root: ModelPart = {
  id: "model",
  icon: "model",
  title: "GPT-2 small",
  summary: "The whole model: word IDs go in, and a score for every word comes out.",
  copies: 1,
  facts: {
    shape: "12 layers × 768 numbers per word",
    weightCount: 124_439_808,
    headline: { value: 124_439_808, unit: "numbers", isComputed: true },
    input: "up to 1,024 word IDs",
    output: "50,257 scores",
    outputDetail: "50,257 scores, one per word it knows",
    sources: [...GPT_2_SETTINGS, "gpt-2-card"],
    notes: [
      { label: "Released", value: "February 2019", source: "gpt-2-card" },
      { label: "Size", value: "124,439,808, worked out from config.json; the model card says 124 million", source: "gpt-2-card" },
      { label: "In the paper", value: "listed as 117M for the smallest size", source: "gpt-2-paper" },
    ],
    reasons: [GPT_2_TRANSFORMER, GPT_2_SIZE_HELPS, GPT_2_RANDOM_START],
  },
  explanation:
    "The whole model, released by OpenAI in 2019. Words go in as ids, and out comes one score for every word in its list. The highest scores are the likely next words. Everything inside is stored numbers, multiplied and added.",
  why: "Its writing looks like understanding, but every step is arithmetic on 124,439,808 stored numbers. Take it apart to see.",
  story: {
    problem: "The parts only work in one order: rows in, 12 layers, a norm, scores out. Without the whole model wired together, no part would get its input or hand on its output.",
    demo: "architecture",
    when: "You write the whole model once, at the Architecture stage, then train it, then use it to write text.",
    feeds: "Its scores, turned into chances, are what picks each next word when the model writes.",
    nextId: "word-table",
  },
  formula: { lines: ["x = word rows + place rows", "chances = softmax(head(norm(layers(x))))"] },
  code: "const x = add(wordRows(ids), placeRows(ids));\nconst h = layers.reduce((x, layer) => layer(x), x);\nconst scores = outputHead(finalNorm(h));",
  children: [wordTable, positionTable, gpt2Layers, finalNorm, outputHead],
};

export const GPT_2_SMALL: RealModel = {
  id: "gpt-2-small",
  title: "GPT-2 small",
  root,
  modes: {
    use: inferenceRoot({
      modelTitle: "GPT-2 small",
      tokenizer: "byte-level BPE",
      wordCount: WORD_COUNT,
      width: WIDTH,
      context: CONTEXT,
      kvWidth: WIDTH,
      sources: GPT_2_SETTINGS,
      textSources: ["gpt-2-paper"],
      tokenizerNotes: [
        { label: "Kind", value: "byte-level BPE, starting from 256 single bytes", source: "gpt-2-paper" },
        { label: "Word list", value: "50,257 words", source: "gpt-2-paper" },
      ],
      merges: { value: 50_000, unit: "merge rules", isComputed: true },
      mergeNotes: [{ label: "Start", value: "256 single bytes, so any text can be split", source: "gpt-2-paper" }, GPT_2_MERGES],
      stopWord: {
        id: 50_256,
        isComputed: false,
        notes: [
          { label: "Released code", value: "never stops at <|endoftext|>; it writes the full length", source: "gpt-2-sample" },
          { label: "Stop word", value: "<|endoftext|>, ID 50,256 (config eos_token_id)", source: "gpt-2-config" },
        ],
      },
      writingNotes: GPT_2_WRITING_NOTES,
      reasons: GPT_2_USE_REASONS,
      embeddingText:
        "Each word ID picks its row of 768 numbers from the word table. GPT-2 then adds a learned row for the word's place, 1 to 1,024, from the position table, so the same word at a different place gets different numbers.",
      embedding: [wordTable, positionTable],
      layers: gpt2Layers,
      finalNorm,
      outputHead,
    }),
    build: lifecycleRoot(root, GPT_2_BUILD),
  },
};

import type { ModelPart, RealModel } from "@/features/inside-real-models/interfaces/model-part.interface";
import { inferenceRoot } from "./inference-steps";
import { lifecycleRoot } from "./lifecycle-steps";
import { LLAMA_3_BUILD } from "./llama-3-build-facts";
import { llama3Layers } from "./llama-3-layer";
import { LLAMA_3_AUTOREGRESSIVE, LLAMA_3_FINAL_NORM, LLAMA_3_NO_REASON, LLAMA_3_OUTPUT, LLAMA_3_PARTS_UNEXPLAINED, LLAMA_3_RMS_NORM, LLAMA_3_ROPE, LLAMA_3_TRANSFORMER, LLAMA_3_USE_REASONS, LLAMA_3_WORD_TABLE, LLAMA_3_WRITING_NOTES } from "./llama-3-reasons";
import { normPart, projectionPart, tablePart } from "./part-builders";
import { LLAMA_3_SETTINGS } from "./sources";

const WIDTH = 4096;
const WORD_COUNT = 128_256;
const CONTEXT = 8192;
const KV_WIDTH = 1024;

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
  sources: LLAMA_3_SETTINGS,
  explanation: "One row of 4,096 numbers for each of the 128,256 words in Llama 3's list. A word's id picks its row.",
  why: "Each word becomes a point in a space with 4,096 directions. Only training decides where each point sits.",
  story: {
    problem: "The model computes with numbers, but a word ID is only a place in a list. Without the word table there would be nothing to multiply and add.",
    demo: "tiny-part",
    when: "First thing, every time text is read: each word ID picks its row. Training sets every number in it.",
    feeds: "Each word's row goes straight into the first layer; word order is added later by RoPE.",
    nextId: "word-table-row",
  },
  reasons: [LLAMA_3_WORD_TABLE, LLAMA_3_NO_REASON],
  codeReasons: [LLAMA_3_WORD_TABLE],
});

const finalNorm = normPart({
  id: "final-norm",
  icon: "norm",
  title: "Final norm",
  summary: "Keeps the last word's numbers in a steady range before they are scored.",
  copies: 1,
  size: WIDTH,
  hasShift: false,
  tinyPart: "final-norm",
  sources: LLAMA_3_SETTINGS,
  explanation: "One last RMSNorm on the last word's 4,096 numbers, with its own 4,096 scale numbers, before they are scored.",
  why: "Each norm sits before its part, so the very last layer's output still needs one.",
  story: {
    problem: "Each norm sits before its part, so the last layer's output has not been steadied yet. Without this one, the scores would come from numbers of any size.",
    demo: "tiny-part",
    when: "Once, after the 32nd layer, on the last word's numbers, each time a new word is guessed.",
    feeds: "The steadied numbers are scored against every word by the output table.",
    nextId: "output-head",
  },
  reasons: [LLAMA_3_FINAL_NORM, LLAMA_3_RMS_NORM],
  codeReasons: [LLAMA_3_RMS_NORM],
});

const outputHead = projectionPart({
  id: "output-head",
  icon: "scores",
  title: "Output table",
  summary: "Scores every word in the list against the last word's numbers.",
  copies: 1,
  inSize: WIDTH,
  outSize: WORD_COUNT,
  hasBias: false,
  input: "4,096 for last word",
  output: "128,256 scores",
  inputDetail: "4,096 numbers for the last word",
  outputDetail: "128,256 scores, one per word",
  tinyPart: "output-table",
  sources: LLAMA_3_SETTINGS,
  explanation: "Turns the last word's 4,096 numbers into one score per word. Llama 3 8B stores its own output table, separate from the word table, so it adds another 525,336,576 numbers.",
  why: "Softmax turns the scores into chances, and one word is picked. The top chance is not knowledge, only the best guess the numbers give.",
  story: {
    problem: "The layers end with 4,096 numbers, but the model has to say something about each of its 128,256 words. Without the output table there would be no score to turn into a chance.",
    demo: "tiny-part",
    when: "Once for each new word, at the very end of the pass.",
    feeds: "The 128,256 scores go to softmax, which turns them into chances, and one word is picked.",
    nextId: "output-head-number",
  },
  reasons: [LLAMA_3_OUTPUT],
  codeReasons: [LLAMA_3_OUTPUT],
});

const root: ModelPart = {
  id: "model",
  icon: "model",
  title: "Llama 3 8B",
  summary: "The whole model: word IDs go in, and a score for every word comes out.",
  copies: 1,
  facts: {
    shape: "32 layers × 4,096 numbers per word",
    weightCount: 8_030_261_248,
    headline: { value: 8_030_261_248, unit: "numbers", isComputed: true },
    input: "up to 8,192 word IDs",
    output: "128,256 scores",
    outputDetail: "128,256 scores, one per word it knows",
    sources: [...LLAMA_3_SETTINGS, "llama-3-weights", "llama-3-card"],
    notes: [
      { label: "Released", value: "April 18, 2024", source: "llama-3-card" },
      { label: "Size", value: "8,030,261,248, worked out from Meta's settings; Hugging Face lists the same count", source: "llama-3-weights" },
      { label: "Text length", value: "8,192 words (tokens)", source: "llama-3-card" },
    ],
    reasons: [LLAMA_3_TRANSFORMER, LLAMA_3_AUTOREGRESSIVE, LLAMA_3_PARTS_UNEXPLAINED],
  },
  explanation:
    "The whole model, released by Meta in 2024. Words go in as ids, and out comes one score for every word in its list. It has no position table: word order is added inside attention by RoPE. Everything inside is stored numbers, multiplied and added.",
  why: "Its answers look like understanding, but every step is arithmetic on 8,030,261,248 stored numbers. Take it apart to see.",
  story: {
    problem: "The parts only work in one order: rows in, 32 layers, a norm, scores out. Without the whole model wired together, no part would get its input or hand on its output.",
    demo: "architecture",
    when: "You write the whole model once, at the Architecture stage, then train it, then use it to write text.",
    feeds: "Its scores, turned into chances, are what picks each next word when the model writes.",
    nextId: "word-table",
  },
  formula: { lines: ["x = word rows", "chances = softmax(head(norm(layers(x))))"] },
  code: "const x = wordRows(ids);\nconst h = layers.reduce((x, layer) => layer(x), x);\nconst scores = outputHead(finalNorm(h));",
  children: [wordTable, llama3Layers, finalNorm, outputHead],
};

const wordOrder: ModelPart = {
  id: "word-order",
  icon: "position",
  title: "Word order comes later",
  summary: "Adds nothing here, because Llama marks word order later, inside attention.",
  copies: 1,
  facts: { shape: "nothing stored here", weightCount: 0, headline: { value: 0, unit: "numbers added here" }, input: "word vectors", output: "the same vectors", sources: LLAMA_3_SETTINGS, notes: [{ label: "Order comes in", value: "inside attention, by RoPE with θ 500,000", source: "llama-3-settings" }], reasons: [LLAMA_3_ROPE, LLAMA_3_NO_REASON] },
  tiny: { part: "position-table", level: "whole" },
  explanation:
    "Nothing is added here: Llama 3 has no position table. Word order comes in later through RoPE, inside each attention head of step 3, which turns every query and key pair by an angle that grows with the word's place.",
  why: "Turning queries and keys makes their match depend on how far apart two words are, and adds no stored numbers.",
  story: {
    problem: "Attention looks at all the words at once, so without some sign of order \"dog bites man\" and \"man bites dog\" would look the same. Llama 3 gives that sign later, by turning queries and keys, so nothing is added here.",
    demo: "tiny-part",
    when: "Every time a message is read. Nothing happens here; the turning happens in every attention head, for every word.",
    feeds: "The word rows go into layer 1 unchanged, and RoPE turns their queries and keys inside attention.",
    nextId: "layers",
  },
  formula: { lines: ["vector = word table[id]", "place: added later by RoPE"] },
  groundZero: "action",
};

export const LLAMA_3_8B: RealModel = {
  id: "llama-3-8b",
  title: "Llama 3 8B",
  root,
  modes: {
    use: inferenceRoot({
      modelTitle: "Llama 3 8B",
      tokenizer: "tiktoken-based BPE",
      wordCount: WORD_COUNT,
      width: WIDTH,
      context: CONTEXT,
      kvWidth: KV_WIDTH,
      sources: LLAMA_3_SETTINGS,
      textSources: ["llama-3-card"],
      tokenizerNotes: [
        { label: "Fewer pieces", value: "up to 15% fewer pieces than Llama 2 for the same text", source: "llama-3-blog" },
        { label: "Word list", value: "128K words, 128,256 exactly", source: "llama-3-settings" },
        { label: "Built on", value: "OpenAI's tiktoken library", source: "llama-3-tokenizer" },
      ],
      merges: { text: "Not published" },
      reasons: LLAMA_3_USE_REASONS,
      writingNotes: LLAMA_3_WRITING_NOTES,
      mergeNotes: [{ label: "Merge rules", value: "Not published; the rules ship inside the tokenizer file, not in a paper", source: "llama-3-tokenizer" }],
      stopWord: {
        id: 128_001,
        isComputed: true,
        notes: [
          { label: "Instruct model", value: "also stops at <|eot_id|>, the end of a turn, ID 128,009", source: "llama-3-tokenizer" },
          { label: "Stop word", value: "<|end_of_text|>, ID 128,001: the 2nd special word after the 128,000 regular ones", source: "llama-3-tokenizer" },
        ],
      },
      embeddingText:
        "Each word ID picks its row of 4,096 numbers from the word table. Llama 3 adds no place numbers here: word order is added later by RoPE, inside attention.",
      embedding: [wordTable, wordOrder],
      layers: llama3Layers,
      finalNorm,
      outputHead,
    }),
    build: lifecycleRoot(root, LLAMA_3_BUILD),
  },
};

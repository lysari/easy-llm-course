import type { ModelPart, RealModel } from "@/features/take-apart/interfaces/model-part.interface";
import { inferenceRoot } from "./inference-steps";
import { lifecycleRoot } from "./lifecycle-steps";
import { llama3Layers } from "./llama-3-layer";
import { normPart, projectionPart, tablePart } from "./part-builders";
import { LLAMA_3_BLOG, LLAMA_3_CARD, LLAMA_3_PAPER } from "./sources";

const WIDTH = 4096;
const WORD_COUNT = 128_256;
const CONTEXT = 8192;
const KV_WIDTH = 1024;

const wordTable = tablePart({
  id: "word-table",
  icon: "grid",
  title: "Word table",
  copies: 1,
  rows: WORD_COUNT,
  cols: WIDTH,
  rowName: "word",
  tinyPart: "word-table",
  lesson: "Lessons 10, 18",
  reverseLesson: "RL-02",
  explanation: "One row of 4,096 numbers for each of the 128,256 words in Llama 3's list. A word's id picks its row.",
  why: "This is where \"meaning\" lives: a point in a space with 4,096 directions. Words used in similar places end up near each other.",
});

const finalNorm = normPart({
  id: "final-norm",
  icon: "gauge",
  title: "Final norm",
  copies: 1,
  size: WIDTH,
  hasShift: false,
  tinyPart: "final-norm",
  missing: "Our tiny model uses a plain LayerNorm with no stored numbers here.",
  lesson: "Lesson 20",
  reverseLesson: "RL-04",
  explanation: "One last RMSNorm on the last word's 4,096 numbers, with its own 4,096 scale numbers, before they are scored.",
  why: "Each norm sits before its part, so the very last layer's output still needs one.",
});

const outputHead = projectionPart({
  id: "output-head",
  icon: "percent",
  title: "Output table",
  copies: 1,
  inSize: WIDTH,
  outSize: WORD_COUNT,
  hasBias: false,
  input: "4,096 numbers for the last word",
  output: "128,256 scores, one per word",
  tinyPart: "output-table",
  lesson: "Lessons 11, 20",
  reverseLesson: "RL-04",
  explanation: "Turns the last word's 4,096 numbers into one score per word. Llama 3 8B stores its own output table, separate from the word table, so it adds another 525,336,576 numbers.",
  why: "Softmax turns the scores into chances, and one word is picked. The top chance is not knowledge, only the best guess the numbers give.",
});

const root: ModelPart = {
  id: "model",
  icon: "network",
  title: "Llama 3 8B",
  copies: 1,
  facts: { shape: "32 layers × 4,096 numbers per word", weightCount: 8_030_261_248, input: "up to 8,192 word ids", output: "128,256 scores, one per word it knows" },
  tiny: { part: "model", level: "whole" },
  lesson: "Lessons 25, 30",
  reverseLesson: "RL-00",
  explanation:
    "The whole model, released by Meta in 2024. Words go in as ids, and out comes one score for every word in its list. It has no position table: word order is added inside attention by RoPE. Everything inside is stored numbers, multiplied and added.",
  why: "Its answers look like understanding, but every step is arithmetic on 8,030,261,248 stored numbers. Take it apart to see.",
  formula: { lines: ["x = word rows", "chances = softmax(head(norm(layers(x))))"] },
  code: "const x = wordRows(ids);\nconst h = layers.reduce((x, layer) => layer(x), x);\nconst scores = outputHead(finalNorm(h));",
  children: [wordTable, llama3Layers, finalNorm, outputHead],
};

const wordOrder: ModelPart = {
  id: "word-order",
  icon: "order",
  title: "Word order comes later",
  copies: 1,
  facts: { shape: "nothing stored here", weightCount: 0, input: "word vectors", output: "the same vectors" },
  tiny: { part: "position-table", level: "whole", missing: "Our tiny model adds a learned position table here instead, like GPT-2." },
  lesson: "Lesson 25",
  reverseLesson: "RL-02",
  explanation:
    "Ground zero. Nothing is added here: Llama 3 has no position table. Word order comes in later through RoPE, inside each attention head of step 3, which turns every query and key pair by an angle that grows with the word's place.",
  why: "Turning queries and keys makes their match depend on how far apart two words are, and adds no stored numbers.",
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
      tokenizer: "BPE",
      wordCount: WORD_COUNT,
      width: WIDTH,
      context: CONTEXT,
      kvWidth: KV_WIDTH,
      tokenizerNotes: [
        { label: "Word list", value: "128K words: 100K from OpenAI's tiktoken list plus 28K more", source: LLAMA_3_PAPER },
        { label: "Fewer pieces", value: "3.94 English characters per piece, up from 3.17 with Llama 2's list", source: LLAMA_3_PAPER },
      ],
      embeddingLesson: "Lessons 10, 25",
      embeddingText:
        "Each word ID picks its row of 4,096 numbers from the word table. Llama 3 adds no place numbers here: word order is added later by RoPE, inside attention.",
      embedding: [wordTable, wordOrder],
      layers: llama3Layers,
      finalNorm,
      outputHead,
    }),
    build: lifecycleRoot({
      model: root,
      data: {
        shape: "over 15 trillion words, public sources",
        notes: [
          { label: "Size", value: "over 15 trillion words (tokens), seven times Llama 2's", source: LLAMA_3_BLOG },
          { label: "Mix", value: "four times more code than Llama 2; over 5% in more than 30 other languages", source: LLAMA_3_BLOG },
          { label: "Cleaning", value: "simple rules, adult-content filters, removing near copies, and text-quality scoring models", source: LLAMA_3_BLOG },
          { label: "Copies removed", value: "by web address, by whole document (MinHash), and line by line", source: LLAMA_3_PAPER },
          { label: "Newest text", value: "March 2023", source: LLAMA_3_CARD },
        ],
      },
      tokenizer: {
        shape: "BPE · 128K words",
        notes: [
          { label: "Word list", value: "128K words: 100K from OpenAI's tiktoken list plus 28K more for other languages", source: LLAMA_3_PAPER },
          { label: "Fewer pieces", value: "3.94 English characters per piece, up from 3.17 with Llama 2's list", source: LLAMA_3_PAPER },
        ],
      },
      preTraining: {
        shape: "over 15 trillion words, 8,192 at a time",
        notes: [
          { label: "Words trained on", value: "over 15 trillion", source: LLAMA_3_CARD },
          { label: "Text length", value: "8,192 words per text", source: LLAMA_3_CARD },
          { label: "Computer time", value: "1.3 million GPU hours on H100-80GB chips", source: LLAMA_3_CARD },
        ],
      },
      postTraining: {
        shape: "SFT, rejection sampling, PPO and DPO",
        notes: [
          { label: "Methods", value: "SFT, rejection sampling, PPO and DPO, for the Instruct version", source: LLAMA_3_BLOG },
          { label: "Example answers", value: "public instruction sets and over 10 million examples marked by people", source: LLAMA_3_CARD },
          { label: "Two versions", value: "a pre-trained base model and an instruction-tuned Instruct model", source: LLAMA_3_CARD },
        ],
      },
      benchmarks: {
        shape: "MMLU 66.6 (base), 68.4 (Instruct)",
        notes: [
          { label: "MMLU, 5 examples", value: "66.6 base, 68.4 Instruct", source: LLAMA_3_CARD },
          { label: "More tests", value: "ARC-Challenge 78.6, Winogrande 76.1 (base); GSM-8K 79.6, HumanEval 62.2 (Instruct)", source: LLAMA_3_CARD },
        ],
      },
      safety: {
        shape: "red teaming, Llama Guard 2",
        notes: [
          { label: "Red teaming", value: "people and programs wrote attack prompts to test the Instruct model", source: LLAMA_3_BLOG },
          { label: "Tools", value: "Llama Guard 2, Code Shield and CyberSec Eval 2", source: LLAMA_3_BLOG },
        ],
      },
      quantization: {
        shape: "not covered for 8B",
        notes: [
          { label: "8B", value: "not covered in the model card", source: LLAMA_3_CARD },
          { label: "Largest model", value: "8-bit (FP8) numbers tested for running the 405B model", source: LLAMA_3_PAPER },
        ],
      },
    }),
  },
};

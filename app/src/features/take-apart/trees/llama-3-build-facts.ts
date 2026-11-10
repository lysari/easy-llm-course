import type { BuildFacts, RealNote, StepFacts } from "@/features/take-apart/interfaces/real-facts.interface";

const NOT_PUBLISHED = { text: "Not published" };

const notPublished = (label: string, shape: string): StepFacts => ({
  shape,
  headline: NOT_PUBLISHED,
  sources: ["llama-3-card", "llama-3-blog"],
  notes: [{ label, value: "Not published for Llama 3 8B in the model card or Meta's blog", source: "llama-3-card" }],
});

const WORD_LIST: RealNote[] = [
  { label: "Word list", value: "128K words, 128,256 exactly", source: "llama-3-settings" },
  { label: "Special words", value: "256 kept for markers such as end of text, so 128,000 regular words (worked out)", source: "llama-3-tokenizer" },
];

const METHODS: RealNote = { label: "Methods", value: "SFT, rejection sampling, PPO and DPO, for the Instruct version", source: "llama-3-blog" };

const MMLU: RealNote = { label: "MMLU, 5 examples first", value: "66.6 base, 68.4 Instruct", source: "llama-3-card" };

export const LLAMA_3_BUILD: BuildFacts = {
  data: {
    shape: "over 15 trillion words (tokens), public sources",
    headline: { value: 15_000_000_000_000, unit: "words", isOver: true },
    sources: ["llama-3-card", "llama-3-blog"],
    notes: [
      { label: "Size", value: "over 15 trillion words (tokens), all from publicly available sources", source: "llama-3-blog" },
      { label: "Compared with Llama 2", value: "seven times more data, and four times more code", source: "llama-3-blog" },
      { label: "Languages", value: "over 5% is non-English, covering over 30 languages", source: "llama-3-blog" },
      { label: "Newest text", value: "March 2023, for the 8B model", source: "llama-3-card" },
      { label: "Meta user data", value: "none", source: "llama-3-card" },
    ],
  },
  gather: {
    shape: "number of documents not published",
    headline: NOT_PUBLISHED,
    sources: ["llama-3-card"],
    notes: [
      { label: "From", value: "\"a new mix of publicly available online data\"", source: "llama-3-card" },
      { label: "Documents and how text was copied", value: "Not published", source: "llama-3-card" },
    ],
    reasons: [
      { label: "Meta's reason", value: "\"the curation of a large, high-quality training dataset is paramount\"", source: "llama-3-blog" },
      { label: "How text was copied out of pages", value: "Not published. The model card and Meta's blog do not say.", source: "llama-3-card" },
    ],
  },
  filter: {
    shape: "filter rules not published",
    headline: NOT_PUBLISHED,
    sources: ["llama-3-blog"],
    notes: [
      { label: "Filters", value: "simple rules (heuristic filters), adult-content (NSFW) filters and text-quality classifiers", source: "llama-3-blog" },
      { label: "Quality scorers", value: "Llama 2 wrote the training data for the text-quality classifiers", source: "llama-3-blog" },
      { label: "Number of rules", value: "Not published", source: "llama-3-blog" },
    ],
  },
  dedupe: {
    shape: "semantic deduplication",
    headline: NOT_PUBLISHED,
    sources: ["llama-3-blog"],
    notes: [
      { label: "Method", value: "semantic deduplication", source: "llama-3-blog" },
      { label: "Copies removed", value: "Not published", source: "llama-3-blog" },
    ],
  },
  tokenizer: {
    shape: "tiktoken-based BPE · 128,256 words",
    headline: { value: 128_256, unit: "words" },
    sources: ["llama-3-card", "llama-3-blog", "llama-3-settings", "llama-3-tokenizer"],
    notes: [
      ...WORD_LIST,
      { label: "Built on", value: "OpenAI's tiktoken library", source: "llama-3-tokenizer" },
      { label: "Fewer pieces", value: "up to 15% fewer pieces than Llama 2 for the same text", source: "llama-3-blog" },
    ],
  },
  "merge-round": notPublished("Merge rounds", "merge rounds not published"),
  "count-pair": notPublished("Pair counts", "pair counts not published"),
  "learn-merge": notPublished("Merge rules", "merge rules not published"),
  "assign-id": {
    shape: "IDs 0 to 128,255",
    headline: { value: 128_256, unit: "IDs given" },
    sources: ["llama-3-settings", "llama-3-tokenizer"],
    notes: WORD_LIST,
  },
  "pre-training": {
    shape: "over 15 trillion words, texts of 8,192",
    headline: { value: 15_000_000_000_000, unit: "words trained on", isOver: true },
    sources: ["llama-3-card", "llama-3-blog"],
    notes: [
      { label: "Words trained on", value: "over 15 trillion (tokens)", source: "llama-3-card" },
      { label: "Text length", value: "8,192 words (tokens)", source: "llama-3-card" },
      { label: "Documents kept apart", value: "a mask stops attention crossing from one document to the next", source: "llama-3-blog" },
      { label: "Computer time", value: "1.3 million GPU hours on H100-80GB chips", source: "llama-3-card" },
      { label: "Carbon", value: "390 tonnes CO2eq, offset by Meta", source: "llama-3-card" },
    ],
  },
  "training-round": notPublished("Batch size and rounds", "batch size not published"),
  guess: {
    shape: "1 chance out of 128,256",
    headline: { value: 128_256, unit: "words to choose from" },
    sources: ["llama-3-settings"],
    notes: [{ label: "Word list", value: "128,256 words (vocab_size)", source: "llama-3-settings" }],
  },
  "loss-term": notPublished("Training loss", "loss values not published"),
  slope: {
    shape: "one slope per stored number",
    headline: { value: 8_030_261_248, unit: "slopes per round", isComputed: true },
    sources: ["llama-3-settings", "llama-3-weights"],
    notes: [{ label: "Slopes", value: "one for each of the 8,030,261,248 stored numbers", source: "llama-3-weights" }],
  },
  "adam-update": notPublished("Optimizer", "optimizer not published"),
  "post-training": {
    shape: "SFT, rejection sampling, PPO and DPO",
    headline: { value: 10_000_000, unit: "human-marked examples", isOver: true },
    sources: ["llama-3-card", "llama-3-blog"],
    notes: [
      METHODS,
      { label: "Data", value: "public instruction sets and over 10 million examples marked by people", source: "llama-3-card" },
      { label: "Two versions", value: "a pre-trained base model and an instruction-tuned Instruct model", source: "llama-3-card" },
    ],
  },
  sft: {
    shape: "prompts with written answers",
    headline: NOT_PUBLISHED,
    sources: ["llama-3-blog", "llama-3-card"],
    notes: [METHODS, { label: "SFT examples", value: "Not published on their own; over 10 million human-marked examples in all", source: "llama-3-card" }],
  },
  "sft-term": notPublished("Answer-word loss", "not published"),
  preference: {
    shape: "preference rankings for PPO and DPO",
    headline: NOT_PUBLISHED,
    sources: ["llama-3-blog"],
    notes: [
      { label: "Used", value: "preference rankings, in PPO and DPO", source: "llama-3-blog" },
      { label: "Number of rankings", value: "Not published", source: "llama-3-blog" },
    ],
  },
  "preference-pair": notPublished("Answers per ranking", "not published"),
  evaluation: {
    shape: "MMLU 66.6 (base), 68.4 (Instruct)",
    headline: { value: 66.6, unit: "% right on MMLU" },
    sources: ["llama-3-card"],
    notes: [MMLU],
  },
  benchmarks: {
    shape: "MMLU 66.6 (base), 68.4 (Instruct)",
    headline: { value: 66.6, unit: "% right on MMLU" },
    sources: ["llama-3-card"],
    notes: [
      MMLU,
      { label: "Base model", value: "ARC-Challenge 78.6, Winogrande 76.1", source: "llama-3-card" },
      { label: "Instruct model", value: "GSM-8K 79.6, HumanEval 62.2", source: "llama-3-card" },
    ],
  },
  "test-question": {
    shape: "MMLU: 5 examples shown first",
    headline: { value: 5, unit: "examples shown first" },
    sources: ["llama-3-card"],
    notes: [MMLU],
  },
  safety: {
    shape: "red teaming, Llama Guard 2, Code Shield",
    headline: { value: 3, unit: "safety tools", isComputed: true },
    sources: ["llama-3-blog", "llama-3-card"],
    notes: [
      { label: "Red teaming", value: "people and programs wrote attack prompts to test the Instruct model", source: "llama-3-blog" },
      { label: "Tools", value: "Llama Guard 2, Code Shield and CyberSec Eval 2", source: "llama-3-blog" },
      { label: "Risk tests", value: "CBRNE weapons, cyber security and child safety", source: "llama-3-card" },
    ],
  },
  "safety-check": notPublished("Attack prompts", "number of prompts not published"),
  quantization: {
    shape: "released as 16-bit numbers",
    headline: NOT_PUBLISHED,
    sources: ["llama-3-weights", "llama-3-card", "llama-3-blog"],
    notes: [
      { label: "Released numbers", value: "bfloat16, 16 bits each", source: "llama-3-weights" },
      { label: "Quantization by Meta", value: "Not published in the model card or blog", source: "llama-3-card" },
    ],
  },
  "round-number": notPublished("Rounding", "no quantization published"),
};

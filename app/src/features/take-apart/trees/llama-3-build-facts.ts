import type { BuildFacts, RealNote, StepFacts } from "@/features/take-apart/interfaces/real-facts.interface";
import { LLAMA_3_FEWER_PIECES, LLAMA_3_RANKS_IN_FILE, LLAMA_3_SPECIAL_IDS, LLAMA_3_TOKENIZER } from "./llama-3-reasons";

const NOT_PUBLISHED = { text: "Not published" };

const notPublished = (label: string, shape: string, reasons: RealNote[]): StepFacts => ({
  shape,
  headline: NOT_PUBLISHED,
  sources: ["llama-3-card", "llama-3-blog"],
  notes: [{ label, value: "Not published for Llama 3 8B in the model card or Meta's blog", source: "llama-3-card" }],
  reasons,
});

const WORD_LIST: RealNote[] = [
  { label: "Special words", value: "256 kept for markers such as end of text, so 128,000 regular words (worked out)", source: "llama-3-tokenizer" },
  { label: "Word list", value: "128K words, 128,256 exactly", source: "llama-3-settings" },
];

const METHODS: RealNote = { label: "Methods", value: "SFT, rejection sampling, PPO and DPO, for the Instruct version", source: "llama-3-blog" };

const MMLU: RealNote = { label: "MMLU, 5 examples first", value: "66.6 base, 68.4 Instruct", source: "llama-3-card" };

const MERGES_NOT_PUBLISHED: RealNote = { label: "How they were learned", value: "Not published", source: "llama-3-card" };

const QUALITY_MATTERS: RealNote = { label: "What matters most", value: "\"The quality of the prompts that are used in SFT and the preference rankings that are used in PPO and DPO has an outsized influence on the performance of aligned models\"", source: "llama-3-blog" };

const NO_QUANTIZATION: RealNote = { label: "Quantization by Meta", value: "Not published in the model card or blog", source: "llama-3-card" };

export const LLAMA_3_BUILD: BuildFacts = {
  data: {
    shape: "over 15 trillion words (tokens), public sources",
    headline: { value: 15_000_000_000_000, unit: "words", isOver: true },
    sources: ["llama-3-card", "llama-3-blog"],
    notes: [
      { label: "Compared with Llama 2", value: "seven times more data, and four times more code", source: "llama-3-blog" },
      { label: "Size", value: "over 15 trillion words (tokens), all from publicly available sources", source: "llama-3-blog" },
      { label: "Languages", value: "over 5% is non-English, covering over 30 languages", source: "llama-3-blog" },
      { label: "Newest text", value: "March 2023, for the 8B model", source: "llama-3-card" },
      { label: "Meta user data", value: "none", source: "llama-3-card" },
    ],
    reasons: [{ label: "Meta's reason", value: "\"To train the best language model, the curation of a large, high-quality training dataset is paramount\"", source: "llama-3-blog" }, { label: "How much", value: "\"we invested heavily in pretraining data\"", source: "llama-3-blog" }],
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
    reasons: [{ label: "Meta's reason", value: "\"To ensure Llama 3 is trained on data of the highest quality, we developed a series of data-filtering pipelines\"", source: "llama-3-blog" }, { label: "Quality scorers", value: "\"we used Llama 2 to generate the training data for the text-quality classifiers that are powering Llama 3\"", source: "llama-3-blog" }, { label: "The rules", value: "Not published", source: "llama-3-blog" }],
  },
  dedupe: {
    shape: "semantic deduplication",
    headline: NOT_PUBLISHED,
    sources: ["llama-3-blog"],
    notes: [
      { label: "Method", value: "semantic deduplication", source: "llama-3-blog" },
      { label: "Copies removed", value: "Not published", source: "llama-3-blog" },
    ],
    reasons: [{ label: "What Meta used", value: "\"These pipelines include using heuristic filters, NSFW filters, semantic deduplication approaches, and text classifiers to predict data quality\"", source: "llama-3-blog" }, { label: "How it worked", value: "Not published", source: "llama-3-blog" }],
  },
  tokenizer: {
    shape: "tiktoken-based BPE · 128,256 words",
    headline: { value: 128_256, unit: "words" },
    sources: ["llama-3-card", "llama-3-blog", "llama-3-settings", "llama-3-tokenizer"],
    notes: [
      { label: "Fewer pieces", value: "up to 15% fewer pieces than Llama 2 for the same text", source: "llama-3-blog" },
      { label: "Built on", value: "OpenAI's tiktoken library", source: "llama-3-tokenizer" },
      ...WORD_LIST,
    ],
    reasons: [LLAMA_3_TOKENIZER, LLAMA_3_FEWER_PIECES],
  },
  "merge-round": notPublished("Merge rounds", "merge rounds not published", [LLAMA_3_RANKS_IN_FILE, MERGES_NOT_PUBLISHED]),
  "count-pair": notPublished("Pair counts", "pair counts not published", [LLAMA_3_RANKS_IN_FILE, MERGES_NOT_PUBLISHED]),
  "learn-merge": notPublished("Merge rules", "merge rules not published", [LLAMA_3_RANKS_IN_FILE, MERGES_NOT_PUBLISHED]),
  "assign-id": {
    shape: "IDs 0 to 128,255",
    headline: { value: 128_256, unit: "IDs given" },
    sources: ["llama-3-settings", "llama-3-tokenizer"],
    notes: WORD_LIST,
    reasons: [LLAMA_3_SPECIAL_IDS],
  },
  "pre-training": {
    shape: "over 15 trillion words, texts of 8,192",
    headline: { value: 15_000_000_000_000, unit: "words trained on", isOver: true },
    sources: ["llama-3-card", "llama-3-blog"],
    notes: [
      { label: "Computer time", value: "1.3 million GPU hours on H100-80GB chips", source: "llama-3-card" },
      { label: "Words trained on", value: "over 15 trillion (tokens)", source: "llama-3-card" },
      { label: "Text length", value: "8,192 words (tokens)", source: "llama-3-card" },
      { label: "Documents kept apart", value: "a mask stops attention crossing from one document to the next", source: "llama-3-blog" },
      { label: "Carbon", value: "390 tonnes CO2eq, offset by Meta", source: "llama-3-card" },
    ],
    reasons: [{ label: "Meta's reason", value: "\"Both our 8B and 70B parameter models continued to improve log-linearly after we trained them on up to 15T tokens\"", source: "llama-3-blog" }, { label: "Far past the usual", value: "\"while the Chinchilla-optimal amount of training compute for an 8B parameter model corresponds to ~200B tokens, we found that model performance continues to improve even after the model is trained on two orders of magnitude more data\"", source: "llama-3-blog" }],
  },
  "training-round": notPublished("Batch size and rounds", "batch size not published", [{ label: "Text per row", value: "\"We trained the models on sequences of 8,192 tokens, using a mask to ensure self-attention does not cross document boundaries\"", source: "llama-3-blog" }, { label: "Batch size and rounds", value: "Not published", source: "llama-3-card" }]),
  guess: {
    shape: "1 chance out of 128,256",
    headline: { value: 128_256, unit: "words to choose from" },
    sources: ["llama-3-settings"],
    notes: [{ label: "Looks back on", value: "up to 8,192 earlier words (tokens)", source: "llama-3-card" }, { label: "Word list", value: "128,256 words (vocab_size)", source: "llama-3-settings" }],
    reasons: [{ label: "What it is", value: "\"Llama 3 is an auto-regressive language model\"", source: "llama-3-card" }],
  },
  "loss-term": notPublished("Training loss", "loss values not published", [{ label: "Planning ahead", value: "\"scaling laws allow us to predict the performance of our largest models on key tasks\"", source: "llama-3-blog" }, { label: "Loss values", value: "Not published", source: "llama-3-card" }]),
  slope: {
    shape: "one slope per stored number",
    headline: { value: 8_030_261_248, unit: "slopes per round", isComputed: true },
    sources: ["llama-3-settings", "llama-3-weights"],
    notes: [{ label: "Training code", value: "Not published; Meta's released code only runs the model to write text", source: "llama-3-code" }, { label: "Slopes", value: "one for each of the 8,030,261,248 stored numbers", source: "llama-3-weights" }],
    reasons: [{ label: "Training code", value: "Not published. Meta's model code, llama/model.py, runs under torch.inference_mode() to write text, not to train.", source: "llama-3-code" }],
  },
  "adam-update": notPublished("Optimizer", "optimizer not published", [{ label: "Optimizer", value: "Not published for Llama 3 8B in the model card or Meta's blog", source: "llama-3-card" }]),
  "post-training": {
    shape: "SFT, rejection sampling, PPO and DPO",
    headline: { value: 10_000_000, unit: "human-marked examples", isOver: true },
    sources: ["llama-3-card", "llama-3-blog"],
    notes: [
      METHODS,
      { label: "Data", value: "public instruction sets and over 10 million examples marked by people", source: "llama-3-card" },
      { label: "Two versions", value: "a pre-trained base model and an instruction-tuned Instruct model", source: "llama-3-card" },
    ],
    reasons: [{ label: "Meta's reason", value: "\"To fully unlock the potential of our pretrained models in chat use cases, we innovated on our approach to instruction-tuning as well\"", source: "llama-3-blog" }, { label: "How", value: "\"Our approach to post-training is a combination of supervised fine-tuning (SFT), rejection sampling, proximal policy optimization (PPO), and direct preference optimization (DPO)\"", source: "llama-3-blog" }],
  },
  sft: {
    shape: "prompts with written answers",
    headline: NOT_PUBLISHED,
    sources: ["llama-3-blog", "llama-3-card"],
    notes: [METHODS, { label: "SFT examples", value: "Not published on their own; over 10 million human-marked examples in all", source: "llama-3-card" }],
    reasons: [QUALITY_MATTERS],
  },
  "sft-term": notPublished("Answer-word loss", "not published", [QUALITY_MATTERS, { label: "The answer-word loss", value: "Not published for Llama 3", source: "llama-3-card" }]),
  preference: {
    shape: "preference rankings for PPO and DPO",
    headline: NOT_PUBLISHED,
    sources: ["llama-3-blog"],
    notes: [
      { label: "Used", value: "preference rankings, in PPO and DPO", source: "llama-3-blog" },
      { label: "Number of rankings", value: "Not published", source: "llama-3-blog" },
    ],
    reasons: [{ label: "Meta's reason", value: "\"The model knows how to produce the right answer, but it does not know how to select it. Training on preference rankings enables the model to learn how to select it.\"", source: "llama-3-blog" }],
  },
  "preference-pair": notPublished("Answers per ranking", "not published", [{ label: "What it helped", value: "\"Learning from preference rankings via PPO and DPO also greatly improved the performance of Llama 3 on reasoning and coding tasks\"", source: "llama-3-blog" }, { label: "Rankings used", value: "Not published", source: "llama-3-blog" }]),
  evaluation: {
    shape: "MMLU 66.6 (base), 68.4 (Instruct)",
    headline: { value: 66.6, unit: "% right on MMLU" },
    sources: ["llama-3-card"],
    notes: [MMLU],
    reasons: [{ label: "Meta's reason", value: "\"we looked at model performance on standard benchmarks and also sought to optimize for performance for real-world scenarios\"", source: "llama-3-blog" }],
  },
  benchmarks: {
    shape: "MMLU 66.6 (base), 68.4 (Instruct)",
    headline: { value: 66.6, unit: "% right on MMLU" },
    sources: ["llama-3-card"],
    notes: [
      { label: "Base model", value: "ARC-Challenge 78.6, Winogrande 76.1", source: "llama-3-card" },
      MMLU,
      { label: "Instruct model", value: "GSM-8K 79.6, HumanEval 62.2", source: "llama-3-card" },
    ],
    reasons: [{ label: "Keeping tests apart", value: "\"To prevent accidental overfitting of our models on this evaluation set, even our own modeling teams do not have access to it\"", source: "llama-3-blog" }],
  },
  "test-question": {
    shape: "MMLU: 5 examples shown first",
    headline: { value: 5, unit: "examples shown first" },
    sources: ["llama-3-card"],
    notes: [MMLU],
    reasons: [{ label: "Meta's own test set", value: "\"This evaluation set contains 1,800 prompts that cover 12 key use cases\"", source: "llama-3-blog" }],
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
    reasons: [{ label: "Meta's reason", value: "\"we conducted extensive red teaming exercises, performed adversarial evaluations and implemented safety mitigations techniques to lower residual risks\"", source: "llama-3-card" }, { label: "How", value: "\"Our red teaming approach leverages human experts and automation methods to generate adversarial prompts that try to elicit problematic responses\"", source: "llama-3-blog" }],
  },
  "safety-check": notPublished("Attack prompts", "number of prompts not published", [{ label: "Why test", value: "\"Llama 3’s potential outputs cannot be predicted in advance\"", source: "llama-3-card" }, { label: "Attack prompts", value: "Not published", source: "llama-3-blog" }]),
  quantization: {
    shape: "released as 16-bit numbers",
    headline: NOT_PUBLISHED,
    sources: ["llama-3-weights", "llama-3-card", "llama-3-blog"],
    notes: [
      { label: "Released numbers", value: "bfloat16, 16 bits each", source: "llama-3-weights" },
      { label: "Quantization by Meta", value: "Not published in the model card or blog", source: "llama-3-card" },
    ],
    reasons: [NO_QUANTIZATION],
  },
  "round-number": notPublished("Rounding", "no quantization published", [NO_QUANTIZATION]),
};

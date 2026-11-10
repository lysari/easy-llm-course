import type { BuildFacts, StepFacts } from "@/features/take-apart/interfaces/real-facts.interface";

const NOT_PUBLISHED = { text: "Not published" };
const BASE_MODEL_ONLY = { text: "None: base model only" };

const notPublished = (label: string, shape: string): StepFacts => ({
  shape,
  headline: NOT_PUBLISHED,
  sources: ["gpt-2-paper"],
  notes: [{ label, value: "Not published. The paper and the model card do not give it.", source: "gpt-2-paper" }],
});

const noPostTraining = (label: string): StepFacts => ({
  shape: "none: GPT-2 is a base model",
  headline: BASE_MODEL_ONLY,
  sources: ["gpt-2-paper"],
  notes: [{ label, value: "None. The paper's results use no fine-tuning: \"No training or fine-tuning was performed\"", source: "gpt-2-paper" }],
});

export const GPT_2_MERGES = { label: "Merge rules", value: "50,000, counted line by line in OpenAI's vocab.bpe file", source: "gpt-2-merges" } as const;

export const GPT_2_BUILD: BuildFacts = {
  data: {
    shape: "WebText: slightly over 8 million documents, 40 GB",
    headline: { value: 8_000_000, unit: "documents", isOver: true },
    sources: ["gpt-2-paper", "gpt-2-card"],
    notes: [
      { label: "Where from", value: "outbound links from Reddit posts with at least 3 karma: 45 million links", source: "gpt-2-paper" },
      { label: "Kept", value: "slightly over 8 million documents, 40 GB of text, after de-duplication and simple rules", source: "gpt-2-paper" },
      { label: "Removed", value: "all Wikipedia documents, and links made after December 2017", source: "gpt-2-paper" },
      { label: "Newest text", value: "the end of 2017", source: "gpt-2-card" },
    ],
  },
  gather: {
    shape: "45 million links, text pulled out of each page",
    headline: { value: 45_000_000, unit: "links" },
    sources: ["gpt-2-paper"],
    notes: [
      { label: "Links", value: "45 million outbound links from Reddit posts with at least 3 karma", source: "gpt-2-paper" },
      { label: "Text pulled out with", value: "the Dragnet and Newspaper content extractors", source: "gpt-2-paper" },
    ],
  },
  filter: {
    shape: "Reddit karma of at least 3",
    headline: { value: 3, unit: "karma or more" },
    sources: ["gpt-2-paper", "gpt-2-card"],
    notes: [
      { label: "Main rule", value: "keep a link only if its Reddit post got at least 3 karma", source: "gpt-2-paper" },
      { label: "Blocklist", value: "subreddits with sexually explicit or offensive content were skipped", source: "gpt-2-card" },
      { label: "Removed", value: "all Wikipedia documents", source: "gpt-2-paper" },
    ],
  },
  dedupe: {
    shape: "method not published",
    headline: NOT_PUBLISHED,
    sources: ["gpt-2-paper"],
    notes: [{ label: "De-duplication", value: "done, but the paper does not say how", source: "gpt-2-paper" }],
  },
  tokenizer: {
    shape: "byte-level BPE · 50,257 words",
    headline: { value: 50_257, unit: "words" },
    sources: ["gpt-2-paper", "gpt-2-merges"],
    notes: [
      { label: "Start", value: "256 single bytes, so any text can be written", source: "gpt-2-paper" },
      { label: "Rule", value: "no merges across kinds of characters, such as letters and punctuation, except for spaces", source: "gpt-2-paper" },
      GPT_2_MERGES,
      { label: "Word list", value: "50,257 words", source: "gpt-2-paper" },
    ],
  },
  "merge-round": {
    shape: "50,000 rounds, one new piece each",
    headline: { value: 50_000, unit: "merge rounds", isComputed: true },
    sources: ["gpt-2-merges"],
    notes: [GPT_2_MERGES],
  },
  "count-pair": notPublished("Pair counts", "pair counts not published"),
  "learn-merge": {
    shape: "50,000 rules, in the order learned",
    headline: { value: 50_000, unit: "merge rules", isComputed: true },
    sources: ["gpt-2-merges"],
    notes: [GPT_2_MERGES, { label: "First rule", value: "\"Ġ\" + \"t\", a space then t (Ġ marks a space)", source: "gpt-2-merges" }],
  },
  "assign-id": {
    shape: "IDs 0 to 50,256",
    headline: { value: 50_257, unit: "IDs given" },
    sources: ["gpt-2-config"],
    notes: [{ label: "Last ID", value: "50,256, the end-of-text word (config eos_token_id)", source: "gpt-2-config" }],
  },
  "pre-training": {
    shape: "WebText, batches of 512 texts × 1,024 words",
    headline: { value: 1024, unit: "words per text" },
    sources: ["gpt-2-paper"],
    notes: [
      { label: "Batch", value: "512 texts", source: "gpt-2-paper" },
      { label: "Text length", value: "1,024 words (tokens)", source: "gpt-2-paper" },
      { label: "Step size", value: "tuned by hand for the best score on 5% of WebText held back", source: "gpt-2-paper" },
      { label: "Words trained on", value: "Not published", source: "gpt-2-paper" },
      { label: "Rounds and computers", value: "Not published", source: "gpt-2-paper" },
    ],
  },
  "training-round": {
    shape: "one batch of 512 texts",
    headline: { value: 512, unit: "texts per round" },
    sources: ["gpt-2-paper"],
    notes: [
      { label: "Batch", value: "512 texts of up to 1,024 words", source: "gpt-2-paper" },
      { label: "Number of rounds", value: "Not published", source: "gpt-2-paper" },
    ],
  },
  guess: {
    shape: "1 chance out of 50,257",
    headline: { value: 50_257, unit: "words to choose from" },
    sources: ["gpt-2-config"],
    notes: [{ label: "Word list", value: "50,257 words (config vocab_size)", source: "gpt-2-config" }],
  },
  "loss-term": notPublished("Training loss", "loss values not published"),
  slope: {
    shape: "one slope per stored number",
    headline: { value: 124_439_808, unit: "slopes per round", isComputed: true },
    sources: ["gpt-2-config", "gpt-2-code"],
    notes: [{ label: "Slopes", value: "one for each of the 124,439,808 stored numbers, worked out from config.json", source: "gpt-2-config" }],
  },
  "adam-update": notPublished("Optimizer", "optimizer not published"),
  "post-training": noPostTraining("Post-training"),
  sft: noPostTraining("Supervised fine-tuning"),
  "sft-term": noPostTraining("Answer-word loss"),
  preference: noPostTraining("Preference tuning"),
  "preference-pair": noPostTraining("Preference pairs"),
  evaluation: {
    shape: "8 language tests, with no extra training",
    headline: { value: 45.99, unit: "% right on LAMBADA" },
    sources: ["gpt-2-paper"],
    notes: [{ label: "Smallest size", value: "LAMBADA 45.99% right (the paper lists this size as 117M numbers)", source: "gpt-2-paper" }],
  },
  benchmarks: {
    shape: "8 language tests, with no extra training",
    headline: { value: 45.99, unit: "% right on LAMBADA" },
    sources: ["gpt-2-paper"],
    notes: [
      { label: "Tests", value: "LAMBADA, Children's Book Test, WikiText-2, Penn Treebank, enwik8, text8, WikiText-103, One Billion Word", source: "gpt-2-paper" },
      { label: "Smallest size", value: "LAMBADA 45.99% right (the paper lists this size as 117M numbers)", source: "gpt-2-paper" },
      { label: "Overlap check", value: "test sets share 1–6% of their 8-word runs with WebText, 3.2% on average", source: "gpt-2-paper" },
    ],
  },
  "test-question": {
    shape: "zero-shot: no examples shown first",
    headline: { value: 0, unit: "examples shown first" },
    sources: ["gpt-2-paper"],
    notes: [{ label: "Setting", value: "zero-shot, \"without any parameter or architecture modification\"", source: "gpt-2-paper" }],
  },
  safety: {
    shape: "no safety tests published",
    headline: NOT_PUBLISHED,
    sources: ["gpt-2-paper", "gpt-2-card"],
    notes: [
      { label: "Safety tests", value: "Not published", source: "gpt-2-paper" },
      { label: "Advice", value: "study biases before using it in systems that talk with people", source: "gpt-2-card" },
    ],
  },
  "safety-check": notPublished("Safety checks", "no safety checks published"),
  quantization: notPublished("Quantization", "no quantization published"),
  "round-number": notPublished("Rounding", "no quantization published"),
};

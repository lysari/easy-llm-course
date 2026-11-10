import type { BuildFacts, RealNote, StepFacts } from "@/features/inside-real-models/interfaces/real-facts.interface";
import { GPT_2_ANY_STRING, GPT_2_BYTES, GPT_2_LIST_SIZE, GPT_2_LOOKUP, GPT_2_MERGES, GPT_2_NEXT_WORD } from "./gpt-2-reasons";

const NOT_PUBLISHED = { text: "Not published" };
const BASE_MODEL_ONLY = { text: "None: base model only" };

const notPublished = (label: string, shape: string, reasons: RealNote[]): StepFacts => ({
  shape,
  headline: NOT_PUBLISHED,
  sources: ["gpt-2-paper"],
  notes: [{ label, value: "Not published. The paper and the model card do not give it.", source: "gpt-2-paper" }],
  reasons,
});

const noPostTraining = (label: string, reasons: RealNote[]): StepFacts => ({
  shape: "none: GPT-2 is a base model",
  headline: BASE_MODEL_ONLY,
  sources: ["gpt-2-paper"],
  notes: [{ label, value: "None. The paper's results use no fine-tuning: \"No training or fine-tuning was performed\"", source: "gpt-2-paper" }],
  reasons,
});

const GREEDY: RealNote = { label: "How BPE builds its list", value: "\"BPE using a greedy frequency based heuristic for building the token vocabulary\"", source: "gpt-2-paper" };

const NO_FINE_TUNING: RealNote = { label: "No post-training", value: "\"No training or fine-tuning was performed for any of these results\"", source: "gpt-2-paper" };

const FINE_TUNING_LATER: RealNote = { label: "Left for later", value: "\"it is not clear where the ceiling is with finetuning\"", source: "gpt-2-paper" };

const STEP_SIZE: RealNote = { label: "Step size", value: "\"The learning rate of each model was manually tuned for the best perplexity\", measured on 5% of WebText held back", source: "gpt-2-paper" };

const OVER_REPORTING: RealNote = { label: "Why copies matter", value: "near-duplicates between training and test data lead to \"an over-reporting of the generalization performance of machine learning systems\"", source: "gpt-2-paper" };

const BIAS_ADVICE: RealNote = { label: "OpenAI's advice", value: "\"we do not recommend that they be deployed into systems that interact with humans unless the deployers first carry out a study of biases relevant to the intended use-case\"", source: "gpt-2-card" };

const NO_QUANTIZATION: RealNote = { label: "Quantization", value: "Not published in the paper or the model card.", source: "gpt-2-paper" };

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
    reasons: [{ label: "OpenAI's reason", value: "\"Our approach motivates building as large and diverse a dataset as possible in order to collect natural language demonstrations of tasks in as varied of domains and contexts as possible\"", source: "gpt-2-paper" }, { label: "The problem with raw scrapes", value: "web scrapes such as Common Crawl \"have significant data quality issues\"", source: "gpt-2-paper" }],
  },
  gather: {
    shape: "45 million links, text pulled out of each page",
    headline: { value: 45_000_000, unit: "links" },
    sources: ["gpt-2-paper"],
    notes: [
      { label: "Text pulled out with", value: "the Dragnet and Newspaper content extractors", source: "gpt-2-paper" },
      { label: "Links", value: "45 million outbound links from Reddit posts with at least 3 karma", source: "gpt-2-paper" },
    ],
    reasons: [
      { label: "Raw web scrapes", value: "web scrapes such as Common Crawl \"have significant data quality issues\"", source: "gpt-2-paper" },
      { label: "Common Crawl pages", value: "many documents \"whose content are mostly unintelligible\" (Trinh & Le, 2018, quoted in the paper)", source: "gpt-2-paper" },
      { label: "What OpenAI did", value: "\"To extract the text from HTML responses we use a combination of the Dragnet … and Newspaper content extractors\"", source: "gpt-2-paper" },
    ],
  },
  filter: {
    shape: "Reddit karma of at least 3",
    headline: { value: 3, unit: "karma or more" },
    sources: ["gpt-2-paper", "gpt-2-card"],
    notes: [
      { label: "Blocklist", value: "subreddits with sexually explicit or offensive content were skipped", source: "gpt-2-card" },
      { label: "Main rule", value: "keep a link only if its Reddit post got at least 3 karma", source: "gpt-2-paper" },
      { label: "Removed", value: "all Wikipedia documents", source: "gpt-2-paper" },
    ],
    reasons: [{ label: "OpenAI's reason", value: "\"we only scraped web pages which have been curated/filtered by humans\"", source: "gpt-2-paper" }, { label: "Why karma", value: "\"This can be thought of as a heuristic indicator for whether other users found the link interesting, educational, or just funny\"", source: "gpt-2-paper" }, { label: "Why no Wikipedia", value: "\"it is a common data source for other datasets and could complicate analysis due to overlapping training data with test evaluation tasks\"", source: "gpt-2-paper" }],
  },
  dedupe: {
    shape: "method not published",
    headline: NOT_PUBLISHED,
    sources: ["gpt-2-paper"],
    notes: [{ label: "De-duplication", value: "done, but the paper does not say how", source: "gpt-2-paper" }],
    reasons: [{ label: "Done, method not published", value: "WebText was cut down \"after de-duplication and some heuristic based cleaning\"; how is not published", source: "gpt-2-paper" }, OVER_REPORTING, { label: "OpenAI's advice", value: "\"we recommend the use of n-gram overlap based de-duplication as an important verification step and sanity check during the creation of training and test splits for new NLP datasets\"", source: "gpt-2-paper" }],
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
    reasons: [{ label: "OpenAI's reason", value: "BPE \"is a practical middle ground between character and word level language modeling\"", source: "gpt-2-paper" }, GPT_2_BYTES, GPT_2_ANY_STRING],
  },
  "merge-round": {
    shape: "50,000 rounds, one new piece each",
    headline: { value: 50_000, unit: "merge rounds", isComputed: true },
    sources: ["gpt-2-merges"],
    notes: [{ label: "Start", value: "256 single bytes; each round adds one new piece", source: "gpt-2-paper" }, GPT_2_MERGES],
    reasons: [GREEDY],
  },
  "count-pair": notPublished("Pair counts", "pair counts not published", [GREEDY, { label: "Pair counts", value: "Not published", source: "gpt-2-paper" }]),
  "learn-merge": {
    shape: "50,000 rules, in the order learned",
    headline: { value: 50_000, unit: "merge rules", isComputed: true },
    sources: ["gpt-2-merges"],
    notes: [{ label: "First rule", value: "\"Ġ\" + \"t\", a space then t (Ġ marks a space)", source: "gpt-2-merges" }, GPT_2_MERGES],
    reasons: [{ label: "A problem OpenAI found", value: "\"We observed BPE including many versions of common words like dog since they occur in many variations such as dog. dog! dog?\"", source: "gpt-2-paper" }, { label: "Their fix", value: "\"we prevent BPE from merging across character categories for any byte sequence\"", source: "gpt-2-paper" }],
  },
  "assign-id": {
    shape: "IDs 0 to 50,256",
    headline: { value: 50_257, unit: "IDs given" },
    sources: ["gpt-2-config"],
    notes: [{ label: "Last ID", value: "50,256, the end-of-text word (config eos_token_id)", source: "gpt-2-config" }],
    reasons: [GPT_2_LOOKUP, GPT_2_LIST_SIZE],
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
    reasons: [{ label: "OpenAI's reason", value: "\"language models begin to learn these tasks without any explicit supervision when trained on a new dataset of millions of webpages called WebText\"", source: "gpt-2-paper" }, { label: "What is learned", value: "\"Language modeling is usually framed as unsupervised distribution estimation\"", source: "gpt-2-paper" }],
  },
  "training-round": {
    shape: "one batch of 512 texts",
    headline: { value: 512, unit: "texts per round" },
    sources: ["gpt-2-paper"],
    notes: [
      { label: "Number of rounds", value: "Not published", source: "gpt-2-paper" },
      { label: "Batch", value: "512 texts of up to 1,024 words", source: "gpt-2-paper" },
    ],
    reasons: [{ label: "Batch", value: "\"a larger batchsize of 512 is used\"", source: "gpt-2-paper" }, { label: "More rounds would help", value: "\"All models still underfit WebText and held-out perplexity has as of yet improved given more training time\"", source: "gpt-2-paper" }],
  },
  guess: {
    shape: "1 chance out of 50,257",
    headline: { value: 50_257, unit: "words to choose from" },
    sources: ["gpt-2-config"],
    notes: [{ label: "Looks back on", value: "up to 1,024 earlier words (tokens)", source: "gpt-2-paper" }, { label: "Word list", value: "50,257 words (config vocab_size)", source: "gpt-2-config" }],
    reasons: [GPT_2_NEXT_WORD],
  },
  "loss-term": notPublished("Training loss", "loss values not published", [STEP_SIZE, { label: "Loss values", value: "Not published", source: "gpt-2-paper" }]),
  slope: {
    shape: "one slope per stored number",
    headline: { value: 124_439_808, unit: "slopes per round", isComputed: true },
    sources: ["gpt-2-config", "gpt-2-code"],
    notes: [{ label: "Training code", value: "Not published; the released code only runs the model forward", source: "gpt-2-code" }, { label: "Slopes", value: "one for each of the 124,439,808 stored numbers, worked out from config.json", source: "gpt-2-config" }],
    reasons: [{ label: "Training code", value: "Not published. OpenAI's released code, src/model.py, only runs the model forward; no training code was released.", source: "gpt-2-code" }],
  },
  "adam-update": notPublished("Optimizer", "optimizer not published", [STEP_SIZE, { label: "Optimizer", value: "Not published", source: "gpt-2-paper" }]),
  "post-training": noPostTraining("Post-training", [NO_FINE_TUNING, FINE_TUNING_LATER]),
  sft: noPostTraining("Supervised fine-tuning", [NO_FINE_TUNING, FINE_TUNING_LATER]),
  "sft-term": noPostTraining("Answer-word loss", [NO_FINE_TUNING]),
  preference: noPostTraining("Preference tuning", [NO_FINE_TUNING]),
  "preference-pair": noPostTraining("Preference pairs", [NO_FINE_TUNING]),
  evaluation: {
    shape: "8 language tests, with no extra training",
    headline: { value: 45.99, unit: "% right on LAMBADA" },
    sources: ["gpt-2-paper"],
    notes: [{ label: "Setting", value: "zero-shot, \"without any parameter or architecture modification\"", source: "gpt-2-paper" }, { label: "Smallest size", value: "LAMBADA 45.99% right (the paper lists this size as 117M numbers)", source: "gpt-2-paper" }],
    reasons: [{ label: "OpenAI's reason", value: "\"We test whether this is the case by analyzing the performance of language models in a zero-shot setting on a wide variety of tasks\"", source: "gpt-2-paper" }],
  },
  benchmarks: {
    shape: "8 language tests, with no extra training",
    headline: { value: 45.99, unit: "% right on LAMBADA" },
    sources: ["gpt-2-paper"],
    notes: [
      { label: "Overlap check", value: "test sets share 1–6% of their 8-word runs with WebText, 3.2% on average", source: "gpt-2-paper" },
      { label: "Tests", value: "LAMBADA, Children's Book Test, WikiText-2, Penn Treebank, enwik8, text8, WikiText-103, One Billion Word", source: "gpt-2-paper" },
      { label: "Smallest size", value: "LAMBADA 45.99% right (the paper lists this size as 117M numbers)", source: "gpt-2-paper" },
    ],
    reasons: [{ label: "Overlap check", value: "\"Common LM datasets' test sets have between 1-6% overlap with WebText train, with an average of overlap of 3.2%\"", source: "gpt-2-paper" }, OVER_REPORTING],
  },
  "test-question": {
    shape: "zero-shot: no examples shown first",
    headline: { value: 0, unit: "examples shown first" },
    sources: ["gpt-2-paper"],
    notes: [{ label: "Setting", value: "zero-shot, \"without any parameter or architecture modification\"", source: "gpt-2-paper" }],
    reasons: [{ label: "Zero-shot", value: "tasks are done \"without any parameter or architecture modification\"", source: "gpt-2-paper" }],
  },
  safety: {
    shape: "no safety tests published",
    headline: NOT_PUBLISHED,
    sources: ["gpt-2-paper", "gpt-2-card"],
    notes: [
      { label: "Advice", value: "study biases before using it in systems that talk with people", source: "gpt-2-card" },
      { label: "Safety tests", value: "Not published", source: "gpt-2-paper" },
    ],
    reasons: [BIAS_ADVICE, { label: "Safety tests", value: "Not published", source: "gpt-2-paper" }],
  },
  "safety-check": notPublished("Safety checks", "no safety checks published", [BIAS_ADVICE]),
  quantization: notPublished("Quantization", "no quantization published", [NO_QUANTIZATION]),
  "round-number": notPublished("Rounding", "no quantization published", [NO_QUANTIZATION]),
};

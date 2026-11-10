export interface Source {
  title: string;
  url: string;
}

export const SOURCES = {
  "gpt-2-paper": {
    title: "Language Models are Unsupervised Multitask Learners (OpenAI, 2019)",
    url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
  },
  "gpt-2-card": { title: "GPT-2 model card (OpenAI, 2019)", url: "https://github.com/openai/gpt-2/blob/master/model_card.md" },
  "gpt-2-config": { title: "GPT-2 config.json (openai-community/gpt2)", url: "https://huggingface.co/openai-community/gpt2/blob/main/config.json" },
  "gpt-2-code": { title: "GPT-2 model code, src/model.py (openai/gpt-2)", url: "https://github.com/openai/gpt-2/blob/master/src/model.py" },
  "gpt-2-encoder": { title: "GPT-2 tokenizer code, src/encoder.py (openai/gpt-2)", url: "https://github.com/openai/gpt-2/blob/master/src/encoder.py" },
  "gpt-2-sample": { title: "GPT-2 writing code, src/sample.py (openai/gpt-2)", url: "https://github.com/openai/gpt-2/blob/master/src/sample.py" },
  "gpt-2-interactive": { title: "GPT-2 writing settings, src/interactive_conditional_samples.py (openai/gpt-2)", url: "https://github.com/openai/gpt-2/blob/master/src/interactive_conditional_samples.py" },
  "gpt-2-merges": { title: "GPT-2 merge rules, vocab.bpe (OpenAI)", url: "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe" },
  "llama-3-card": { title: "Meta Llama 3 model card (2024)", url: "https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md" },
  "llama-3-blog": { title: "Introducing Meta Llama 3, Meta blog (2024)", url: "https://ai.meta.com/blog/meta-llama-3/" },
  "llama-3-settings": { title: "Llama 3 8B settings, sku_list.py (meta-llama/llama-models)", url: "https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py" },
  "llama-3-code": { title: "Llama 3 model code, llama/model.py (meta-llama/llama3)", url: "https://github.com/meta-llama/llama3/blob/main/llama/model.py" },
  "llama-3-tokenizer": { title: "Llama 3 tokenizer code, llama/tokenizer.py (meta-llama/llama3)", url: "https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py" },
  "llama-3-generation": { title: "Llama 3 writing code, llama/generation.py (meta-llama/llama3)", url: "https://github.com/meta-llama/llama3/blob/main/llama/generation.py" },
  "llama-3-text-example": { title: "Llama 3 base model example, example_text_completion.py (meta-llama/llama3)", url: "https://github.com/meta-llama/llama3/blob/main/example_text_completion.py" },
  "llama-3-chat-example": { title: "Llama 3 chat example, example_chat_completion.py (meta-llama/llama3)", url: "https://github.com/meta-llama/llama3/blob/main/example_chat_completion.py" },
  "llama-3-readme": { title: "Llama 3 README, Download (meta-llama/llama3)", url: "https://github.com/meta-llama/llama3/blob/main/README.md" },
  "llama-3-weights": { title: "Meta-Llama-3-8B weight count (Hugging Face)", url: "https://huggingface.co/api/models/meta-llama/Meta-Llama-3-8B?expand[]=safetensors" },
} satisfies Record<string, Source>;

export type SourceId = keyof typeof SOURCES;

export const GPT_2_SETTINGS: SourceId[] = ["gpt-2-config", "gpt-2-code"];

export const LLAMA_3_SETTINGS: SourceId[] = ["llama-3-settings", "llama-3-code"];

import type { WritingNotes } from "@/features/inside-real-models/interfaces/inference-spec.interface";
import type { RealNote, UseReasons } from "@/features/inside-real-models/interfaces/real-facts.interface";

export const LLAMA_3_TRANSFORMER: RealNote = { label: "Built on", value: "\"we opted for a relatively standard decoder-only transformer architecture in Llama 3\"", source: "llama-3-blog" };

export const LLAMA_3_NO_REASON: RealNote = { label: "Meta's reason", value: "Not published.", source: "llama-3-card" };

export const LLAMA_3_PARTS_UNEXPLAINED: RealNote = { label: "Meta's reasons for each part", value: "Not published. The model card and blog do not say why each part is built the way it is.", source: "llama-3-card" };

export const LLAMA_3_AUTOREGRESSIVE: RealNote = { label: "What it is", value: "\"Llama 3 is an auto-regressive language model that uses an optimized transformer architecture\"", source: "llama-3-card" };

export const LLAMA_3_LINEAR: RealNote = { label: "In Llama 3's code", value: "every table is a Linear layer with \"bias=False\", for example \"self.wq = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, bias=False, …)\"", source: "llama-3-code" };

export const LLAMA_3_QKV: RealNote = { label: "In Llama 3's code", value: "\"xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)\"", source: "llama-3-code" };

export const LLAMA_3_GQA: RealNote = { label: "Meta's reason", value: "\"To improve the inference efficiency of Llama 3 models, we’ve adopted grouped query attention (GQA) across both the 8B and 70B sizes\"", source: "llama-3-blog" };

export const LLAMA_3_REPEAT_KV: RealNote = { label: "In Llama 3's code", value: "\"# repeat k/v heads if n_kv_heads < n_heads\", then \"keys = repeat_kv(keys, self.n_rep)\"", source: "llama-3-code" };

export const LLAMA_3_ROPE: RealNote = { label: "In Llama 3's code", value: "\"xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)\", with angles from \"freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))\" and rope_theta 500,000", source: "llama-3-code" };

export const LLAMA_3_SCORES: RealNote = { label: "In Llama 3's code", value: "\"scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)\", then \"scores = F.softmax(scores.float(), dim=-1)\"", source: "llama-3-code" };

export const LLAMA_3_MASK: RealNote = { label: "Earlier words only", value: "\"mask = torch.triu(mask, diagonal=1)\" hides every later word", source: "llama-3-code" };

export const LLAMA_3_DOCUMENT_MASK: RealNote = { label: "In training", value: "\"using a mask to ensure self-attention does not cross document boundaries\"", source: "llama-3-blog" };

export const LLAMA_3_MIX: RealNote = { label: "In Llama 3's code", value: "\"output = torch.matmul(scores, values)\"", source: "llama-3-code" };

export const LLAMA_3_JOIN_HEADS: RealNote = { label: "In Llama 3's code", value: "\"return self.wo(output)\", after the heads are joined back into one row", source: "llama-3-code" };

export const LLAMA_3_RMS_NORM: RealNote = { label: "In Llama 3's code", value: "\"return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)\", times \"self.weight = nn.Parameter(torch.ones(dim))\"", source: "llama-3-code" };

export const LLAMA_3_ADD_BACK: RealNote = { label: "Adding back", value: "\"h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)\" and \"out = h + self.feed_forward(self.ffn_norm(h))\"", source: "llama-3-code" };

export const LLAMA_3_FEED_FORWARD: RealNote = { label: "In Llama 3's code", value: "\"return self.w2(F.silu(self.w1(x)) * self.w3(x))\"", source: "llama-3-code" };

export const LLAMA_3_WORD_TABLE: RealNote = { label: "In Llama 3's code", value: "\"self.tok_embeddings = VocabParallelEmbedding(params.vocab_size, params.dim, …)\" and \"h = self.tok_embeddings(tokens)\"", source: "llama-3-code" };

export const LLAMA_3_OUTPUT: RealNote = { label: "In Llama 3's code", value: "\"self.output = ColumnParallelLinear(params.dim, params.vocab_size, bias=False, …)\": its own table, not the word table", source: "llama-3-code" };

export const LLAMA_3_FINAL_NORM: RealNote = { label: "In Llama 3's code", value: "\"h = self.norm(h)\" runs after the last layer, before the scores", source: "llama-3-code" };

export const LLAMA_3_LAYER_LOOP: RealNote = { label: "In Llama 3's code", value: "\"for layer in self.layers: h = layer(h, start_pos, freqs_cis, mask)\", with 32 layers in sku_list.py", source: "llama-3-code" };

export const LLAMA_3_TOKENIZER: RealNote = { label: "Meta's reason", value: "\"Llama 3 uses a tokenizer with a vocabulary of 128K tokens that encodes language much more efficiently, which leads to substantially improved model performance\"", source: "llama-3-blog" };

export const LLAMA_3_FEWER_PIECES: RealNote = { label: "Fewer pieces", value: "\"yielding up to 15% fewer tokens compared to Llama 2\"", source: "llama-3-blog" };

export const LLAMA_3_RANKS_IN_FILE: RealNote = { label: "Where the rules are", value: "\"mergeable_ranks = load_tiktoken_bpe(model_path)\": the ranked merge rules ship inside the tokenizer file", source: "llama-3-tokenizer" };

export const LLAMA_3_SPECIAL_IDS: RealNote = { label: "In Llama 3's code", value: "\"token: num_base_tokens + i for i, token in enumerate(special_tokens)\": the 256 special words get the IDs after the regular ones", source: "llama-3-tokenizer" };

const LLAMA_3_SOFTMAX: RealNote = { label: "In Llama 3's code", value: "\"probs = torch.softmax(logits[:, -1] / temperature, dim=-1)\": the last word's scores, divided by the temperature, become chances", source: "llama-3-generation" };

export const LLAMA_3_USE_REASONS: UseReasons = {
  tokenize: [LLAMA_3_TOKENIZER, LLAMA_3_FEWER_PIECES],
  split: [{ label: "In Llama 3's code", value: "the split rule \"pat_str\" holds \"\\p{N}{1,3}\": numbers are cut into runs of at most 3 digits before any merge", source: "llama-3-tokenizer" }, LLAMA_3_NO_REASON],
  "apply-merge": [LLAMA_3_RANKS_IN_FILE, { label: "The rules", value: "Not published in a paper; they ship in the tokenizer file, which Meta gives out only after you accept its licence", source: "llama-3-readme" }],
  lookup: [{ label: "In Llama 3's code", value: "\"num_base_tokens = len(mergeable_ranks)\": the regular pieces come first in the list", source: "llama-3-tokenizer" }, LLAMA_3_SPECIAL_IDS],
  embedding: [LLAMA_3_WORD_TABLE, LLAMA_3_NO_REASON],
  output: [LLAMA_3_FINAL_NORM, LLAMA_3_OUTPUT],
  softmax: [LLAMA_3_SOFTMAX],
  chance: [LLAMA_3_SOFTMAX, LLAMA_3_AUTOREGRESSIVE],
  generate: [LLAMA_3_AUTOREGRESSIVE, { label: "In Llama 3's code", value: "\"for cur_pos in range(min_prompt_len, total_len):\" runs one round per new word", source: "llama-3-generation" }],
  "write-round": [{ label: "In Llama 3's code", value: "\"logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)\": each round runs only the words not read yet", source: "llama-3-generation" }],
  pick: [
    { label: "Meta's defaults", value: "\"temperature: float = 0.6\" and \"top_p: float = 0.9\"", source: "llama-3-generation" },
    { label: "Temperature 0", value: "takes \"next_token = torch.argmax(logits[:, -1], dim=-1)\": always the top word", source: "llama-3-generation" },
    { label: "Top-p", value: "\"Top-p sampling selects the smallest set of tokens whose cumulative probability mass exceeds the threshold p.\"", source: "llama-3-generation" },
  ],
  append: [{ label: "In Llama 3's code", value: "\"tokens[:, cur_pos] = next_token\": the picked word is written into the next place", source: "llama-3-generation" }],
  "stop-check": [
    { label: "In Llama 3's code", value: "\"torch.isin(next_token, stop_tokens)\" checks each pick against the stop words, 128,001 and 128,009", source: "llama-3-generation" },
    { label: "Base model", value: "the length limit, max_gen_len, \"is needed because pre-trained models usually do not stop completions naturally\"", source: "llama-3-text-example" },
    { label: "Chat model", value: "max_gen_len \"is optional because finetuned models are able to stop generations naturally\"", source: "llama-3-chat-example" },
  ],
  "kv-cache": [
    { label: "In Llama 3's code", value: "\"self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk\" saves the new keys, and \"keys = self.cache_k[:bsz, : start_pos + seqlen]\" reads them all back", source: "llama-3-code" },
    LLAMA_3_GQA,
  ],
};

export const LLAMA_3_WRITING_NOTES: WritingNotes = {
  softmax: { label: "Default temperature", value: "0.6; the scores are divided by it first", source: "llama-3-generation" },
  chance: { label: "Whose scores", value: "only the last word's, logits[:, -1]", source: "llama-3-generation" },
  generate: { label: "Meta's defaults", value: "temperature 0.6 and top-p 0.9", source: "llama-3-generation" },
  "write-round": { label: "Words run each round", value: "only the ones not read yet", source: "llama-3-generation" },
  pick: { label: "Temperature 0", value: "always takes the top word, with no roll", source: "llama-3-generation" },
  append: { label: "Written into", value: "the next place, by tokens[:, cur_pos] = next_token", source: "llama-3-generation" },
};

import type { RealNote } from "@/features/take-apart/interfaces/real-facts.interface";

export const LLAMA_3_TRANSFORMER: RealNote = { label: "Built on", value: "\"we opted for a relatively standard decoder-only transformer architecture in Llama 3\"", source: "llama-3-blog" };

export const LLAMA_3_NO_REASON: RealNote = { label: "Meta's own reason", value: "Not published. The model card and blog do not say why this part is built this way.", source: "llama-3-card" };

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

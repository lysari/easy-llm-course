import type { WritingNotes } from "@/features/take-apart/interfaces/inference-spec.interface";
import type { RealNote, UseReasons } from "@/features/take-apart/interfaces/real-facts.interface";

export const GPT_2_TRANSFORMER: RealNote = { label: "Built on", value: "\"We use a Transformer (Vaswani et al., 2017) based architecture for our LMs\"", source: "gpt-2-paper" };

export const GPT_2_NO_REASON: RealNote = { label: "OpenAI's own reason", value: "Not published. The paper says the model \"largely follows the details of the OpenAI GPT model\" and does not say why each part is there.", source: "gpt-2-paper" };

export const GPT_2_SIZE_HELPS: RealNote = { label: "Why size matters", value: "\"The capacity of the language model is essential to the success of zero-shot task transfer and increasing it improves performance in a log-linear fashion across tasks\"", source: "gpt-2-paper" };

export const GPT_2_RANDOM_START: RealNote = { label: "Starting numbers", value: "every table starts random: \"initializer=tf.random_normal_initializer(stddev=w_init_stdev)\", with w_init_stdev=0.02", source: "gpt-2-code" };

export const GPT_2_MATMUL: RealNote = { label: "In GPT-2's code", value: "every table is used through conv1d: \"tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b\"", source: "gpt-2-code" };

export const GPT_2_BIAS: RealNote = { label: "Its biases", value: "\"b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))\": each bias starts at 0", source: "gpt-2-code" };

export const GPT_2_QKV: RealNote = { label: "In GPT-2's code", value: "\"c = conv1d(x, 'c_attn', n_state*3)\", then \"q, k, v = map(split_heads, tf.split(c, 3, axis=2))\": one table makes all three, split into 12 heads", source: "gpt-2-code" };

export const GPT_2_SCORES: RealNote = { label: "In GPT-2's code", value: "\"w = tf.matmul(q, k, transpose_b=True)\", scaled by \"tf.rsqrt\" of the head size, masked, then \"w = softmax(w)\"", source: "gpt-2-code" };

export const GPT_2_MASK: RealNote = { label: "Earlier words only", value: "the mask keeps \"1's in the lower triangle\", so a word never sees the words after it", source: "gpt-2-code" };

export const GPT_2_MIX: RealNote = { label: "In GPT-2's code", value: "\"a = tf.matmul(w, v)\": the shares times the values", source: "gpt-2-code" };

export const GPT_2_JOIN_HEADS: RealNote = { label: "In GPT-2's code", value: "\"a = merge_heads(a)\", then \"a = conv1d(a, 'c_proj', n_state)\"", source: "gpt-2-code" };

export const GPT_2_HEADS: RealNote = { label: "Heads", value: "\"assert n_state % hparams.n_head == 0\": the 768 numbers are split evenly into 12 heads of 64", source: "gpt-2-code" };

export const GPT_2_NORM_PLACE: RealNote = { label: "Where the norms go", value: "layer normalization \"was moved to the input of each sub-block\" and \"an additional layer normalization was added after the final self-attention block\"", source: "gpt-2-paper" };

export const GPT_2_NORM_CODE: RealNote = { label: "In GPT-2's code", value: "\"Normalize to mean = 0, std = 1, then do a diagonal affine transform.\" The scale g starts at 1 and the shift b at 0", source: "gpt-2-code" };

export const GPT_2_ADD_BACK: RealNote = { label: "Adding back", value: "\"x = x + a\" after attention and \"x = x + m\" after the feed-forward", source: "gpt-2-code" };

export const GPT_2_DEPTH: RealNote = { label: "Depth", value: "\"A modified initialization which accounts for the accumulation on the residual path with model depth is used\"", source: "gpt-2-paper" };

export const GPT_2_MLP: RealNote = { label: "In GPT-2's code", value: "\"h = gelu(conv1d(x, 'c_fc', n_state))\" with n_state = nx*4, then \"h2 = conv1d(h, 'c_proj', nx)\"", source: "gpt-2-code" };

export const GPT_2_GELU: RealNote = { label: "In GPT-2's code", value: "\"return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))\"", source: "gpt-2-code" };

export const GPT_2_WORD_TABLE: RealNote = { label: "In GPT-2's code", value: "\"wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd], …)\" and \"h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))\"", source: "gpt-2-code" };

export const GPT_2_POSITION_TABLE: RealNote = { label: "In GPT-2's code", value: "\"wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd], …)\": one learned row per place", source: "gpt-2-code" };

export const GPT_2_CONTEXT: RealNote = { label: "Places", value: "\"We also increase the context size from 512 to 1024 tokens\"", source: "gpt-2-paper" };

export const GPT_2_OUTPUT: RealNote = { label: "In GPT-2's code", value: "\"logits = tf.matmul(h_flat, wte, transpose_b=True)\": the word table is reused to score every word", source: "gpt-2-code" };

export const GPT_2_FINAL_NORM: RealNote = { label: "In GPT-2's code", value: "\"h = norm(h, 'ln_f')\" runs after the last layer, before the scores", source: "gpt-2-code" };

export const GPT_2_LAYER_LOOP: RealNote = { label: "In GPT-2's code", value: "\"for layer, past in enumerate(pasts):\" runs \"block\" once per layer, and config.json sets n_layer to 12", source: "gpt-2-code" };

export const GPT_2_MERGES: RealNote = { label: "Merge rules", value: "50,000, counted line by line in OpenAI's vocab.bpe file", source: "gpt-2-merges" };

export const GPT_2_BYTES: RealNote = { label: "Why bytes", value: "\"a byte-level version of BPE only requires a base vocabulary of size 256\"", source: "gpt-2-paper" };

export const GPT_2_ANY_STRING: RealNote = { label: "What it gives", value: "\"our approach can assign a probability to any Unicode string\"", source: "gpt-2-paper" };

export const GPT_2_LOOKUP: RealNote = { label: "In GPT-2's code", value: "\"self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')\": each piece is looked up for its ID", source: "gpt-2-encoder" };

export const GPT_2_LIST_SIZE: RealNote = { label: "List size", value: "\"The vocabulary is expanded to 50,257\"", source: "gpt-2-paper" };

export const GPT_2_NEXT_WORD: RealNote = { label: "Next-word guessing", value: "\"it is common to factorize the joint probabilities over symbols as the product of conditional probabilities\"", source: "gpt-2-paper" };

const GPT_2_LAST_WORD: RealNote = { label: "Only the last word", value: "\"logits = next_outputs['logits'][:, -1, :]\": only the last word's scores are used to pick the next word", source: "gpt-2-sample" };

export const GPT_2_USE_REASONS: UseReasons = {
  tokenize: [GPT_2_ANY_STRING, { label: "In GPT-2's code", value: "\"for token in re.findall(self.pat, text):\" cuts your text into words, and each word is then split into pieces", source: "gpt-2-encoder" }],
  split: [{ label: "In GPT-2's code", value: "\"token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))\": each word starts as its bytes", source: "gpt-2-encoder" }, GPT_2_BYTES],
  "apply-merge": [{ label: "In GPT-2's code", value: "\"bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))\": of the pairs side by side, the rule learned first is used first", source: "gpt-2-encoder" }, GPT_2_MERGES],
  lookup: [GPT_2_LOOKUP, GPT_2_LIST_SIZE],
  embedding: [GPT_2_WORD_TABLE, GPT_2_NO_REASON],
  output: [GPT_2_FINAL_NORM, GPT_2_OUTPUT, GPT_2_LAST_WORD],
  softmax: [GPT_2_NEXT_WORD, { label: "In GPT-2's code", value: "the scores go straight into \"tf.multinomial(logits\", TensorFlow's random draw, which turns them into chances itself", source: "gpt-2-sample" }],
  chance: [GPT_2_NEXT_WORD, GPT_2_LAST_WORD],
  generate: [{ label: "OpenAI's examples", value: "\"Top-k random sampling with k = 40 was used for generation.\"", source: "gpt-2-paper" }, { label: "In GPT-2's code", value: "\"tf.while_loop\" runs one round per new word, up to \"maximum_iterations=length - 1\"", source: "gpt-2-sample" }],
  "write-round": [{ label: "In GPT-2's code", value: "\"next_outputs = step(hparams, prev, past=past)\" runs the model once per round, on the newest word, with the saved keys and values", source: "gpt-2-sample" }, GPT_2_NEXT_WORD],
  pick: [
    { label: "In GPT-2's code", value: "\"logits = top_k_logits(logits, k=top_k)\" and \"logits = top_p_logits(logits, p=top_p)\" cut the list, then \"tf.multinomial(logits\" rolls for one word", source: "gpt-2-sample" },
    { label: "OpenAI's advice on top-k", value: "\"0 (default) is a special setting meaning no restrictions. 40 generally is a good value.\"", source: "gpt-2-interactive" },
    { label: "Temperature", value: "\"As the temperature approaches zero, the model will become deterministic and repetitive.\"", source: "gpt-2-interactive" },
  ],
  append: [GPT_2_NEXT_WORD, { label: "Its place", value: "\"positions_for(X, past_length)\": the added word gets the next place number, after the words already read", source: "gpt-2-code" }],
  "stop-check": [
    { label: "In GPT-2's code", value: "the loop's check, \"def cond(*args):\", always does \"return True\"; it stops only after \"maximum_iterations=length - 1\", so the released code never stops at <|endoftext|>", source: "gpt-2-sample" },
    { label: "Length when none is given", value: "\"length = hparams.n_ctx // 2\": 512 words", source: "gpt-2-interactive" },
  ],
  "kv-cache": [
    { label: "In GPT-2's code", value: "\"present = tf.stack([k, v], axis=1)\" saves each layer's keys and values, and \"k = tf.concat([pk, k], axis=-2)\" puts the saved ones first", source: "gpt-2-code" },
    { label: "Its size", value: "\"[batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]\": 12 layers × 2 × 12 heads × 64 numbers per word", source: "gpt-2-code" },
  ],
};

export const GPT_2_WRITING_NOTES: WritingNotes = {
  softmax: { label: "Default temperature", value: "1, so the scores are used as they are", source: "gpt-2-interactive" },
  chance: { label: "Whose scores", value: "only the last word's; the other words' scores are not used", source: "gpt-2-sample" },
  generate: { label: "OpenAI's examples", value: "top-k 40, so each pick comes from the 40 likeliest words", source: "gpt-2-paper" },
  "write-round": { label: "Words run each round", value: "only the newest one, with the saved keys and values", source: "gpt-2-sample" },
  pick: { label: "Default settings", value: "temperature 1, top-k 0 (off) and top-p 1 (off)", source: "gpt-2-interactive" },
  append: { label: "Its place", value: "the next place number, after the words already read", source: "gpt-2-code" },
};

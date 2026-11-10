import type { RealNote } from "@/features/take-apart/interfaces/real-facts.interface";

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

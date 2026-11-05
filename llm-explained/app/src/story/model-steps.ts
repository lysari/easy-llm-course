import type { FlowStep, Status } from "../interfaces/story.interface";
import { MOMENTUM_DECAY, LEARNING_RATE, VARIANCE_DECAY } from "../model/adam";
import { END_TOKEN, TRAINING_LINES, TRAINING_SEQUENCES, VOCABULARY, joinTokens } from "../model/corpus";
import { TARGET_STEPS } from "../model/lab";
import { BLOCK_COUNT, EXPANDED_SIZE, VECTOR_SIZE, countWeights } from "../model/tiny-gpt";
import { chanceBars, count, escapeHtml, lossCurve, lossPlot, percent, shortList, showWord, wordCards } from "./pieces";
import { type Session, TEST_ANSWER, TEST_WORDS } from "./session";

const LAST = TEST_WORDS.length - 1;
const TEST_TEXT = joinTokens(TEST_WORDS);
const GUESSING_LOSS = Math.log(VOCABULARY.length);
const tokenTotal = TRAINING_SEQUENCES.reduce((sum, sequence) => sum + sequence.length, 0);
const questionTotal = TRAINING_SEQUENCES.reduce((sum, sequence) => sum + sequence.length - 1, 0);
const gridShape = `${TEST_WORDS.length} × ${VECTOR_SIZE}`;
const FIRST_LINE = TRAINING_LINES[0].text.split(" ");

const trainStatus = (session: Session): Status => {
  if (session.isRunning("train")) return "running";
  return session.lab.isTrained ? "done" : "needs";
};
const modelState = (session: Session) => (session.lab.isTrained ? "the trained model" : session.lab.step > 0 ? `round ${session.lab.step}` : "the untrained model");

export const dataSteps: FlowStep[] = [
  {
    id: "text",
    stage: "data",
    icon: "text",
    title: "Training text",
    lesson: "Lessons 09, 14",
    goal: "Collect the text the model will learn from.",
    from: () => "Nothing",
    to: () => `${count(tokenTotal)} tokens of text`,
    status: () => "done",
    rowsTitle: `All ${TRAINING_LINES.length} kinds of sentence`,
    rows: () => TRAINING_LINES.map((line) => ({ text: joinTokens(line.text.split(" ")), value: `${line.times}×` })),
    note: () => "A hungry dog gets a bone 3 times and a treat 1 time. Watch for that later.",
    why: "A model only knows what is in its text. GPT-1 read about 800 million words. Ours reads a few sentences, so it can only memorise them.",
  },
  {
    id: "tokenizer",
    stage: "data",
    icon: "hash",
    title: "Tokenizer",
    lesson: "Lessons 09, 18",
    goal: "Turn text into numbers, and back again.",
    from: () => "Text",
    to: () => `${VOCABULARY.length} token IDs`,
    status: () => "done",
    visual: () => wordCards(VOCABULARY, { numbers: VOCABULARY.map((_, i) => i) }),
    rowsTitle: "Three ways to cut text",
    rows: () => [
      { text: "Characters (Lesson 09)", value: "small dictionary, long text" },
      { text: "Whole words (this model)", value: `${VOCABULARY.length} tokens, unknown words fail` },
      { text: "BPE word pieces (Lesson 18)", value: "GPT-2: 50,257 tokens, any text" },
      { text: `Encode "my dog"`, value: `[${VOCABULARY.indexOf("my")}, ${VOCABULARY.indexOf("dog")}]` },
      { text: `Decode [${VOCABULARY.indexOf("my")}, ${VOCABULARY.indexOf("dog")}]`, value: `"my dog"` },
    ],
    note: () => `"stop" (ID ${VOCABULARY.indexOf(END_TOKEN)}) is a real token. It is how a model learns to finish.`,
    code: "encode(text: string): number[] {\n  return text.split(/\\s+/).map(w => this.wordToId.get(w));\n}\ndecode(ids: number[]): string {\n  return ids.map(i => this.idToWord.get(i)).join(' ');\n}",
    why: "A token number has no meaning. It is only an address, like a locker number.",
  },
  {
    id: "pairs",
    stage: "data",
    icon: "question",
    title: "Training pairs",
    lesson: "Lessons 09, 11",
    goal: "Make questions: given these tokens, what comes next?",
    from: () => "Token IDs",
    to: () => `${count(questionTotal)} questions`,
    status: () => "done",
    rowsTitle: "From the first sentence",
    rows: () => FIRST_LINE.slice(0, -1).map((_, i) => ({ text: joinTokens(FIRST_LINE.slice(0, i + 1)), value: `→ ${FIRST_LINE[i + 1]}` })),
    code: "const inputs  = tokens.slice(0, -1);\nconst targets = tokens.slice(1);",
    why: "Nobody writes the answers. The next word in the text is always the answer. This is the only task a base model trains on.",
  },
];

export const forwardSteps: FlowStep[] = [
  {
    id: "embedding",
    stage: "forward",
    icon: "grid",
    title: "Token embedding",
    lesson: "Lesson 10",
    goal: "Each token ID opens a row of numbers.",
    from: () => `${TEST_WORDS.length} token IDs`,
    to: () => `${gridShape} numbers`,
    status: () => "done",
    rowsTitle: "Real lookups",
    rows: (session) => {
      const { tokenVectors } = session.trace();
      return TEST_WORDS.slice(0, 4).map((word, i) => ({ text: `"${word}" → row ${session.testIds[i]}`, value: shortList(tokenVectors[i], 3) }));
    },
    note: (session) => {
      const before = session.similarity("dog", "cat", session.lab.untrained);
      const now = session.similarity("dog", "cat", session.lab.live);
      return `Similarity of "dog" and "cat": ${before.toFixed(2)} at the start, ${now.toFixed(2)} now. With huge text, training pulls words used in similar ways closer. Our tiny text barely shows this.`;
    },
    formula: `table shape: ${VOCABULARY.length} × ${VECTOR_SIZE} (words × numbers)\ncosine(a, b) = a · b / (|a| × |b|)`,
    code: "function lookup(tokenId: number, table: number[][]): number[] {\n  return table[tokenId];\n}",
    why: "It is a lookup, not a calculation. The numbers start random, and training fills them with meaning.",
  },
  {
    id: "position",
    stage: "forward",
    icon: "order",
    title: "Position",
    lesson: "Lessons 14, 25",
    goal: "Add word order to the numbers.",
    from: () => `${gridShape} numbers`,
    to: () => `${gridShape} numbers, with order`,
    status: () => "done",
    rowsTitle: `"${TEST_WORDS[1]}" at place 2`,
    rows: (session) => {
      const { tokenVectors, positionVectors, combined } = session.trace();
      return [
        { text: "Token list", value: shortList(tokenVectors[1], 4) },
        { text: "+ place 2 list", value: shortList(positionVectors[1], 4) },
        { text: "= new list", value: shortList(combined[1], 4) },
      ];
    },
    formula: "final[position] = token_embedding + position_embedding",
    code: "const X = tokens.map((t, i) =>\n  add(tokenTable[t], positionTable[i]));",
    why: "Attention treats tokens as a set, so without this, \"dog bites man\" and \"man bites dog\" look the same. Newer models use RoPE (Lesson 25).",
  },
  {
    id: "layernorm",
    stage: "forward",
    icon: "gauge",
    title: "LayerNorm",
    lesson: "Lessons 13, 20",
    goal: "Keep each list's numbers in a steady range.",
    from: () => `${gridShape} numbers`,
    to: () => `${gridShape}, mean 0, spread 1`,
    status: () => "done",
    rowsTitle: "The lesson's example: [2, 4, 6, 8]",
    rows: (session) => {
      const row = session.trace().blocks[0].normalized[LAST];
      const mean = row.reduce((sum, value) => sum + value, 0) / row.length;
      return [
        { text: "mean", value: "(2 + 4 + 6 + 8) / 4 = 5" },
        { text: "variance", value: "(9 + 1 + 1 + 9) / 4 = 5" },
        { text: "(x − mean) / √5", value: "−1.34, −0.45, 0.45, 1.34" },
        { text: `Real result for "${TEST_WORDS[LAST]}"`, value: shortList(row, 3) },
        { text: "Its mean now", value: Math.abs(mean) < 0.005 ? "0" : mean.toFixed(2) },
      ];
    },
    formula: "LayerNorm(x) = (x − mean(x)) / √(variance(x) + ε)",
    code: "const mean = x.reduce((s, v) => s + v, 0) / x.length;\nconst variance = x.reduce((s, v) => s + (v - mean) ** 2, 0) / x.length;\nreturn x.map(v => (v - mean) / Math.sqrt(variance + eps));",
    why: "Numbers can grow too large or too small as they pass through layers, which makes training unstable. This model normalizes before each job (Pre-LN, from GPT-2).",
  },
  {
    id: "attention",
    stage: "forward",
    icon: "eye",
    title: "Attention",
    lesson: "Lessons 12, 19",
    goal: "Each word looks back and collects from earlier words.",
    from: () => `${gridShape} numbers`,
    to: () => `${gridShape} numbers, with context`,
    status: () => "done",
    visual: (session) => {
      const shares = session.trace().blocks[0].attention[LAST];
      return `${wordCards(TEST_WORDS, { numbers: shares.map(percent) })}`;
    },
    rowsTitle: `Five steps for "${TEST_WORDS[LAST]}", real numbers`,
    rows: (session) => {
      const block = session.trace().blocks[0];
      const shares = block.attention[LAST];
      const top = shares.indexOf(Math.max(...shares));
      return [
        { text: "1. Query Q = X @ Wq", value: shortList(block.queries[LAST], 3) },
        { text: `2. Key K of "${TEST_WORDS[top]}" = X @ Wk`, value: shortList(block.keys[top], 3) },
        { text: "3. Score Q · K / √24", value: block.scores[LAST][top].toFixed(2) },
        { text: "4. Softmax over this row", value: percent(shares[top]) },
        { text: "5. Output = Σ share × V", value: shortList(block.collected[LAST], 3) },
      ];
    },
    note: (session) => `Showing ${modelState(session)}, layer 1. Later words are hidden (causal mask). This model has 1 head. GPT-2 small runs 12 in parallel, each of 64 numbers.`,
    formula: "attention = softmax(Q · Kᵀ / √d) · V\nQ: what am I looking for?\nK: what do I offer?\nV: what do I contain?",
    code: "const scores = matmul(Q, transpose(K))\n  .map(row => row.map(s => s / Math.sqrt(dk)));\nif (causal)\n  for (let i = 0; i < T; i++)\n    for (let j = i + 1; j < T; j++)\n      scores[i][j] = -Infinity;\nconst weights = scores.map(row => softmax(row));",
    why: "This lets a word use clues from far back in the sentence. It works like a search: match a query against keys, then take the values.",
  },
  {
    id: "ffn",
    stage: "forward",
    icon: "merge",
    title: "Feed-forward network",
    lesson: "Lessons 13, 15",
    goal: "Each word thinks alone about what it collected.",
    from: () => `${gridShape} numbers`,
    to: () => `${gridShape} numbers`,
    status: () => "done",
    rowsTitle: `For "${TEST_WORDS[LAST]}", real numbers`,
    rows: (session) => {
      const block = session.trace().blocks[0];
      const expanded = block.expanded[LAST];
      const active = expanded.filter((value) => value > 0).length;
      return [
        { text: "Attention + old list (residual)", value: shortList(block.afterAttention[LAST], 3) },
        { text: `1. Expand: ${VECTOR_SIZE} → ${EXPANDED_SIZE} numbers`, value: "× W1" },
        { text: "2. ReLU: negatives become 0", value: `${active} of ${EXPANDED_SIZE} stay on` },
        { text: `3. Contract: ${EXPANDED_SIZE} → ${VECTOR_SIZE} numbers`, value: "× W2" },
        { text: "4. Add old list back (residual)", value: shortList(block.afterFeedForward[LAST], 3) },
      ];
    },
    formula: "FFN(x) = W2 · ReLU(W1 · x)\noutput = x + FFN(LayerNorm(x))",
    code: "// Lesson 15 uses GELU, a smoother curve\nconst pre = matmul(x, W1);\nconst h = pre.map(relu);\nconst out = matmul(h, W2);\nreturn add(x, out);",
    why: "Attention is communication between words. This is computation inside one word. The 4× width gives it working space.",
  },
  {
    id: "repeat-layer",
    stage: "forward",
    icon: "repeat",
    title: "Stack the hidden layers",
    lesson: "Lessons 13, 20",
    goal: "Repeat attention and feed-forward, layer after layer.",
    from: () => `${gridShape} numbers`,
    to: () => `${gridShape} numbers (same size)`,
    status: () => "done",
    rowsTitle: `The list for "${TEST_WORDS[LAST]}" at each point`,
    rows: (session) => {
      const { combined, blocks } = session.trace();
      return [
        { text: "Going in", value: shortList(combined[LAST], 3) },
        ...blocks.flatMap((block, i) => [
          { text: `Layer ${i + 1} attention`, value: shortList(block.afterAttention[LAST], 3) },
          { text: `Layer ${i + 1} feed-forward`, value: shortList(block.afterFeedForward[LAST], 3) },
        ]),
      ];
    },
    note: () => `This model has ${BLOCK_COUNT} layers. GPT-1 and GPT-2 small have 12. The largest GPT-2 has 48.`,
    formula: "x = x + attention(LayerNorm(x))\nx = x + FFN(LayerNorm(x))",
    why: "These are the hidden layers. The size going in equals the size coming out, so layers can stack. Residual additions keep gradients flowing through deep stacks.",
  },
  {
    id: "logits",
    stage: "forward",
    icon: "percent",
    title: "Output scores (logits)",
    lesson: "Lessons 11, 20",
    goal: "Turn the last word's list into one score per dictionary word.",
    from: () => `1 × ${VECTOR_SIZE} numbers`,
    to: () => `${VOCABULARY.length} scores`,
    status: () => "done",
    rowsTitle: `Top scores after "${TEST_TEXT}"`,
    rows: (session) => session.guess().slice(0, 6).map((option) => ({ text: showWord(option.token), value: option.score.toFixed(2) })),
    note: () => "Only the last token's list predicts the next word. In training, every position predicts its own next word.",
    formula: `logits = LayerNorm(x) @ Woutput\n(1 × ${VECTOR_SIZE}) @ (${VECTOR_SIZE} × ${VOCABULARY.length}) = (1 × ${VOCABULARY.length})`,
    code: "const logits = matmul(layerNorm(x), outputTable);",
    why: "Scores can be any size, and they do not add up to anything yet. GPT-2 reuses its token table here, which is called weight tying.",
  },
  {
    id: "softmax",
    stage: "forward",
    icon: "percent",
    title: "Softmax",
    lesson: "Lesson 11",
    goal: "Turn the scores into percents that add up to 100.",
    from: () => `${VOCABULARY.length} scores`,
    to: () => `${VOCABULARY.length} percents`,
    status: () => "done",
    visual: (session) => chanceBars(session.guess(), TEST_ANSWER),
    rowsTitle: "Worked out for the top word",
    rows: (session) => {
      const guess = session.guess();
      const total = guess.reduce((sum, option) => sum + Math.exp(option.score), 0);
      const [top] = guess;
      return [
        { text: `e^score for "${showWord(top.token)}"`, value: Math.exp(top.score).toFixed(2) },
        { text: "Sum of e^score for every word", value: total.toFixed(2) },
        { text: "Divide", value: `${(top.chance * 100).toFixed(1)}%` },
      ];
    },
    note: (session) => `Showing ${modelState(session)}. The bar in a different blue is the right answer, "${TEST_ANSWER}".`,
    formula: "p(i) = e^score(i) / Σ e^score(j)",
    code: "const max = Math.max(...logits);\nconst exps = logits.map(l => Math.exp(l - max));\nconst sum = exps.reduce((a, b) => a + b, 0);\nreturn exps.map(e => e / sum);",
    why: "Exponents make gaps bigger and keep every value positive. Subtracting the largest score avoids overflow.",
  },
];

export const learnSteps: FlowStep[] = [
  {
    id: "loss",
    stage: "learn",
    icon: "gauge",
    title: "Cross-entropy loss",
    lesson: "Lesson 11",
    goal: "Score the guess by the chance given to the right word.",
    from: () => `${VOCABULARY.length} percents + the answer`,
    to: () => "1 number: the loss",
    status: trainStatus,
    rowsTitle: "Real numbers",
    rows: (session) => {
      const chance = session.answerChanceNow();
      return [
        { text: `Chance given to "${TEST_ANSWER}"`, value: `${(chance * 100).toFixed(1)}%` },
        { text: "Loss = −ln(chance)", value: (-Math.log(chance)).toFixed(2) },
        { text: "Average over all the training text", value: (session.lab.latestLoss ?? session.startLoss).toFixed(2) },
        { text: `Random guessing = ln(${VOCABULARY.length})`, value: GUESSING_LOSS.toFixed(2) },
      ];
    },
    visual: (session) => lossCurve(session.answerChanceNow()),
    formula: "loss = −ln(p(right word))\np = 100% → loss 0",
    code: "function crossEntropyLoss(probs: number[], targetIndex: number) {\n  return -Math.log(probs[targetIndex] + 1e-9);\n}",
    why: "Being confidently wrong is punished hard. This is the number the whole training loop tries to lower. An untrained model starts near random guessing.",
  },
  {
    id: "backprop",
    stage: "learn",
    icon: "back",
    title: "Backprop through attention",
    lesson: "Lesson 16",
    goal: "Go right to left, and find the gradient of every number.",
    from: () => "1 loss",
    to: (session) => `${count(countWeights(session.lab.live))} gradients`,
    status: trainStatus,
    rowsTitle: "Path of the gradient, right to left",
    rows: () => [
      { text: "Loss → logits", value: "p − target" },
      { text: "logits → output table", value: "dC · Bᵀ" },
      { text: "FFN → W2, ReLU, W1", value: "chain rule" },
      { text: "attnOut → V and A", value: "dV = Aᵀ · dOut" },
      { text: "A → scores (softmax)", value: "A(i,j) (δ − A(i,l))" },
      { text: "scores → Q and K", value: "÷ √d" },
      { text: "Q, K, V → Wq, Wk, Wv", value: "chain rule" },
      { text: "→ position and token tables", value: "row by row" },
    ],
    visual: (session) => {
      const tables = session.gradientTables();
      const largest = Math.max(...tables.map((table) => table.size));
      return `<p class="section-title">Average gradient per table, first training step</p>${chanceBarsFromSizes(tables, largest)}`;
    },
    note: (session) => {
      const test = session.nudgeTest();
      return `Checked for real: one number has gradient ${test.gradient.toFixed(3)}. That predicts raising it by 0.01 changes the loss by ${test.predicted.toFixed(4)}. Measured: ${test.measured.toFixed(4)}.`;
    },
    formula: "C = A·B → dA = dC·Bᵀ, dB = Aᵀ·dC\n∂A[i][j]/∂s[i][l] = A[i][j] (δjl − A[i][l])",
    why: "One backward pass gives the answer for every number at once. Trying each number separately would be far too slow.",
  },
  {
    id: "adam",
    stage: "learn",
    icon: "sliders",
    title: "Adam optimizer",
    lesson: "Lesson 17",
    goal: "Give every number its own step size.",
    from: () => "Gradients",
    to: () => "Slightly better numbers",
    status: trainStatus,
    rowsTitle: "First update of one real number",
    rows: (session) => {
      const step = session.adamFirstStep();
      return [
        { text: "Number now", value: step.value.toFixed(4) },
        { text: "Gradient g", value: step.gradient.toFixed(4) },
        { text: `1. m = ${MOMENTUM_DECAY} × 0 + 0.1 × g`, value: step.m.toFixed(5) },
        { text: `2. v = ${VARIANCE_DECAY} × 0 + 0.001 × g²`, value: step.v.toExponential(2) },
        { text: "3. m̂ = m / (1 − 0.9¹)", value: step.mHat.toFixed(4) },
        { text: "4. v̂ = v / (1 − 0.999¹)", value: step.vHat.toExponential(2) },
        { text: `5. step = ${LEARNING_RATE} × m̂ / √v̂`, value: step.change.toFixed(4) },
        { text: "New number", value: step.after.toFixed(4) },
      ];
    },
    note: (session) => `The first step is always exactly the learning rate, ${LEARNING_RATE}. Later steps vary. So far, numbers have moved ${session.averageMove().toFixed(3)} on average.`,
    formula: "m = β1·m + (1−β1)·g\nv = β2·v + (1−β2)·g²\nθ = θ − lr · m̂ / (√v̂ + ε)",
    code: "m = 0.9 * m + 0.1 * g;\nv = 0.999 * v + 0.001 * g * g;\nconst mHat = m / (1 - 0.9 ** t);\nconst vHat = v / (1 - 0.999 ** t);\ntheta -= lr * mHat / (Math.sqrt(vHat) + 1e-8);",
    why: "m remembers direction, like momentum. v tracks how much a number bounces. Steady numbers get bigger steps, and bouncing numbers get smaller ones.",
  },
  {
    id: "train-loop",
    stage: "learn",
    icon: "repeat",
    title: "The training loop",
    lesson: "Lessons 14, 22",
    goal: `Repeat the whole path ${TARGET_STEPS} times.`,
    from: () => "Random numbers",
    to: (session) => (session.lab.isTrained ? "A trained model" : `Round ${session.lab.step} of ${TARGET_STEPS}`),
    status: trainStatus,
    actions: (session) => {
      if (session.lab.isTrained) return [{ label: "Forget everything", command: "forget" }];
      if (session.isRunning("train")) return [{ label: "Finish quickly", command: "finish", isStrong: true }];
      return [{ label: session.lab.step > 0 ? "Continue" : "Start training", command: "start:train", isStrong: true }];
    },
    visual: (session) => lossPlot(session.lab.history, TARGET_STEPS, GUESSING_LOSS),
    rowsTitle: "Right now",
    rows: (session) => [
      { text: "Round", value: `${session.lab.step} of ${TARGET_STEPS}` },
      { text: "Loss on all the text", value: (session.lab.latestLoss ?? session.startLoss).toFixed(2) },
      { text: `Chance of "${TEST_ANSWER}"`, value: `${(session.answerChanceNow() * 100).toFixed(1)}%` },
      { text: "Time", value: `${Math.round(session.trainSeconds)} s` },
    ],
    note: (session) => `It writes: "${escapeHtml(session.sampleWriting())}". The loss stops near 0.2 because the text itself is unsure: bone or treat.`,
    code: "for (let round = 0; round < 200; round++) {\n  const loss = forward(text);   // guess and score\n  backward(loss);               // find gradients\n  adam.update();                // fix the numbers\n}",
    why: "One step does almost nothing. Thousands of small steps add up to learning. Real models read so much they must learn general patterns, but ours just memorises.",
  },
];

function chanceBarsFromSizes(tables: { name: string; size: number }[], largest: number) {
  return `<div class="chances">${tables
    .map(({ name, size }) => `<div class="chance" title="${escapeHtml(name)}: ${size.toFixed(4)}"><span>${escapeHtml(name.replace("feed-forward", "FFN").replace("attention", "attn").replace("Layer ", "L"))}</span><span class="chance-track"><i class="chance-fill" style="width:${Math.round((size / largest) * 100)}%"></i></span><span>${size.toFixed(3)}</span></div>`)
    .join("")}</div>`;
}

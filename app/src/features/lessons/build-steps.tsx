import { TEST_ANSWER, TEST_WORDS, VOCABULARY, joinTokens } from "@/core/model/corpus";
import { LAYER_COUNT, EXPANDED_SIZE, VECTOR_SIZE } from "@/core/model/tiny-gpt";
import type { FlowStep } from "@/features/lessons/interfaces/story.interface";
import { ChanceBars } from "@/shared/components/chance-bars";
import { percent, shortList, showWord } from "@/shared/utils/format";
import { WordCards } from "./lesson-plots";
import type { Session } from "./session";

const LAST = TEST_WORDS.length - 1;
const TEST_TEXT = joinTokens(TEST_WORDS);
const gridShape = `${TEST_WORDS.length} × ${VECTOR_SIZE}`;

const modelState = (session: Session) => (session.lab.isTrained ? "the trained model" : session.lab.round > 0 ? `round ${session.lab.round}` : "the untrained model");

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
    formula: { lines: [`table shape: ${VOCABULARY.length} × ${VECTOR_SIZE} (words × numbers)`, "cosine(a, b) = a · b / (|a| × |b|)"] },
    code: "function lookup(\n  tokenId: number,\n  table: number[][],\n): number[] {\n  return table[tokenId];\n}",
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
    formula: { lines: ["final[position] = token_embedding + position_embedding"] },
    code: "const X = tokens.map((t, i) =>\n  add(tokenTable[t], positionTable[i]),\n);",
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
      const row = session.trace().layers[0].normalized[LAST];
      const mean = row.reduce((sum, value) => sum + value, 0) / row.length;
      return [
        { text: "mean", value: "(2 + 4 + 6 + 8) / 4 = 5" },
        { text: "variance", value: "(9 + 1 + 1 + 9) / 4 = 5" },
        { text: "(x − mean) / √5", value: "−1.34, −0.45, 0.45, 1.34" },
        { text: `Real result for "${TEST_WORDS[LAST]}"`, value: shortList(row, 3) },
        { text: "Its mean now", value: Math.abs(mean) < 0.005 ? "0" : mean.toFixed(2) },
      ];
    },
    formula: { lines: ["LayerNorm(x) = (x − mean(x)) / √(variance(x) + ε)"] },
    code: "const mean = x.reduce((s, v) => s + v, 0) / x.length;\nconst variance =\n  x.reduce((s, v) => s + (v - mean) ** 2, 0) / x.length;\nreturn x.map(\n  (v) => (v - mean) / Math.sqrt(variance + eps),\n);",
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
    visual: (session) => <WordCards words={TEST_WORDS} numbers={session.trace().layers[0].attention[LAST].map(percent)} />,
    rowsTitle: `Five steps for "${TEST_WORDS[LAST]}", real numbers`,
    rows: (session) => {
      const layer = session.trace().layers[0];
      const shares = layer.attention[LAST];
      const top = shares.indexOf(Math.max(...shares));
      return [
        { text: "1. Query Q = X @ Wq", value: shortList(layer.queries[LAST], 3) },
        { text: `2. Key K of "${TEST_WORDS[top]}" = X @ Wk`, value: shortList(layer.keys[top], 3) },
        { text: "3. Score Q · K / √24", value: layer.scores[LAST][top].toFixed(2) },
        { text: "4. Softmax over this row", value: percent(shares[top]) },
        { text: "5. Output = Σ share × V", value: shortList(layer.collected[LAST], 3) },
      ];
    },
    note: (session) => `Showing ${modelState(session)}, layer 1. Later words are hidden (causal mask). This model has 1 head. GPT-2 small runs 12 in parallel, each of 64 numbers.`,
    formula: {
      lines: ["attention = softmax(Q · Kᵀ / √d) · V"],
      legend: [
        { term: "Q", meaning: "what am I looking for?" },
        { term: "K", meaning: "what do I offer?" },
        { term: "V", meaning: "what do I contain?" },
      ],
    },
    code: "const scores = matmul(Q, transpose(K)).map((row) =>\n  row.map((s) => s / Math.sqrt(dk)),\n);\nif (causal)\n  for (let i = 0; i < T; i++)\n    for (let j = i + 1; j < T; j++)\n      scores[i][j] = -Infinity;\nconst weights = scores.map((row) => softmax(row));",
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
      const layer = session.trace().layers[0];
      const expanded = layer.expanded[LAST];
      const active = expanded.filter((value) => value > 0).length;
      return [
        { text: "Attention + old list (residual)", value: shortList(layer.afterAttention[LAST], 3) },
        { text: `1. Expand: ${VECTOR_SIZE} → ${EXPANDED_SIZE} numbers`, value: "× W1" },
        { text: "2. ReLU: negatives become 0", value: `${active} of ${EXPANDED_SIZE} stay on` },
        { text: `3. Contract: ${EXPANDED_SIZE} → ${VECTOR_SIZE} numbers`, value: "× W2" },
        { text: "4. Add old list back (residual)", value: shortList(layer.afterFeedForward[LAST], 3) },
      ];
    },
    formula: { lines: ["FFN(x) = W2 · ReLU(W1 · x)", "output = x + FFN(LayerNorm(x))"] },
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
      const { combined, layers } = session.trace();
      return [
        { text: "Going in", value: shortList(combined[LAST], 3) },
        ...layers.flatMap((layer, i) => [
          { text: `Layer ${i + 1} attention`, value: shortList(layer.afterAttention[LAST], 3) },
          { text: `Layer ${i + 1} feed-forward`, value: shortList(layer.afterFeedForward[LAST], 3) },
        ]),
      ];
    },
    note: () => `This model has ${LAYER_COUNT} layers. GPT-1 and GPT-2 small have 12. The largest GPT-2 has 48.`,
    formula: { lines: ["x = x + attention(LayerNorm(x))", "x = x + FFN(LayerNorm(x))"] },
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
    formula: { lines: ["logits = LayerNorm(x) @ Woutput", `(1 × ${VECTOR_SIZE}) @ (${VECTOR_SIZE} × ${VOCABULARY.length}) = (1 × ${VOCABULARY.length})`] },
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
    visual: (session) => <ChanceBars options={session.guess()} answer={TEST_ANSWER} />,
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
    formula: { lines: ["p(i) = e^score(i) / Σ e^score(j)"] },
    code: "const max = Math.max(...logits);\nconst exps = logits.map((l) => Math.exp(l - max));\nconst sum = exps.reduce((a, b) => a + b, 0);\nreturn exps.map((e) => e / sum);",
    why: "Exponents make gaps bigger and keep every value positive. Subtracting the largest score avoids overflow.",
  },
];

import { VOCABULARY } from "@/core/model/corpus";
import type { Tensor } from "@/core/model/tensor";
import { LAYER_COUNT, VECTOR_SIZE } from "@/core/model/tiny-gpt";
import type { ModelId, ModelPart, TinyView } from "@/features/take-apart/interfaces/model-part.interface";
import { showWord } from "@/shared/utils/format";
import { dot, gelu, layerNormParts, rmsNormParts, rope, ropeAngle, sigmoid, silu } from "./exact-formulas";

export interface NumberList {
  label: string;
  values: number[];
}

export interface Terms {
  leftName: string;
  rightName: string;
  pairs: [number, number][];
  scale: number;
  add: number;
  total: number;
}

export interface TinyStep {
  before: NumberList[];
  after: NumberList[];
  terms: Terms | null;
  pick: { label: string; count: number } | null;
  formula: string;
  note: string;
}

export interface StepChoice {
  word: number;
  out: number;
  extra: number;
}

const fixed = (value: number) => value.toFixed(3);

const columnOf = (table: Tensor, j: number) => Array.from({ length: table.rows }, (_, i) => table.data[i * table.cols + j]);

const termsOf = (leftName: string, rightName: string, pairs: [number, number][], scale = 1, add = 0): Terms => ({
  leftName,
  rightName,
  pairs,
  scale,
  add,
  total: pairs.reduce((sum, [a, b]) => sum + a * b, 0) * scale + add,
});

const step = (before: NumberList[], after: NumberList[], formula: string, note: string, extra: Partial<TinyStep> = {}): TinyStep => ({ before, after, formula, note, terms: null, pick: null, ...extra });

function readLayer(view: TinyView, word: number) {
  const first = view.trace.layers[0];
  const ffIn = layerNormParts(first.afterAttention[word]).out;
  const pre = columnsTimes(ffIn, view.weights.layers[0].expand);
  const finalIn = view.trace.layers[LAYER_COUNT - 1].afterFeedForward[word];
  return { first, ffIn, pre, finalIn, finalOut: layerNormParts(finalIn).out };
}

function columnsTimes(values: number[], table: Tensor) {
  return Array.from({ length: table.cols }, (_, j) => dot(values, columnOf(table, j)));
}

function projection(name: string, input: number[], table: Tensor, level: string, out: number, note: string): TinyStep {
  const output = columnsTimes(input, table);
  if (level === "whole") return step([{ label: `${input.length} numbers in`, values: input }], [{ label: `${output.length} numbers out`, values: output }], `out = in × ${name} (${table.rows} × ${table.cols} stored numbers)`, note);
  const j = out % table.cols;
  const terms = termsOf("in", "stored", input.map((x, i) => [x, table.data[i * table.cols + j]]));
  return step([{ label: `${input.length} numbers in`, values: input }], [{ label: `output number ${j + 1}`, values: [terms.total] }], `out ${j + 1} = in1 × w1 + in2 × w2 + … + in${input.length} × w${input.length}`, note, { terms, pick: { label: "Output number", count: table.cols } });
}

const TINY = "Live in our tiny model, on your sentence.";

function normStep(isRms: boolean, input: number[], level: string, out: number, inName: string): TinyStep {
  const j = out % input.length;
  if (isRms) {
    const { rms, out: scaled } = rmsNormParts(input);
    const note = "Llama 3's RMSNorm formula from its code, run on our tiny model's real numbers. Each scale number is 1, the value the code starts it at; Llama 3's trained scale numbers are in its weights, which this app does not ship.";
    if (level === "whole") return step([{ label: inName, values: input }], [{ label: "divided by their root mean square", values: scaled }], `out = x / ${fixed(rms)}, then × each scale number (1)`, note);
    return step([{ label: `number ${j + 1}`, values: [input[j]] }], [{ label: "rescaled, times its scale", values: [scaled[j]] }], `${fixed(input[j])} / ${fixed(rms)} × 1 = ${fixed(scaled[j])}`, note, { terms: termsOf("rescaled", "scale", [[scaled[j], 1]]), pick: { label: "Number", count: input.length } });
  }
  const { mean, spread, out: normed } = layerNormParts(input);
  const note = "Our tiny model's own LayerNorm, live. It has no stored scale or shift; GPT-2's code starts its scale g at 1 and its shift b at 0, and its trained values are in its weights, which this app does not ship.";
  if (level === "whole") return step([{ label: inName, values: input }], [{ label: "average 0, spread 1", values: normed }], `out = (x − ${fixed(mean)}) / ${fixed(spread)}`, note);
  return step([{ label: `number ${j + 1}`, values: [input[j]] }], [{ label: "× g + b, with g = 1, b = 0", values: [normed[j]] }], `(${fixed(input[j])} − ${fixed(mean)}) / ${fixed(spread)} × 1 + 0 = ${fixed(normed[j])}`, note, { terms: termsOf("rescaled", "g", [[normed[j], 1]]), pick: { label: "Number", count: input.length } });
}

function ropeStep(query: number[], place: number, level: string, out: number): TinyStep {
  const turned = rope(query, place);
  const note = `Llama 3's RoPE formula from its code (θ = 500,000), run on our tiny model's real query numbers for word ${place + 1}. Our tiny query has ${VECTOR_SIZE} numbers, Llama 3's has 128, so the angles differ from Llama's own.`;
  if (level === "whole") return step([{ label: "query numbers", values: query }], [{ label: `turned for place ${place + 1}`, values: turned }], `${VECTOR_SIZE / 2} pairs, each turned by place × 500000^(−2i / ${VECTOR_SIZE})`, note);
  const pair = Math.floor((out % VECTOR_SIZE) / 2);
  const angle = ropeAngle(pair, VECTOR_SIZE, place);
  const terms = termsOf("number", "cos or −sin", [
    [query[pair * 2], Math.cos(angle)],
    [query[pair * 2 + 1], -Math.sin(angle)],
  ]);
  return step([{ label: `pair ${pair + 1}`, values: [query[pair * 2], query[pair * 2 + 1]] }], [{ label: "first number, turned", values: [terms.total] }], `x′ = x × cos(${fixed(angle)}) + y × (−sin(${fixed(angle)}))`, note, { terms, pick: { label: "Pair", count: VECTOR_SIZE / 2 } });
}

export function tinyStep(part: ModelPart, modelId: ModelId, view: TinyView, { word, out, extra }: StepChoice): TinyStep {
  const { trace, weights, lab } = view;
  const w = Math.min(word, view.ids.length - 1);
  const layer = weights.layers[0];
  const { first, ffIn, pre, finalIn, finalOut } = readLayer(view, w);
  const level = part.tiny.level;
  const isRms = modelId === "llama-3-8b";
  const base = part.id.replace(/-(number|multiply-add|bias)$/, "");
  const nearest = part.tiny.missing ?? TINY;
  const tableOf: Partial<Record<string, [string, number[], Tensor]>> = {
    query: ["Wq", first.normalized[w], layer.query],
    key: ["Wk", first.normalized[w], layer.key],
    value: ["Wv", first.normalized[w], layer.value],
    expand: ["W1", ffIn, layer.expand],
    gate: ["W1", ffIn, layer.expand],
    up: ["W1", ffIn, layer.expand],
    contract: ["W2", first.expanded[w], layer.contract],
    down: ["W2", first.expanded[w], layer.contract],
    "output-head": ["the output table", finalOut, weights.outputTable],
    "output-score": ["the output table", finalOut, weights.outputTable],
  };
  if (part.id.endsWith("-bias")) {
    const found = tableOf[base];
    const sum = found ? (projection(found[0], found[1], found[2], "number", out, "").terms?.total ?? 0) : first.collected[w][out % VECTOR_SIZE];
    return step([{ label: "the finished sum", values: [sum] }], [{ label: "plus the bias b", values: [sum + extra] }], `out = ${fixed(sum)} + ${fixed(extra)}`, `The sum is live from our tiny model${found ? "" : " (its head's mix, since it has no output projection)"}. It has no bias, so you set b with the slider. GPT-2's code starts every bias at 0; its trained biases are in its weights, which this app does not ship.`, { pick: { label: "Output number", count: found ? found[2].cols : VECTOR_SIZE } });
  }
  const projected = tableOf[base];
  if (projected) return projection(projected[0], projected[1], projected[2], level, out, nearest);
  if (base === "norm-1") return normStep(isRms, trace.combined[w], level, out, "the word's numbers, entering layer 1");
  if (base === "norm-2") return normStep(isRms, first.afterAttention[w], level, out, "the word's numbers after attention");
  if (base === "final-norm") return normStep(isRms, finalIn, level, out, "the last layer's numbers");
  if (base.startsWith("rope")) return ropeStep(first.queries[w], w, part.id === "rope" ? "whole" : "number", out);
  if (base === "gelu") {
    const j = out % pre.length;
    const note = "GPT-2's GELU formula from its code, run on our tiny model's real expanded numbers. Our tiny model itself uses ReLU, shown beside it.";
    if (part.id === "gelu") return step([{ label: "expanded numbers", values: pre }], [{ label: "after GELU", values: pre.map(gelu) }, { label: "after our tiny model's ReLU", values: first.expanded[w] }], "gelu(x) = 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))", note);
    return step([{ label: `number ${j + 1}`, values: [pre[j]] }], [{ label: "after GELU", values: [gelu(pre[j])] }], `0.5 × ${fixed(pre[j])} × (1 + tanh(…)) = ${fixed(gelu(pre[j]))}`, note, { pick: { label: "Number", count: pre.length } });
  }
  if (["swiglu", "swiglu-number", "silu", "gate-times-up"].includes(part.id)) {
    const j = out % pre.length;
    const note = "Llama 3's SiLU formula from its code, run on our tiny model's real expanded numbers as the gate. Our tiny model has no up table, so you set the up number with the slider.";
    if (part.id === "swiglu") return step([{ label: "gate numbers (our expand table)", values: pre }], [{ label: "after SiLU", values: pre.map(silu) }], "silu(g) = g × sigmoid(g), then × the matching up number", note);
    if (part.id === "silu") return step([{ label: `gate number ${j + 1}`, values: [pre[j]] }], [{ label: "after SiLU", values: [silu(pre[j])] }], `${fixed(pre[j])} × sigmoid(${fixed(pre[j])}) = ${fixed(silu(pre[j]))}`, note, { terms: termsOf("gate", "sigmoid", [[pre[j], sigmoid(pre[j])]]), pick: { label: "Number", count: pre.length } });
    return step([{ label: `SiLU of gate ${j + 1}`, values: [silu(pre[j])] }, { label: "up number (you set it)", values: [extra] }], [{ label: "gate × up", values: [silu(pre[j]) * extra] }], `${fixed(silu(pre[j]))} × ${fixed(extra)} = ${fixed(silu(pre[j]) * extra)}`, note, { terms: termsOf("SiLU", "up", [[silu(pre[j]), extra]]), pick: { label: "Number", count: pre.length } });
  }
  if (base === "attention-output") {
    const j = out % VECTOR_SIZE;
    const values = level === "whole" ? [trace.combined[w], first.collected[w]] : [[trace.combined[w][j]], [first.collected[w][j]]];
    const after = level === "whole" ? first.afterAttention[w] : [first.afterAttention[w][j]];
    return step([{ label: "the word's numbers", values: values[0] }, { label: "what the head mixed in", values: values[1] }], [{ label: "added back", values: after }], "out = x + mix", nearest, { pick: level === "whole" ? null : { label: "Number", count: VECTOR_SIZE } });
  }
  switch (part.tiny.part) {
    case "word-table":
    case "position-table": {
      const table = part.tiny.part === "word-table" ? weights.tokenTable : weights.positionTable;
      const start = part.tiny.part === "word-table" ? lab.untrained.tokenTable : lab.untrained.positionTable;
      const row = part.tiny.part === "word-table" ? view.ids[w] : w;
      const name = part.tiny.part === "word-table" ? `ID ${row}, "${showWord(VOCABULARY[row])}"` : `place ${w + 1}`;
      const j = out % table.cols;
      if (level === "number") return step([{ label: `number ${j + 1} of ${name}, before training`, values: [start.data[row * table.cols + j]] }], [{ label: `now, after ${lab.round} rounds`, values: [table.data[row * table.cols + j]] }], "w = w − step size × slope, every round", TINY, { pick: { label: "Number", count: table.cols } });
      return step([{ label: part.tiny.part === "word-table" ? "word IDs" : "places", values: part.tiny.part === "word-table" ? view.ids : view.ids.map((_, i) => i + 1) }], [{ label: `row for ${name}`, values: table.row(row) }], `row = table[${row}]`, TINY);
    }
    case "layers":
      return step([{ label: "the word's numbers in", values: trace.combined[w] }], trace.layers.map((each, i) => ({ label: `after layer ${i + 1}`, values: each.afterFeedForward[w] })), `${LAYER_COUNT} layers, one after the other`, TINY);
    case "layer":
      return step([{ label: "the word's numbers in", values: trace.combined[w] }], [{ label: "after attention, added back", values: first.afterAttention[w] }, { label: "after feed-forward, added back", values: first.afterFeedForward[w] }], "x = x + attention(norm(x)); x = x + ffn(norm(x))", TINY);
    case "attention":
      return step([{ label: "the word's numbers, normed", values: first.normalized[w] }], [{ label: "shares of each word up to here", values: first.attention[w].slice(0, w + 1) }, { label: "mixed values", values: first.collected[w] }], "mix = Σ share × value", nearest);
    case "feed-forward":
      return step([{ label: "the word's numbers, normed", values: ffIn }], [{ label: "expanded, after ReLU", values: first.expanded[w] }, { label: "contracted, added back", values: first.afterFeedForward[w] }], "x = x + contract(relu(expand(x)))", TINY);
    case "scores": {
      const shares = first.attention[w].slice(0, w + 1);
      if (level === "whole") return step([{ label: "this word's query", values: first.queries[w] }], [{ label: "scores for each word up to here", values: first.scores[w].slice(0, w + 1) }, { label: "shares after softmax", values: shares }], `score = q · k / √${VECTOR_SIZE}; shares = softmax(scores)`, TINY);
      const j = out % (w + 1);
      const terms = termsOf("query", "key", first.queries[w].map((q, i) => [q, first.keys[j][i]]), 1 / Math.sqrt(VECTOR_SIZE));
      return step([{ label: "query", values: first.queries[w] }, { label: `key of word ${j + 1}`, values: first.keys[j] }], [{ label: "score", values: [terms.total] }], `score = (q1 × k1 + … + q${VECTOR_SIZE} × k${VECTOR_SIZE}) / √${VECTOR_SIZE}`, TINY, { terms, pick: { label: "Earlier word", count: w + 1 } });
    }
    case "mix": {
      if (level === "whole") return step([{ label: "shares", values: first.attention[w].slice(0, w + 1) }], [{ label: "mixed numbers", values: first.collected[w] }], "mix = share1 × value1 + share2 × value2 + …", TINY);
      const i = out % VECTOR_SIZE;
      const terms = termsOf("share", "value", first.attention[w].slice(0, w + 1).map((share, j) => [share, first.values[j][i]]));
      return step([{ label: "shares", values: first.attention[w].slice(0, w + 1) }], [{ label: `mixed number ${i + 1}`, values: [terms.total] }], "out = s1 × v1 + s2 × v2 + …", TINY, { terms, pick: { label: "Mixed number", count: VECTOR_SIZE } });
    }
    default:
      return step([{ label: "the word's numbers, normed", values: first.normalized[w] }], [{ label: "mixed values", values: first.collected[w] }], "mix = Σ share × value", nearest);
  }
}

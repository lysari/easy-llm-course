import { END_TOKEN, MAX_CONTEXT, TEST_ANSWER, TEST_WORDS, TRAINING_LINES, TRAINING_SEQUENCES, VOCABULARY, joinTokens } from "@/core/model/corpus";
import { nextOptions, writeText } from "@/core/model/generate";
import { TARGET_ROUNDS, type ModelLab } from "@/core/model/lab";
import { seededRandom } from "@/core/model/random";
import { type Tensor, layerNorm, multiply, parameter } from "@/core/model/tensor";
import { LAYER_COUNT, VECTOR_SIZE, countWeights, predictNext, traceForward } from "@/core/model/tiny-gpt";
import type { TinyLevel, TinyLink, TinyReadout, TinyView } from "@/features/take-apart/interfaces/model-part.interface";
import type { RankedToken } from "@/shared/interfaces/model.interface";
import { percent, shortList, showWord } from "@/shared/utils/format";

const WRITE_SEED = 7;
const TEST_IDS = TEST_WORDS.map((word) => VOCABULARY.indexOf(word));
const QUESTION_COUNT = TRAINING_SEQUENCES.reduce((sum, sequence) => sum + sequence.length - 1, 0);

const chanceOf = (options: RankedToken[], word: string) => options.find((option) => option.token === word)?.chance ?? 0;

export function viewTiny(lab: ModelLab, words: string[]): TinyView {
  const weights = lab.live;
  const ids = words.map((word) => VOCABULARY.indexOf(word));
  const all = writeText(words, weights, seededRandom(WRITE_SEED));
  const written = all.slice(words.length).map((word, i) => ({ word, chance: chanceOf(nextOptions(all.slice(0, words.length + i), weights), word) }));
  const testNow = predictNext(TEST_IDS, weights);
  return {
    lab,
    weights,
    words,
    ids,
    trace: traceForward(ids, weights),
    options: nextOptions(words, weights),
    roll: seededRandom(WRITE_SEED)(),
    written,
    isStopped: all.length < MAX_CONTEXT,
    test: { before: chanceOf(predictNext(TEST_IDS, lab.untrained), TEST_ANSWER), now: chanceOf(testNow, TEST_ANSWER), top: testNow[0].token },
  };
}

export const pickedWord = (tiny: TinyView) => tiny.written[0]?.word ?? END_TOKEN;

const idList = (words: string[], limit = 4) => `${words.slice(0, limit).map((word) => `${showWord(word)} ${VOCABULARY.indexOf(word)}`).join(" · ")}${words.length > limit ? " …" : ""}`;

const rowOf = (values: number[]) => parameter(1, values.length, Float64Array.from(values));
const normalize = (values: number[]) => Array.from(layerNorm(rowOf(values), []).data);
const times = (table: Tensor, values: number[]) => Array.from(multiply(rowOf(values), table, []).data);
const sizeOf = (tables: Tensor[]) => tables.reduce((sum, table) => sum + table.data.length, 0);

function tableReadout(title: string, table: Tensor, row: number[], level: TinyLevel): TinyReadout {
  if (level === "number") return { title: `${title}: one stored number`, shape: "1", weightCount: 1, live: row[0].toFixed(3) };
  if (level === "row") return { title: `${title}: one row`, shape: `1 × ${table.cols}`, weightCount: table.cols, live: shortList(row) };
  return { title, shape: `${table.rows} × ${table.cols}`, weightCount: table.data.length, live: shortList(row) };
}

function matrixReadout(title: string, table: Tensor, input: number[], output: number[], level: TinyLevel): TinyReadout {
  if (level === "number") return { title: `${title}: one output number`, shape: `${table.rows} × 1`, weightCount: table.rows, live: output[0].toFixed(3) };
  if (level === "multiply-add") {
    const weight = table.data[0];
    return { title: `${title}: one multiply-and-add`, shape: "1 × 1", weightCount: 1, live: `${input[0].toFixed(2)} × ${weight.toFixed(2)} = ${(input[0] * weight).toFixed(3)}` };
  }
  return { title, shape: `${table.rows} × ${table.cols}`, weightCount: table.data.length, live: shortList(output) };
}

function noWeights(title: string, shape: string, live: string): TinyReadout {
  return { title, shape, weightCount: 0, live };
}

export function readTinyPart({ part, level }: TinyLink, tiny: TinyView): TinyReadout {
  const { lab, weights, words, ids, trace, test } = tiny;
  const last = ids.length - 1;
  const top = tiny.options[0];
  const picked = showWord(pickedWord(tiny));
  const loss = lab.latestLoss === null ? "not started" : `round ${lab.round}, loss ${lab.latestLoss.toFixed(2)}`;
  const layer = weights.layers[0];
  const steps = trace.layers[0];
  const ffInput = normalize(steps.afterAttention[last]);
  const ffOutput = steps.afterFeedForward[last].map((value, i) => value - steps.afterAttention[last][i]);
  const finalInput = normalize(trace.layers[LAYER_COUNT - 1].afterFeedForward[last]);
  switch (part) {
    case "model":
      return { title: "Tiny GPT", shape: `${LAYER_COUNT} layers × ${VECTOR_SIZE} numbers per word`, weightCount: countWeights(weights), live: `next word "${showWord(top.token)}" ${percent(top.chance)}` };
    case "word-table":
      return tableReadout("Word table", weights.tokenTable, trace.tokenVectors[last], level);
    case "position-table":
      return tableReadout("Position table", weights.positionTable, trace.positionVectors[last], level);
    case "layers":
      return { title: "Layer stack", shape: `${LAYER_COUNT} layers`, weightCount: countWeights(weights) - sizeOf([weights.tokenTable, weights.positionTable, weights.outputTable]), live: shortList(trace.layers[LAYER_COUNT - 1].afterFeedForward[last]) };
    case "layer":
      return { title: "Layer 1", shape: "norm, attention, norm, feed-forward", weightCount: sizeOf([layer.query, layer.key, layer.value, layer.expand, layer.contract]), live: shortList(steps.afterFeedForward[last]) };
    case "norm":
      return level === "number" ? noWeights("Layer 1 norm: one number", "1", steps.normalized[last][0].toFixed(3)) : noWeights("Layer 1 norm", `1 × ${VECTOR_SIZE}`, shortList(steps.normalized[last]));
    case "final-norm":
      return level === "number" ? noWeights("Final norm: one number", "1", finalInput[0].toFixed(3)) : noWeights("Final norm", `1 × ${VECTOR_SIZE}`, shortList(finalInput));
    case "attention":
      return { title: "Layer 1 attention, one head", shape: `3 tables of ${VECTOR_SIZE} × ${VECTOR_SIZE}`, weightCount: sizeOf([layer.query, layer.key, layer.value]), live: shortList(steps.collected[last]) };
    case "query":
      return matrixReadout("Layer 1 query table", layer.query, steps.normalized[last], steps.queries[last], level);
    case "key":
      return matrixReadout("Layer 1 key table", layer.key, steps.normalized[last], steps.keys[last], level);
    case "value":
      return matrixReadout("Layer 1 value table", layer.value, steps.normalized[last], steps.values[last], level);
    case "scores": {
      if (level === "number") return noWeights("Layer 1: one match score", "1", steps.scores[last][0].toFixed(3));
      if (level === "multiply-add") return noWeights("Layer 1 score: one multiply-and-add", "1 × 1", `${steps.queries[last][0].toFixed(2)} × ${steps.keys[0][0].toFixed(2)}`);
      return noWeights("Layer 1 match scores", `${ids.length} × ${ids.length}`, `shares ${shortList(steps.attention[last])}`);
    }
    case "mix": {
      if (level === "number") return noWeights("Layer 1: one mixed number", "1", steps.collected[last][0].toFixed(3));
      if (level === "multiply-add") return noWeights("Layer 1 mix: one multiply-and-add", "1 × 1", `${steps.attention[last][0].toFixed(2)} × ${steps.values[0][0].toFixed(2)}`);
      return noWeights("Layer 1 weighted mix", `${ids.length} × ${VECTOR_SIZE}`, shortList(steps.collected[last]));
    }
    case "feed-forward":
      return { title: "Layer 1 feed-forward", shape: `${VECTOR_SIZE} × ${layer.expand.cols} + ${layer.expand.cols} × ${VECTOR_SIZE}`, weightCount: sizeOf([layer.expand, layer.contract]), live: shortList(ffOutput) };
    case "expand":
      return matrixReadout("Layer 1 expand table", layer.expand, ffInput, times(layer.expand, ffInput), level);
    case "relu": {
      const before = times(layer.expand, ffInput);
      if (level === "number") return noWeights("Layer 1 ReLU: one number", "1", `max(0, ${before[0].toFixed(2)}) = ${steps.expanded[last][0].toFixed(2)}`);
      const zeros = steps.expanded[last].filter((value) => value === 0).length;
      return noWeights("Layer 1 ReLU", `1 × ${before.length}`, `${zeros} of ${before.length} set to 0`);
    }
    case "contract":
      return matrixReadout("Layer 1 contract table", layer.contract, steps.expanded[last], ffOutput, level);
    case "output-table":
      return matrixReadout("Output table", weights.outputTable, finalInput, trace.logits[last], level);
    case "tokens":
      return noWeights("Tiny tokenizer: split on spaces", `${words.length} words`, idList(words));
    case "lookup":
      return noWeights("Tiny word list: one lookup", "1 row", `"${words[last]}" → ${ids[last]}`);
    case "embedding":
      return { title: "Word row + position row", shape: `${ids.length} × ${VECTOR_SIZE}`, weightCount: sizeOf([weights.tokenTable, weights.positionTable]), live: shortList(trace.combined[last]) };
    case "chances":
      return noWeights(level === "number" ? "One word's chance" : "Tiny softmax", level === "number" ? "1" : `${VOCABULARY.length} chances`, `"${showWord(top.token)}" ${percent(top.chance)}`);
    case "pick":
      if (level === "number") return noWeights("One pick", "1 roll", `roll ${tiny.roll.toFixed(2)} → "${picked}"`);
      return noWeights("Add it to the end", `${words.length + 1} words`, `… ${showWord(words[last])} + ${picked}`);
    case "write": {
      if (level === "row") return noWeights("One writing round", "1 word", `+ "${picked}"`);
      if (level === "number") return noWeights("Stop check", "2 checks", tiny.isStopped ? `picked stop after ${tiny.written.length} words` : `full at ${MAX_CONTEXT} words`);
      const added = tiny.written.map((round) => round.word);
      return noWeights("Tiny writing loop", `up to ${MAX_CONTEXT} words`, added.length > 0 ? `"… ${joinTokens([words[last], ...added])}"` : "stops right away");
    }
    case "corpus":
      if (level === "row") return noWeights("One sentence", "1 sentence", joinTokens(TRAINING_LINES[0].text.split(" ")));
      return noWeights("Tiny corpus", `${TRAINING_LINES.length} sentences`, `${TRAINING_SEQUENCES.length} with repeats · ${QUESTION_COUNT} questions`);
    case "vocabulary":
      if (level === "number") return noWeights("Tiny word list: one row", "1 row", `"${words[last]}" = ${ids[last]}`);
      return noWeights("Tiny word list", `${VOCABULARY.length} words`, idList(VOCABULARY));
    case "training":
      return noWeights(level === "row" ? "One tiny training round" : "Tiny pre-training", `${TARGET_ROUNDS} rounds on ${TRAINING_SEQUENCES.length} sentences`, loss);
    case "loss":
      if (level === "row") return noWeights(`Guess: chance of "${TEST_ANSWER}"`, "1 chance", percent(test.now));
      return noWeights("One loss term", "1 number", `−ln(${test.now.toFixed(3)}) = ${(-Math.log(test.now)).toFixed(2)}`);
    case "slope":
      return noWeights(`Slope of the first number of "${VOCABULARY[0]}"`, "1 number", weights.tokenTable.grad[0].toFixed(4));
    case "adam":
      return { title: `Word table: first number of "${VOCABULARY[0]}"`, shape: "1 stored number", weightCount: 1, live: `${lab.untrained.tokenTable.data[0].toFixed(3)} → ${weights.tokenTable.data[0].toFixed(3)}` };
    case "evaluation":
      if (level === "number") return noWeights("One test question", "1 question", `top "${showWord(test.top)}": ${test.top === TEST_ANSWER ? "right" : "wrong"}`);
      return noWeights("Tiny test sentence", "1 question", `"${TEST_ANSWER}" ${percent(test.before)} → ${percent(test.now)}`);
  }
}

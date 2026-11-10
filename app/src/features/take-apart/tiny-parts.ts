import { END_ID, END_TOKEN, MAX_CONTEXT, TEST_ANSWER, TEST_WORDS, TRAINING_LINES, TRAINING_SEQUENCES, VOCABULARY, joinTokens } from "@/core/model/corpus";
import { nextOptions, writeText } from "@/core/model/generate";
import { TARGET_ROUNDS, type ModelLab } from "@/core/model/lab";
import { seededRandom } from "@/core/model/random";
import { type Tensor, layerNorm, multiply, parameter } from "@/core/model/tensor";
import { LAYER_COUNT, VECTOR_SIZE, countWeights, predictNext, traceForward } from "@/core/model/tiny-gpt";
import type { NoneToLearn, TinyLevel, TinyLink, TinyReadout, TinyView } from "@/features/take-apart/interfaces/model-part.interface";
import type { Amount } from "@/features/take-apart/interfaces/real-facts.interface";
import type { RankedToken } from "@/shared/interfaces/model.interface";
import { percent, shortList, showWord } from "@/shared/utils/format";

const WRITE_SEED = 7;
const TEST_IDS = TEST_WORDS.map((word) => VOCABULARY.indexOf(word));
const QUESTION_COUNT = TRAINING_SEQUENCES.reduce((sum, sequence) => sum + sequence.length - 1, 0);
const LONGEST_TEXT = Math.max(...TRAINING_SEQUENCES.map((sequence) => sequence.length));

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

const numbers = (value: number): Amount => ({ value, unit: value === 1 ? "number" : "numbers" });

function tableReadout(title: string, table: Tensor, row: number[], level: TinyLevel): TinyReadout {
  if (level === "number") return { title: `${title}: one stored number`, shape: "1", weightCount: 1, headline: { value: table.data.length, unit: "in the table" }, live: row[0].toFixed(3) };
  if (level === "row") return { title: `${title}: one row`, shape: `1 × ${table.cols}`, weightCount: table.cols, headline: numbers(table.cols), live: shortList(row) };
  return { title, shape: `${table.rows} × ${table.cols}`, weightCount: table.data.length, headline: numbers(table.data.length), live: shortList(row) };
}

function matrixReadout(title: string, table: Tensor, input: number[], output: number[], level: TinyLevel): TinyReadout {
  if (level === "number") return { title: `${title}: one output number`, shape: `${table.rows} × 1`, weightCount: table.rows, headline: numbers(table.rows), live: output[0].toFixed(3) };
  if (level === "multiply-add") {
    const weight = table.data[0];
    return { title: `${title}: one multiply-and-add`, shape: "1 × 1", weightCount: 1, headline: { value: table.rows, unit: "per output number" }, live: `${input[0].toFixed(2)} × ${weight.toFixed(2)} = ${(input[0] * weight).toFixed(3)}` };
  }
  return { title, shape: `${table.rows} × ${table.cols}`, weightCount: table.data.length, headline: numbers(table.data.length), live: shortList(output) };
}

const noScale = (name: string): NoneToLearn => ({ reason: `Our tiny model's ${name} has no scale or shift numbers, so it has nothing to learn.` });

function noWeights(title: string, shape: string, headline: Amount | NoneToLearn, live: string): TinyReadout {
  return { title, shape, weightCount: 0, headline, live };
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
    case "model": {
      const modelNumbers = countWeights(weights);
      return { title: "Tiny GPT", shape: `${LAYER_COUNT} layers × ${VECTOR_SIZE} numbers per word`, weightCount: modelNumbers, headline: numbers(modelNumbers), live: `next word "${showWord(top.token)}" ${percent(top.chance)}` };
    }
    case "word-table":
      return tableReadout("Word table", weights.tokenTable, trace.tokenVectors[last], level);
    case "position-table":
      return tableReadout("Position table", weights.positionTable, trace.positionVectors[last], level);
    case "layers": {
      const layerNumbers = countWeights(weights) - sizeOf([weights.tokenTable, weights.positionTable, weights.outputTable]);
      return { title: "Layer stack", shape: `${LAYER_COUNT} layers`, weightCount: layerNumbers, headline: numbers(layerNumbers), live: shortList(trace.layers[LAYER_COUNT - 1].afterFeedForward[last]) };
    }
    case "layer": {
      const layerNumbers = sizeOf([layer.query, layer.key, layer.value, layer.expand, layer.contract]);
      return { title: "Layer 1", shape: "norm, attention, norm, feed-forward", weightCount: layerNumbers, headline: numbers(layerNumbers), live: shortList(steps.afterFeedForward[last]) };
    }
    case "norm":
      return level === "number" ? noWeights("Layer 1 norm: one number", "1", noScale("norm"), steps.normalized[last][0].toFixed(3)) : noWeights("Layer 1 norm", `1 × ${VECTOR_SIZE}`, noScale("norm"), shortList(steps.normalized[last]));
    case "final-norm":
      return level === "number" ? noWeights("Final norm: one number", "1", noScale("final norm"), finalInput[0].toFixed(3)) : noWeights("Final norm", `1 × ${VECTOR_SIZE}`, noScale("final norm"), shortList(finalInput));
    case "attention": {
      const attentionNumbers = sizeOf([layer.query, layer.key, layer.value]);
      return { title: "Layer 1 attention, one head", shape: `3 tables of ${VECTOR_SIZE} × ${VECTOR_SIZE}`, weightCount: attentionNumbers, headline: numbers(attentionNumbers), live: shortList(steps.collected[last]) };
    }
    case "query":
      return matrixReadout("Layer 1 query table", layer.query, steps.normalized[last], steps.queries[last], level);
    case "key":
      return matrixReadout("Layer 1 key table", layer.key, steps.normalized[last], steps.keys[last], level);
    case "value":
      return matrixReadout("Layer 1 value table", layer.value, steps.normalized[last], steps.values[last], level);
    case "scores": {
      if (level === "number") return noWeights("Layer 1: one match score", "1", { value: VECTOR_SIZE, unit: "pairs multiplied" }, steps.scores[last][0].toFixed(3));
      if (level === "multiply-add") return noWeights("Layer 1 score: one multiply-and-add", "1 × 1", { value: VECTOR_SIZE, unit: "per score" }, `${steps.queries[last][0].toFixed(2)} × ${steps.keys[0][0].toFixed(2)}`);
      return noWeights("Layer 1 match scores", `${ids.length} × ${ids.length}`, { value: MAX_CONTEXT, unit: "words at most" }, `shares ${shortList(steps.attention[last])}`);
    }
    case "mix": {
      if (level === "number") return noWeights("Layer 1: one mixed number", "1", { value: MAX_CONTEXT, unit: "words at most" }, steps.collected[last][0].toFixed(3));
      if (level === "multiply-add") return noWeights("Layer 1 mix: one multiply-and-add", "1 × 1", { value: MAX_CONTEXT, unit: "per mixed number" }, `${steps.attention[last][0].toFixed(2)} × ${steps.values[0][0].toFixed(2)}`);
      return noWeights("Layer 1 weighted mix", `${ids.length} × ${VECTOR_SIZE}`, { value: VECTOR_SIZE, unit: "numbers per word" }, shortList(steps.collected[last]));
    }
    case "feed-forward": {
      const feedForwardNumbers = sizeOf([layer.expand, layer.contract]);
      return { title: "Layer 1 feed-forward", shape: `${VECTOR_SIZE} × ${layer.expand.cols} + ${layer.expand.cols} × ${VECTOR_SIZE}`, weightCount: feedForwardNumbers, headline: numbers(feedForwardNumbers), live: shortList(ffOutput) };
    }
    case "expand":
      return matrixReadout("Layer 1 expand table", layer.expand, ffInput, times(layer.expand, ffInput), level);
    case "relu": {
      const before = times(layer.expand, ffInput);
      if (level === "number") return noWeights("Layer 1 ReLU: one number", "1", { value: before.length, unit: "per word" }, `max(0, ${before[0].toFixed(2)}) = ${steps.expanded[last][0].toFixed(2)}`);
      const zeros = steps.expanded[last].filter((value) => value === 0).length;
      return noWeights("Layer 1 ReLU", `1 × ${before.length}`, { value: before.length, unit: "numbers per word" }, `${zeros} of ${before.length} set to 0`);
    }
    case "contract":
      return matrixReadout("Layer 1 contract table", layer.contract, steps.expanded[last], ffOutput, level);
    case "output-table":
      return matrixReadout("Output table", weights.outputTable, finalInput, trace.logits[last], level);
    case "tokens":
      return noWeights("Tiny tokenizer: split on spaces", `${words.length} words`, { value: VOCABULARY.length, unit: "words" }, idList(words));
    case "lookup":
      return noWeights("Tiny word list: one lookup", "1 row", { value: VOCABULARY.length, unit: "words in the list" }, `"${words[last]}" → ${ids[last]}`);
    case "embedding":
      return { title: "Word row + position row", shape: `${ids.length} × ${VECTOR_SIZE}`, weightCount: sizeOf([weights.tokenTable, weights.positionTable]), headline: { value: VECTOR_SIZE, unit: "numbers per word" }, live: shortList(trace.combined[last]) };
    case "chances":
      return noWeights(level === "number" ? "One word's chance" : "Tiny softmax", level === "number" ? "1" : `${VOCABULARY.length} chances`, { value: VOCABULARY.length, unit: level === "number" ? "per new word" : "chances" }, `"${showWord(top.token)}" ${percent(top.chance)}`);
    case "pick":
      if (level === "number") return noWeights("One pick", "1 roll", { value: VOCABULARY.length, unit: "words to pick from" }, `roll ${tiny.roll.toFixed(2)} → "${picked}"`);
      return noWeights("Add it to the end", `${words.length + 1} words`, { value: MAX_CONTEXT, unit: "words at most" }, `… ${showWord(words[last])} + ${picked}`);
    case "write": {
      if (level === "row") return noWeights("One writing round", "1 word", { value: 1, unit: "word added" }, `+ "${picked}"`);
      if (level === "number") return noWeights("Stop check", "stop word, or full length", { value: END_ID, unit: "is the stop word's ID" }, tiny.isStopped ? `picked stop after ${tiny.written.length} words` : `full at ${MAX_CONTEXT} words`);
      const added = tiny.written.map((round) => round.word);
      return noWeights("Tiny writing loop", `up to ${MAX_CONTEXT} words`, { value: MAX_CONTEXT, unit: "words at most" }, added.length > 0 ? `"… ${joinTokens([words[last], ...added])}"` : "stops right away");
    }
    case "corpus":
      if (level === "row") return noWeights("One hand-written sentence", "1 sentence", { value: TRAINING_LINES.length, unit: "sentences, all hand-written" }, joinTokens(TRAINING_LINES[0].text.split(" ")));
      return noWeights("Tiny corpus, written by hand for this course", `${TRAINING_LINES.length} sentences`, { value: TRAINING_LINES.length, unit: "sentences" }, `${TRAINING_SEQUENCES.length} with repeats · ${QUESTION_COUNT} questions`);
    case "vocabulary":
      if (level === "number") return noWeights("Tiny word list: one row", "1 row", { value: VOCABULARY.length, unit: "IDs given" }, `"${words[last]}" = ${ids[last]}`);
      return noWeights("Tiny word list", `${VOCABULARY.length} words`, { value: VOCABULARY.length, unit: "words" }, idList(VOCABULARY));
    case "training":
      if (level === "row") return noWeights("One tiny training round", `${TARGET_ROUNDS} rounds on ${TRAINING_SEQUENCES.length} sentences`, { value: TRAINING_SEQUENCES.length, unit: "texts per round" }, loss);
      return noWeights("Tiny pre-training", `${TARGET_ROUNDS} rounds on ${TRAINING_SEQUENCES.length} sentences`, { value: LONGEST_TEXT, unit: "words per text" }, loss);
    case "loss":
      if (level === "row") return noWeights(`Guess: chance of "${TEST_ANSWER}"`, "1 chance", { value: VOCABULARY.length, unit: "words to choose from" }, percent(test.now));
      return noWeights("One loss term", "1 number", { value: 1, unit: "number per word" }, `−ln(${test.now.toFixed(3)}) = ${(-Math.log(test.now)).toFixed(2)}`);
    case "slope":
      return noWeights(`Slope of the first number of "${VOCABULARY[0]}"`, "1 number", { value: countWeights(weights), unit: "slopes per round" }, weights.tokenTable.grad[0].toFixed(4));
    case "adam":
      return { title: `Word table: first number of "${VOCABULARY[0]}"`, shape: "1 stored number", weightCount: 1, headline: { value: countWeights(weights), unit: "numbers nudged per round" }, live: `${lab.untrained.tokenTable.data[0].toFixed(3)} → ${weights.tokenTable.data[0].toFixed(3)}` };
    case "evaluation":
      if (level === "number") return noWeights("One test question", "1 question", { value: 0, unit: "examples shown first" }, `top "${showWord(test.top)}": ${test.top === TEST_ANSWER ? "right" : "wrong"}`);
      return noWeights("Tiny test sentence", "1 question", { value: Math.round(test.now * 100), unit: "% sure of the right word" }, `"${TEST_ANSWER}" ${percent(test.before)} → ${percent(test.now)}`);
  }
}

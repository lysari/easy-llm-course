import type { LayerTrace, LayerWeights, ForwardTrace, ModelShape, RankedToken, TinyGptWeights } from "@/shared/interfaces/model.interface";
import { MAX_CONTEXT, VOCABULARY } from "./corpus";
import { seededRandom } from "./random";
import { Tensor, add, causalSoftmax, crossEntropy, layerNorm, lookupRows, matchScores, multiply, parameter, relu, runBackward } from "./tensor";

export const VECTOR_SIZE = 24;
export const LAYER_COUNT = 2;
export const EXPANDED_SIZE = VECTOR_SIZE * 4;
const DEFAULT_SEED = 42;
const FULL_LAYER = { hasAttention: true, hasFeedForward: true };
const DEFAULT_SHAPE: ModelShape = { vectorSize: VECTOR_SIZE, layers: Array.from({ length: LAYER_COUNT }, () => FULL_LAYER) };

export function createWeights(seed = DEFAULT_SEED, shape = DEFAULT_SHAPE): TinyGptWeights {
  const uniform = seededRandom(seed);
  const normal = () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
  const table = (rows: number, cols: number, spread: number) =>
    parameter(rows, cols, Float64Array.from({ length: rows * cols }, () => normal() * spread));
  const { vectorSize } = shape;
  const expandedSize = vectorSize * 4;
  const layerSpread = 1 / Math.sqrt(vectorSize);
  return {
    tokenTable: table(VOCABULARY.length, vectorSize, 0.3),
    positionTable: table(MAX_CONTEXT, vectorSize, 0.1),
    layers: shape.layers.map((layer) => ({
      ...layer,
      query: table(vectorSize, vectorSize, layerSpread),
      key: table(vectorSize, vectorSize, layerSpread),
      value: table(vectorSize, vectorSize, layerSpread),
      expand: table(vectorSize, expandedSize, layerSpread),
      contract: table(expandedSize, vectorSize, 1 / Math.sqrt(expandedSize)),
    })),
    outputTable: table(vectorSize, VOCABULARY.length, layerSpread),
  };
}

export function allTables(weights: TinyGptWeights): { name: string; tensor: Tensor }[] {
  return [
    { name: "Token table", tensor: weights.tokenTable },
    { name: "Position table", tensor: weights.positionTable },
    ...weights.layers.flatMap((layer, i) => [
      ...(layer.hasAttention
        ? [
            { name: `Layer ${i + 1} attention: question`, tensor: layer.query },
            { name: `Layer ${i + 1} attention: name tag`, tensor: layer.key },
            { name: `Layer ${i + 1} attention: information`, tensor: layer.value },
          ]
        : []),
      ...(layer.hasFeedForward
        ? [
            { name: `Layer ${i + 1} feed-forward: expand`, tensor: layer.expand },
            { name: `Layer ${i + 1} feed-forward: contract`, tensor: layer.contract },
          ]
        : []),
    ]),
    { name: "Output table", tensor: weights.outputTable },
  ];
}

export function copyWeights(weights: TinyGptWeights): TinyGptWeights {
  const clone = (tensor: Tensor) => parameter(tensor.rows, tensor.cols, tensor.data.slice());
  return {
    tokenTable: clone(weights.tokenTable),
    positionTable: clone(weights.positionTable),
    layers: weights.layers.map((layer) => ({
      hasAttention: layer.hasAttention,
      hasFeedForward: layer.hasFeedForward,
      query: clone(layer.query),
      key: clone(layer.key),
      value: clone(layer.value),
      expand: clone(layer.expand),
      contract: clone(layer.contract),
    })),
    outputTable: clone(weights.outputTable),
  };
}

export const countWeights = (weights: TinyGptWeights) =>
  allTables(weights).reduce((sum, { tensor }) => sum + tensor.data.length, 0);

function runAttention(input: Tensor, layer: LayerWeights, tape: Tensor[]) {
  const normalized = layerNorm(input, tape);
  const queries = multiply(normalized, layer.query, tape);
  const keys = multiply(normalized, layer.key, tape);
  const values = multiply(normalized, layer.value, tape);
  const scores = matchScores(queries, keys, 1 / Math.sqrt(layer.query.cols), tape);
  const attention = causalSoftmax(scores, tape);
  const collected = multiply(attention, values, tape);
  const afterAttention = add(input, collected, tape);
  return { normalized, queries, keys, values, scores, attention, collected, afterAttention };
}

function runFeedForward(input: Tensor, layer: LayerWeights, tape: Tensor[]) {
  const expanded = relu(multiply(layerNorm(input, tape), layer.expand, tape), tape);
  const afterFeedForward = add(input, multiply(expanded, layer.contract, tape), tape);
  return { expanded, afterFeedForward };
}

function runForward(ids: number[], weights: TinyGptWeights, tape: Tensor[]) {
  const tokenVectors = lookupRows(weights.tokenTable, ids, tape);
  const positionVectors = lookupRows(weights.positionTable, ids.map((_, i) => i), tape);
  const combined = add(tokenVectors, positionVectors, tape);
  const layers: (ReturnType<typeof runAttention> & ReturnType<typeof runFeedForward>)[] = [];
  let current = combined;
  for (const layer of weights.layers) {
    const attention = layer.hasAttention ? runAttention(current, layer, tape) : null;
    const afterAttention = attention?.afterAttention ?? current;
    const feedForward = layer.hasFeedForward ? runFeedForward(afterAttention, layer, tape) : null;
    if (attention && feedForward) layers.push({ ...attention, ...feedForward });
    current = feedForward?.afterFeedForward ?? afterAttention;
  }
  const logits = multiply(layerNorm(current, tape), weights.outputTable, tape);
  return { tokenVectors, positionVectors, combined, layers, logits };
}

export function traceForward(ids: number[], weights: TinyGptWeights): ForwardTrace {
  const result = runForward(ids, weights, []);
  const layers: LayerTrace[] = result.layers.map((layer) => ({
    normalized: layer.normalized.toRows(),
    queries: layer.queries.toRows(),
    keys: layer.keys.toRows(),
    values: layer.values.toRows(),
    scores: layer.scores.toRows(),
    attention: layer.attention.toRows(),
    collected: layer.collected.toRows(),
    afterAttention: layer.afterAttention.toRows(),
    expanded: layer.expanded.toRows(),
    afterFeedForward: layer.afterFeedForward.toRows(),
  }));
  return {
    tokenVectors: result.tokenVectors.toRows(),
    positionVectors: result.positionVectors.toRows(),
    combined: result.combined.toRows(),
    layers,
    logits: result.logits.toRows(),
  };
}

export function lossAndGradients(sequences: number[][], weights: TinyGptWeights): number {
  const tables = allTables(weights);
  for (const { tensor } of tables) tensor.grad.fill(0);
  let totalLoss = 0;
  for (const sequence of sequences) {
    const tape: Tensor[] = [];
    const { logits } = runForward(sequence.slice(0, -1), weights, tape);
    const loss = crossEntropy(logits, sequence.slice(1), tape);
    runBackward(loss, tape);
    totalLoss += loss.data[0];
  }
  for (const { tensor } of tables) {
    for (let i = 0; i < tensor.grad.length; i++) tensor.grad[i] /= sequences.length;
  }
  return totalLoss / sequences.length;
}

export function measureLoss(sequences: number[][], weights: TinyGptWeights): number {
  let total = 0;
  for (const sequence of sequences) {
    const tape: Tensor[] = [];
    const { logits } = runForward(sequence.slice(0, -1), weights, tape);
    total += crossEntropy(logits, sequence.slice(1), tape).data[0];
  }
  return total / sequences.length;
}

export function predictNext(ids: number[], weights: TinyGptWeights): RankedToken[] {
  const { logits } = runForward(ids, weights, []);
  const row = logits.row(logits.rows - 1);
  const top = Math.max(...row);
  const raw = row.map((score) => Math.exp(score - top));
  const total = raw.reduce((sum, value) => sum + value, 0);
  return VOCABULARY.map((token, id) => ({ token, id, score: row[id], chance: raw[id] / total })).sort((a, b) => b.chance - a.chance);
}

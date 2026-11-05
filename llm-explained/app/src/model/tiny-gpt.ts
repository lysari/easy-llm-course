import type { BlockTrace, BlockWeights, ForwardTrace, RankedToken, TinyGptWeights } from "../interfaces/model.interface";
import { MAX_CONTEXT, VOCABULARY } from "./corpus";
import { seededRandom } from "./random";
import { Tensor, add, causalSoftmax, crossEntropy, layerNorm, lookupRows, matchScores, multiply, parameter, relu, runBackward } from "./tensor";

export const VECTOR_SIZE = 24;
export const BLOCK_COUNT = 2;
export const EXPANDED_SIZE = VECTOR_SIZE * 4;
const DEFAULT_SEED = 42;

export function createWeights(seed = DEFAULT_SEED): TinyGptWeights {
  const uniform = seededRandom(seed);
  const normal = () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
  const table = (rows: number, cols: number, spread: number) =>
    parameter(rows, cols, Float64Array.from({ length: rows * cols }, () => normal() * spread));
  const layerSpread = 1 / Math.sqrt(VECTOR_SIZE);
  return {
    tokenTable: table(VOCABULARY.length, VECTOR_SIZE, 0.3),
    positionTable: table(MAX_CONTEXT, VECTOR_SIZE, 0.1),
    blocks: Array.from({ length: BLOCK_COUNT }, () => ({
      query: table(VECTOR_SIZE, VECTOR_SIZE, layerSpread),
      key: table(VECTOR_SIZE, VECTOR_SIZE, layerSpread),
      value: table(VECTOR_SIZE, VECTOR_SIZE, layerSpread),
      expand: table(VECTOR_SIZE, EXPANDED_SIZE, layerSpread),
      contract: table(EXPANDED_SIZE, VECTOR_SIZE, 1 / Math.sqrt(EXPANDED_SIZE)),
    })),
    outputTable: table(VECTOR_SIZE, VOCABULARY.length, layerSpread),
  };
}

export function allTables(weights: TinyGptWeights): { name: string; tensor: Tensor }[] {
  return [
    { name: "Token table", tensor: weights.tokenTable },
    { name: "Position table", tensor: weights.positionTable },
    ...weights.blocks.flatMap((block, i) => [
      { name: `Layer ${i + 1} attention: question`, tensor: block.query },
      { name: `Layer ${i + 1} attention: name tag`, tensor: block.key },
      { name: `Layer ${i + 1} attention: information`, tensor: block.value },
      { name: `Layer ${i + 1} feed-forward: expand`, tensor: block.expand },
      { name: `Layer ${i + 1} feed-forward: contract`, tensor: block.contract },
    ]),
    { name: "Output table", tensor: weights.outputTable },
  ];
}

export function copyWeights(weights: TinyGptWeights): TinyGptWeights {
  const clone = (tensor: Tensor) => parameter(tensor.rows, tensor.cols, tensor.data.slice());
  return {
    tokenTable: clone(weights.tokenTable),
    positionTable: clone(weights.positionTable),
    blocks: weights.blocks.map((block) => ({
      query: clone(block.query),
      key: clone(block.key),
      value: clone(block.value),
      expand: clone(block.expand),
      contract: clone(block.contract),
    })),
    outputTable: clone(weights.outputTable),
  };
}

export const countWeights = (weights: TinyGptWeights) =>
  allTables(weights).reduce((sum, { tensor }) => sum + tensor.data.length, 0);

function runBlock(input: Tensor, block: BlockWeights, tape: Tensor[]) {
  const normalized = layerNorm(input, tape);
  const queries = multiply(normalized, block.query, tape);
  const keys = multiply(normalized, block.key, tape);
  const values = multiply(normalized, block.value, tape);
  const scores = matchScores(queries, keys, 1 / Math.sqrt(VECTOR_SIZE), tape);
  const attention = causalSoftmax(scores, tape);
  const collected = multiply(attention, values, tape);
  const afterAttention = add(input, collected, tape);
  const expanded = relu(multiply(layerNorm(afterAttention, tape), block.expand, tape), tape);
  const afterFeedForward = add(afterAttention, multiply(expanded, block.contract, tape), tape);
  return { normalized, queries, keys, values, scores, attention, collected, afterAttention, expanded, afterFeedForward };
}

function runForward(ids: number[], weights: TinyGptWeights, tape: Tensor[]) {
  const tokenVectors = lookupRows(weights.tokenTable, ids, tape);
  const positionVectors = lookupRows(weights.positionTable, ids.map((_, i) => i), tape);
  const combined = add(tokenVectors, positionVectors, tape);
  const blocks: ReturnType<typeof runBlock>[] = [];
  let current = combined;
  for (const block of weights.blocks) {
    const result = runBlock(current, block, tape);
    blocks.push(result);
    current = result.afterFeedForward;
  }
  const logits = multiply(layerNorm(current, tape), weights.outputTable, tape);
  return { tokenVectors, positionVectors, combined, blocks, logits };
}

export function traceForward(ids: number[], weights: TinyGptWeights): ForwardTrace {
  const result = runForward(ids, weights, []);
  const blocks: BlockTrace[] = result.blocks.map((block) => ({
    normalized: block.normalized.toRows(),
    queries: block.queries.toRows(),
    keys: block.keys.toRows(),
    values: block.values.toRows(),
    scores: block.scores.toRows(),
    attention: block.attention.toRows(),
    collected: block.collected.toRows(),
    afterAttention: block.afterAttention.toRows(),
    expanded: block.expanded.toRows(),
    afterFeedForward: block.afterFeedForward.toRows(),
  }));
  return {
    tokenVectors: result.tokenVectors.toRows(),
    positionVectors: result.positionVectors.toRows(),
    combined: result.combined.toRows(),
    blocks,
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

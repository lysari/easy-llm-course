import type { Tensor } from "../model/tensor";

export type Matrix = number[][];

export interface TinyGptWeights {
  tokenTable: Tensor;
  positionTable: Tensor;
  blocks: BlockWeights[];
  outputTable: Tensor;
}

export interface BlockWeights {
  query: Tensor;
  key: Tensor;
  value: Tensor;
  expand: Tensor;
  contract: Tensor;
}

export interface BlockTrace {
  normalized: Matrix;
  queries: Matrix;
  keys: Matrix;
  values: Matrix;
  scores: Matrix;
  attention: Matrix;
  collected: Matrix;
  afterAttention: Matrix;
  expanded: Matrix;
  afterFeedForward: Matrix;
}

export interface ForwardTrace {
  tokenVectors: Matrix;
  positionVectors: Matrix;
  combined: Matrix;
  blocks: BlockTrace[];
  logits: Matrix;
}

export interface RankedToken {
  token: string;
  id: number;
  score: number;
  chance: number;
}

export interface LossPoint {
  step: number;
  loss: number;
}

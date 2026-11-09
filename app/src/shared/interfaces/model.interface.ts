import type { Tensor } from "@/core/model/tensor";

export type Matrix = number[][];

export interface TinyGptWeights {
  tokenTable: Tensor;
  positionTable: Tensor;
  layers: LayerWeights[];
  outputTable: Tensor;
}

export interface LayerWeights extends LayerShape {
  query: Tensor;
  key: Tensor;
  value: Tensor;
  expand: Tensor;
  contract: Tensor;
}

export interface LayerShape {
  hasAttention: boolean;
  hasFeedForward: boolean;
}

export interface ModelShape {
  vectorSize: number;
  layers: LayerShape[];
}

export interface Sampling {
  temperature: number;
  topK: number;
  topP: number;
}

export interface LayerTrace {
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
  layers: LayerTrace[];
  logits: Matrix;
}

export interface RankedToken {
  token: string;
  id: number;
  score: number;
  chance: number;
}

export interface LossPoint {
  round: number;
  loss: number;
}

import type { ModelLab } from "@/core/model/lab";
import type { Formula } from "@/shared/interfaces/formula.interface";
import type { ForwardTrace, RankedToken, TinyGptWeights } from "@/shared/interfaces/model.interface";

export type ModelId = "gpt-2-small" | "llama-3-8b";

export type ModeId = "use" | "build";

export type GroundZero = "multiply-add" | "stored-number" | "action";

export type TinyPartId =
  | "model"
  | "word-table"
  | "position-table"
  | "layers"
  | "layer"
  | "norm"
  | "attention"
  | "query"
  | "key"
  | "value"
  | "scores"
  | "mix"
  | "feed-forward"
  | "expand"
  | "relu"
  | "contract"
  | "final-norm"
  | "output-table"
  | "tokens"
  | "lookup"
  | "embedding"
  | "chances"
  | "pick"
  | "write"
  | "corpus"
  | "vocabulary"
  | "training"
  | "loss"
  | "slope"
  | "adam"
  | "evaluation";

export type TinyLevel = "whole" | "row" | "number" | "multiply-add";

export interface RealNote {
  label: string;
  value: string;
  source: string;
}

export interface RealFacts {
  shape: string;
  weightCount: number;
  input: string;
  output: string;
  notes?: RealNote[];
}

export interface StepFacts {
  shape: string;
  notes: RealNote[];
}

export interface TinyLink {
  part: TinyPartId;
  level: TinyLevel;
  missing?: string;
}

export interface ModelPart {
  id: string;
  icon: string;
  title: string;
  copies: number;
  facts: RealFacts;
  tiny: TinyLink;
  lesson: string;
  reverseLesson: string;
  explanation: string;
  why: string;
  formula?: Formula;
  code?: string;
  groundZero?: GroundZero;
  children?: ModelPart[];
}

export interface RealModel {
  id: ModelId;
  title: string;
  root: ModelPart;
  modes: Record<ModeId, ModelPart>;
}

export interface TinyReadout {
  title: string;
  shape: string;
  weightCount: number;
  live: string;
}

export interface WrittenWord {
  word: string;
  chance: number;
}

export interface TestChance {
  before: number;
  now: number;
  top: string;
}

export interface TinyView {
  lab: ModelLab;
  weights: TinyGptWeights;
  words: string[];
  ids: number[];
  trace: ForwardTrace;
  options: RankedToken[];
  roll: number;
  written: WrittenWord[];
  isStopped: boolean;
  test: TestChance;
}

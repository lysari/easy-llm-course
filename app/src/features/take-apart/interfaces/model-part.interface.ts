import type { ModelLab } from "@/core/model/lab";
import type { Formula } from "@/shared/interfaces/formula.interface";
import type { ForwardTrace, RankedToken, TinyGptWeights } from "@/shared/interfaces/model.interface";
import type { Amount, RealFacts } from "./real-facts.interface";

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

export interface TinyLink {
  part: TinyPartId;
  level: TinyLevel;
  missing?: string;
}

export type DemoId = "copy-document";

export interface PartStory {
  problem: string;
  demo: DemoId;
  when: string;
  feeds: string;
  nextId?: string;
}

export interface ModelPart {
  id: string;
  icon: string;
  title: string;
  summary: string;
  copies: number;
  facts: RealFacts;
  tiny: TinyLink;
  lesson: string;
  reverseLesson: string;
  explanation: string;
  why: string;
  story?: PartStory;
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

export interface NoneToLearn {
  reason: string;
}

export interface TinyReadout {
  title: string;
  shape: string;
  weightCount: number;
  headline: Amount | NoneToLearn;
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

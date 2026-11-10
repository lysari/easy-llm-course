import type { SourceId } from "@/features/inside-real-models/trees/sources";

export interface RealNote {
  label: string;
  value: string;
  source: SourceId;
}

export interface KeyFact extends RealNote {
  isComputed: boolean;
}

export interface Amount {
  value: number;
  unit: string;
  isOver?: boolean;
  isComputed?: boolean;
  isId?: boolean;
}

export interface Unstated {
  text: string;
}

export type Headline = Amount | Unstated;

export interface RealFacts {
  shape: string;
  weightCount: number;
  headline: Headline;
  input: string;
  output: string;
  inputDetail?: string;
  outputDetail?: string;
  sources: SourceId[];
  notes?: RealNote[];
  reasons?: RealNote[];
  isPerEarlierWord?: boolean;
}

export interface StepFacts {
  shape: string;
  headline: Headline;
  sources: SourceId[];
  notes: RealNote[];
  reasons?: RealNote[];
}

export type BuildStepId =
  | "data"
  | "gather"
  | "filter"
  | "dedupe"
  | "tokenizer"
  | "merge-round"
  | "count-pair"
  | "learn-merge"
  | "assign-id"
  | "pre-training"
  | "training-round"
  | "guess"
  | "loss-term"
  | "slope"
  | "adam-update"
  | "post-training"
  | "sft"
  | "sft-term"
  | "preference"
  | "preference-pair"
  | "evaluation"
  | "benchmarks"
  | "test-question"
  | "safety"
  | "safety-check"
  | "quantization"
  | "round-number";

export type BuildFacts = Record<BuildStepId, StepFacts>;

export type UseStepId =
  | "tokenize"
  | "split"
  | "apply-merge"
  | "lookup"
  | "embedding"
  | "output"
  | "softmax"
  | "chance"
  | "generate"
  | "write-round"
  | "pick"
  | "append"
  | "stop-check"
  | "kv-cache";

export type UseReasons = Record<UseStepId, RealNote[]>;

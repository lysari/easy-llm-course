import type { ModelLab } from "@/core/model/lab";
import type { ConceptName } from "@/shared/interfaces/concept.interface";
import type { Formula } from "@/shared/interfaces/formula.interface";
import type { ForwardTrace, RankedToken, TinyGptWeights } from "@/shared/interfaces/model.interface";
import type { RealFacts } from "./real-facts.interface";

export type ModelId = "gpt-2-small" | "llama-3-8b";

export type ModeId = "use" | "build";

export type GroundZero = "multiply-add" | "stored-number" | "action";

export type TinyPartId =
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
  | "embedding";

export type TinyLevel = "whole" | "row" | "number" | "multiply-add";

export interface TinyLink {
  part: TinyPartId;
  level: TinyLevel;
  missing?: string;
}

export type DemoId =
  | "copy-document"
  | "data-pipeline"
  | "quality-check"
  | "dedupe"
  | "gpt-2-tokenizer"
  | "bpe-learn"
  | "architecture"
  | "tiny-part"
  | "training"
  | "chat-format"
  | "answer-loss"
  | "preference"
  | "tiny-test"
  | "safety"
  | "quantize"
  | "next-word"
  | "generate"
  | "kv-cache";

export interface PartStory {
  problem: string;
  demo: DemoId;
  when: string;
  feeds: string;
  nextId?: string;
}

export interface ModelPart {
  id: string;
  icon: ConceptName;
  title: string;
  summary: string;
  copies: number;
  facts: RealFacts;
  tiny?: TinyLink;
  explanation: string;
  why: string;
  story?: PartStory;
  formula?: Formula;
  code?: string;
  groundZero?: GroundZero;
  children?: ModelPart[];
}

export interface DemoProps {
  part: ModelPart;
  modelId: ModelId;
  tiny: TinyView;
}

export interface RealModel {
  id: ModelId;
  title: string;
  root: ModelPart;
  modes: Record<ModeId, ModelPart>;
}

export interface TinyView {
  lab: ModelLab;
  weights: TinyGptWeights;
  words: string[];
  ids: number[];
  trace: ForwardTrace;
  options: RankedToken[];
  written: string[];
}

import { type ComponentType, type LazyExoticComponent, lazy } from "react";
import type { DemoId, DemoProps, ModelId } from "@/features/inside-real-models/interfaces/model-part.interface";

type DemoUse = boolean | ModelId;

interface Demo {
  component: LazyExoticComponent<ComponentType<DemoProps>>;
  usesSentence: DemoUse;
  usesTraining: DemoUse;
}

export const DEMOS: Record<DemoId, Demo> = {
  "copy-document": { component: lazy(() => import("./data/copy-document-demo").then((module) => ({ default: module.CopyDocumentDemo }))), usesSentence: false, usesTraining: false },
  "data-pipeline": { component: lazy(() => import("./data/data-pipeline-demo").then((module) => ({ default: module.DataPipelineDemo }))), usesSentence: false, usesTraining: false },
  "quality-check": { component: lazy(() => import("./data/quality-check-demo").then((module) => ({ default: module.QualityCheckDemo }))), usesSentence: false, usesTraining: false },
  dedupe: { component: lazy(() => import("./data/dedupe-demo").then((module) => ({ default: module.DedupeDemo }))), usesSentence: false, usesTraining: false },
  "gpt-2-tokenizer": { component: lazy(() => import("./tokenizer/gpt-2-tokenizer-demo").then((module) => ({ default: module.Gpt2TokenizerDemo }))), usesSentence: false, usesTraining: false },
  "bpe-learn": { component: lazy(() => import("./tokenizer/bpe-learn-demo").then((module) => ({ default: module.BpeLearnDemo }))), usesSentence: false, usesTraining: false },
  architecture: { component: lazy(() => import("./architecture/architecture-demo").then((module) => ({ default: module.ArchitectureDemo }))), usesSentence: true, usesTraining: true },
  "tiny-part": { component: lazy(() => import("./architecture/tiny-part-demo").then((module) => ({ default: module.TinyPartDemo }))), usesSentence: true, usesTraining: true },
  training: { component: lazy(() => import("./training/training-demo").then((module) => ({ default: module.TrainingDemo }))), usesSentence: false, usesTraining: false },
  "chat-format": { component: lazy(() => import("./post-training/chat-format-demo").then((module) => ({ default: module.ChatFormatDemo }))), usesSentence: "gpt-2-small", usesTraining: "gpt-2-small" },
  "answer-loss": { component: lazy(() => import("./post-training/answer-loss-demo").then((module) => ({ default: module.AnswerLossDemo }))), usesSentence: false, usesTraining: true },
  preference: { component: lazy(() => import("./post-training/preference-demo").then((module) => ({ default: module.PreferenceDemo }))), usesSentence: false, usesTraining: true },
  "tiny-test": { component: lazy(() => import("./evaluation/tiny-test-demo").then((module) => ({ default: module.TinyTestDemo }))), usesSentence: false, usesTraining: true },
  safety: { component: lazy(() => import("./evaluation/safety-demo").then((module) => ({ default: module.SafetyDemo }))), usesSentence: false, usesTraining: true },
  quantize: { component: lazy(() => import("./evaluation/quantize-demo").then((module) => ({ default: module.QuantizeDemo }))), usesSentence: false, usesTraining: true },
  "next-word": { component: lazy(() => import("./output/next-word-demo").then((module) => ({ default: module.NextWordDemo }))), usesSentence: true, usesTraining: true },
  generate: { component: lazy(() => import("./generate/generate-demo").then((module) => ({ default: module.GenerateDemo }))), usesSentence: true, usesTraining: true },
  "kv-cache": { component: lazy(() => import("./generate/kv-cache-demo").then((module) => ({ default: module.KvCacheDemo }))), usesSentence: true, usesTraining: true },
};

export const isUsedOn = (use: DemoUse, modelId: ModelId) => use === true || use === modelId;

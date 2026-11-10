import { type ComponentType, type LazyExoticComponent, lazy } from "react";
import type { DemoId, DemoProps } from "@/features/take-apart/interfaces/model-part.interface";

type Demo = LazyExoticComponent<ComponentType<DemoProps>>;

export const DEMOS: Record<DemoId, Demo> = {
  "copy-document": lazy(() => import("./data/copy-document-demo").then((module) => ({ default: module.CopyDocumentDemo }))),
  "data-pipeline": lazy(() => import("./data/data-pipeline-demo").then((module) => ({ default: module.DataPipelineDemo }))),
  "quality-check": lazy(() => import("./data/quality-check-demo").then((module) => ({ default: module.QualityCheckDemo }))),
  dedupe: lazy(() => import("./data/dedupe-demo").then((module) => ({ default: module.DedupeDemo }))),
  "gpt-2-tokenizer": lazy(() => import("./tokenizer/gpt-2-tokenizer-demo").then((module) => ({ default: module.Gpt2TokenizerDemo }))),
  "bpe-learn": lazy(() => import("./tokenizer/bpe-learn-demo").then((module) => ({ default: module.BpeLearnDemo }))),
  architecture: lazy(() => import("./architecture/architecture-demo").then((module) => ({ default: module.ArchitectureDemo }))),
  "tiny-part": lazy(() => import("./architecture/tiny-part-demo").then((module) => ({ default: module.TinyPartDemo }))),
  training: lazy(() => import("./training/training-demo").then((module) => ({ default: module.TrainingDemo }))),
  "chat-format": lazy(() => import("./post-training/chat-format-demo").then((module) => ({ default: module.ChatFormatDemo }))),
  "answer-loss": lazy(() => import("./post-training/answer-loss-demo").then((module) => ({ default: module.AnswerLossDemo }))),
  preference: lazy(() => import("./post-training/preference-demo").then((module) => ({ default: module.PreferenceDemo }))),
  "tiny-test": lazy(() => import("./evaluation/tiny-test-demo").then((module) => ({ default: module.TinyTestDemo }))),
  safety: lazy(() => import("./evaluation/safety-demo").then((module) => ({ default: module.SafetyDemo }))),
  quantize: lazy(() => import("./evaluation/quantize-demo").then((module) => ({ default: module.QuantizeDemo }))),
};

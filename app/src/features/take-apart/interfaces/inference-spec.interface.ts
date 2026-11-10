import type { SourceId } from "@/features/take-apart/trees/sources";
import type { ModelPart } from "./model-part.interface";
import type { Headline, RealNote, UseReasons } from "./real-facts.interface";

export interface StopWord {
  id: number;
  isComputed: boolean;
  notes: RealNote[];
}

export interface InferenceSpec {
  modelTitle: string;
  tokenizer: string;
  wordCount: number;
  width: number;
  context: number;
  kvWidth: number;
  sources: SourceId[];
  textSources: SourceId[];
  tokenizerNotes: RealNote[];
  merges: Headline;
  mergeNotes: RealNote[];
  stopWord: StopWord;
  reasons: UseReasons;
  embeddingLesson: string;
  embeddingText: string;
  embedding: ModelPart[];
  layers: ModelPart;
  finalNorm: ModelPart;
  outputHead: ModelPart;
}

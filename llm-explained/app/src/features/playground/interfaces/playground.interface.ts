import type { PlaygroundRun } from "@/features/playground/playground-run";
import type { LayerShape, LossPoint } from "@/shared/interfaces/model.interface";

export type RowKey = "isEvenAnswers" | "vectorSize" | "rounds" | "learningRate" | "optimizer" | "start" | "temperature" | "topK" | "topP";

export interface PlaygroundSettings {
  textGroups: string[];
  isEvenAnswers: boolean;
  vectorSize: number;
  layers: LayerShape[];
  rounds: number;
  learningRate: number;
  optimizer: "adam" | "plain";
  start: string;
  temperature: number;
  topK: number;
  topP: number;
}

export interface PlaygroundState {
  settings: PlaygroundSettings;
  addedRows: RowKey[];
  isAutoTrain: boolean;
  run: PlaygroundRun | null;
  isStale: boolean;
  earlier: LossPoint[];
  earlierLoss: number | null;
  writeSeed: number;
}

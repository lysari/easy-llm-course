import type { ModelPart } from "./model-part.interface";

export interface PartPlace {
  path: ModelPart[];
  copies: number;
  isInModel: boolean;
  isUpTo: boolean;
  total: number;
}

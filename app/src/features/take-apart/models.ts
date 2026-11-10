import type { ModelId, ModelPart, RealModel } from "@/features/take-apart/interfaces/model-part.interface";
import { GPT_2_SMALL } from "./trees/gpt-2-small";
import { TAKE_APART_PATH } from "./take-apart-path";
import { LLAMA_3_8B } from "./trees/llama-3-8b";

export const MODELS: RealModel[] = [GPT_2_SMALL, LLAMA_3_8B];

export const DEFAULT_MODEL: RealModel = GPT_2_SMALL;

export const findModel = (id: string | undefined) => MODELS.find((model) => model.id === id);

export function openPath(root: ModelPart, ids: string[]): ModelPart[] {
  const path = [root];
  for (const id of ids) {
    const next = path[path.length - 1].children?.find((child) => child.id === id);
    if (!next?.children) break;
    path.push(next);
  }
  return path;
}

export function pathTo(part: ModelPart, id: string): ModelPart[] | null {
  if (part.id === id) return [part];
  for (const child of part.children ?? []) {
    const found = pathTo(child, id);
    if (found) return [part, ...found];
  }
  return null;
}

export function matchingPath(root: ModelPart, path: ModelPart[]): ModelPart[] {
  for (const part of [...path].reverse()) {
    const found = pathTo(root, part.id);
    if (found && found[found.length - 1].children) return found;
  }
  return [root];
}

export function readPath(model: RealModel, ids: string[]): ModelPart[] {
  const [first, ...rest] = ids;
  if (first === "use" || first === "build") return openPath(model.modes[first], rest);
  return matchingPath(model.modes.use, openPath(model.root, ids));
}

export const pathUrl = (modelId: ModelId, path: ModelPart[]) => [TAKE_APART_PATH, modelId, ...path.map((part) => part.id)].join("/");

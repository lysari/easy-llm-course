import { traceForward } from "@/core/model/tiny-gpt";
import type { ModelPart, RealModel } from "@/features/inside-real-models/interfaces/model-part.interface";
import type { TinyGptWeights } from "@/shared/interfaces/model.interface";

export interface CacheEntry {
  keys: number[];
  values: number[];
}

export interface RealCache {
  layers: number;
  kvWidth: number;
  width: number;
  context: number;
}

export const cacheFor = (ids: number[], weights: TinyGptWeights): CacheEntry[][] => traceForward(ids, weights).layers.map((layer) => ids.map((_, word) => ({ keys: layer.keys[word], values: layer.values[word] })));

export function reuseGap(ids: number[], nextId: number, weights: TinyGptWeights) {
  const before = cacheFor(ids, weights);
  const after = cacheFor([...ids, nextId], weights);
  let gap = 0;
  before.forEach((layer, l) =>
    layer.forEach((entry, word) => {
      const again = after[l][word];
      entry.keys.forEach((key, i) => (gap = Math.max(gap, Math.abs(key - again.keys[i]), Math.abs(entry.values[i] - again.values[i]))));
    }),
  );
  return gap;
}

export function wordPasses(promptLength: number, newWords: number) {
  const without = newWords * promptLength + (newWords * (newWords - 1)) / 2;
  return { without, with: newWords === 0 ? 0 : promptLength + newWords - 1 };
}

export const cacheNumbers = (layers: number, kvWidth: number, words: number) => 2 * kvWidth * layers * words;

function findPart(part: ModelPart, id: string): ModelPart | undefined {
  if (part.id === id) return part;
  for (const child of part.children ?? []) {
    const found = findPart(child, id);
    if (found) return found;
  }
  return undefined;
}

function valueOf(part: ModelPart | undefined) {
  const headline = part?.facts.headline;
  if (!headline || !("value" in headline)) throw new Error(`${part?.id} has no number`);
  return headline.value;
}

export function realCacheOf(model: RealModel): RealCache {
  const use = model.modes.use;
  const layers = findPart(model.root, "layer")?.copies;
  if (!layers) throw new Error(`${model.id} has no layer`);
  return { layers, kvWidth: valueOf(findPart(use, "kv-cache")) / 2, width: valueOf(findPart(use, "embedding")), context: valueOf(findPart(use, "generate")) };
}

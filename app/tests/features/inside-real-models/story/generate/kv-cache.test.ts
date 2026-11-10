import { describe, expect, test } from "bun:test";
import { TEST_WORDS, VOCABULARY } from "@/core/model/corpus";
import { ModelLab } from "@/core/model/lab";
import { LAYER_COUNT, VECTOR_SIZE, traceForward } from "@/core/model/tiny-gpt";
import { MODELS } from "@/features/inside-real-models/models";
import { cacheFor, cacheNumbers, realCacheOf, reuseGap, wordPasses } from "@/features/inside-real-models/story/generate/kv-cache";

const lab = new ModelLab();
lab.trainRounds(5);
const ids = TEST_WORDS.map((word) => VOCABULARY.indexOf(word));

describe("the tiny model's keys and values", () => {
  test("one key and one value row per word, per layer, as the model works them out", () => {
    const cache = cacheFor(ids, lab.live);
    expect(cache).toHaveLength(LAYER_COUNT);
    expect(cache[1]).toHaveLength(ids.length);
    expect(cache[1][3].keys).toEqual(traceForward(ids, lab.live).layers[1].keys[3]);
    expect(cache[0][0].values).toHaveLength(VECTOR_SIZE);
  });

  test("adding a word leaves every earlier word's keys and values exactly the same", () => {
    expect(reuseGap(ids, VOCABULARY.indexOf("bone"), lab.live)).toBe(0);
  });

  test("without a cache every round runs all the words again", () => {
    expect(wordPasses(3, 4)).toEqual({ without: 3 + 4 + 5 + 6, with: 3 + 3 });
    expect(wordPasses(5, 1)).toEqual({ without: 5, with: 5 });
    expect(wordPasses(5, 0)).toEqual({ without: 0, with: 0 });
  });
});

describe("the real models' cache sizes, worked out from their settings", () => {
  const [gpt2, llama] = MODELS.map(realCacheOf);

  test("GPT-2 small: 12 layers of 768 keys and 768 values per word", () => {
    expect(gpt2).toEqual({ layers: 12, kvWidth: 768, width: 768, context: 1024 });
    expect(cacheNumbers(gpt2.layers, gpt2.kvWidth, 1)).toBe(18_432);
    expect(cacheNumbers(gpt2.layers, gpt2.kvWidth, gpt2.context)).toBe(18_874_368);
  });

  test("Llama 3 8B: 32 layers of 8 groups × 128, four times smaller than one per head", () => {
    expect(llama).toEqual({ layers: 32, kvWidth: 1024, width: 4096, context: 8192 });
    expect(cacheNumbers(llama.layers, llama.kvWidth, 1)).toBe(65_536);
    expect(cacheNumbers(llama.layers, llama.kvWidth, llama.context)).toBe(536_870_912);
    expect(cacheNumbers(llama.layers, llama.width, 1) / cacheNumbers(llama.layers, llama.kvWidth, 1)).toBe(4);
  });
});

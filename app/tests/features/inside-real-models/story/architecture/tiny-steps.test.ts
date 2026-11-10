import { describe, expect, test } from "bun:test";
import { ModelLab } from "@/core/model/lab";
import type { ModelPart } from "@/features/inside-real-models/interfaces/model-part.interface";
import { MODELS } from "@/features/inside-real-models/models";
import { readSentence } from "@/features/inside-real-models/sentence";
import { gelu, layerNormParts, rmsNormParts, rope, silu } from "@/features/inside-real-models/story/architecture/exact-formulas";
import { tinyStep } from "@/features/inside-real-models/story/architecture/tiny-steps";
import { viewTiny } from "@/features/inside-real-models/tiny-parts";

const allParts = (part: ModelPart): ModelPart[] => [part, ...(part.children ?? []).flatMap(allParts)];
const lab = new ModelLab();
lab.trainRounds(5);
const tiny = viewTiny(lab, readSentence("my dog is hungry").words);

describe("the exact formulas", () => {
  test("GELU matches GPT-2's tanh formula at known points", () => {
    expect(gelu(0)).toBe(0);
    expect(gelu(1)).toBeCloseTo(0.841192, 5);
    expect(gelu(-1)).toBeCloseTo(-0.158808, 5);
  });

  test("SiLU is x times sigmoid(x)", () => {
    expect(silu(0)).toBe(0);
    expect(silu(2)).toBeCloseTo(2 / (1 + Math.exp(-2)), 10);
  });

  test("LayerNorm gives average 0 and spread 1; RMSNorm gives root mean square 1", () => {
    const values = [1, 2, 3, 6];
    const { out } = layerNormParts(values);
    expect(out.reduce((sum, value) => sum + value, 0)).toBeCloseTo(0, 10);
    expect(Math.sqrt(out.reduce((sum, value) => sum + value * value, 0) / out.length)).toBeCloseTo(1, 4);
    const rms = rmsNormParts(values).out;
    expect(Math.sqrt(rms.reduce((sum, value) => sum + value * value, 0) / rms.length)).toBeCloseTo(1, 4);
  });

  test("RoPE keeps each pair's length and does nothing at place 0", () => {
    const values = [1, 2, 3, 4];
    expect(rope(values, 0)).toEqual(values);
    const turned = rope(values, 5);
    expect(Math.hypot(turned[0], turned[1])).toBeCloseTo(Math.hypot(1, 2), 10);
    expect(Math.hypot(turned[2], turned[3])).toBeCloseTo(Math.hypot(3, 4), 10);
  });
});

describe.each(MODELS)("$title model parts in the tiny-part demo", (model) => {
  const parts = [...new Set([model.modes.use, model.modes.build].flatMap(allParts))].filter((part) => part.story?.demo === "tiny-part");

  test("every one shows finite numbers before and after, for every word", () => {
    expect(parts.length).toBeGreaterThan(30);
    for (const part of parts) {
      for (let word = 0; word < tiny.words.length; word++) {
        const shown = tinyStep(part, model.id, tiny, { word, out: 3, extra: 0.5 });
        const numbers = [...shown.before, ...shown.after].flatMap((list) => list.values);
        expect(numbers.length, part.id).toBeGreaterThan(0);
        expect(numbers.every(Number.isFinite), part.id).toBe(true);
        expect(shown.note.length, part.id).toBeGreaterThan(10);
      }
    }
  });

  test("a projection's output number is exactly its multiply-and-adds added up", () => {
    const query = parts.find((part) => part.id === "query-number");
    if (!query) throw new Error("query-number");
    const shown = tinyStep(query, model.id, tiny, { word: 3, out: 2, extra: 0 });
    expect(shown.terms?.total).toBeCloseTo(tiny.trace.layers[0].queries[3][2], 10);
    expect(shown.terms?.pairs.length).toBe(tiny.trace.layers[0].normalized[3].length);
  });

  test("the embedding shows what goes into layer 1: word row plus place row in GPT-2, the word row alone in Llama 3", () => {
    const embedding = parts.find((part) => part.id === "embedding");
    if (!embedding) throw new Error("embedding");
    const shown = tinyStep(embedding, model.id, tiny, { word: 2, out: 0, extra: 0 });
    const into = model.id === "gpt-2-small" ? tiny.trace.combined[2] : tiny.trace.tokenVectors[2];
    expect(shown.after[0].values).toEqual(into);
  });

  test("a score is the query times the key, scaled, as the tiny model computed it", () => {
    const score = parts.find((part) => part.id === "score");
    if (!score) throw new Error("score");
    const shown = tinyStep(score, model.id, tiny, { word: 3, out: 1, extra: 0 });
    expect(shown.terms?.total).toBeCloseTo(tiny.trace.layers[0].scores[3][1], 10);
  });
});

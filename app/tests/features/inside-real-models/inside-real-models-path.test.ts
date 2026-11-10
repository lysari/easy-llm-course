import { describe, expect, test } from "bun:test";
import { movedPath } from "@/features/inside-real-models/inside-real-models-path";

describe("old take-apart links", () => {
  test("move to the same model, mode and parts under the new path", () => {
    expect(movedPath("/take-apart/gpt-2-small/build/architecture/model/word-table")).toBe("/inside-real-models/gpt-2-small/build/architecture/model/word-table");
    expect(movedPath("/take-apart/llama-3-8b/use/layers/layer/attention")).toBe("/inside-real-models/llama-3-8b/use/layers/layer/attention");
    expect(movedPath("/take-apart/gpt-2-small/use")).toBe("/inside-real-models/gpt-2-small/use");
  });

  test("move the bare old path to the bare new path", () => {
    expect(movedPath("/take-apart")).toBe("/inside-real-models");
    expect(movedPath("/take-apart/")).toBe("/inside-real-models");
  });

  test("match the old path in any letter case, as the router does", () => {
    expect(movedPath("/Take-Apart/gpt-2-small/build")).toBe("/inside-real-models/gpt-2-small/build");
  });
});

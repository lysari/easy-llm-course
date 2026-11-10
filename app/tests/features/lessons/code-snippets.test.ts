import { describe, expect, test } from "bun:test";
import { format } from "prettier";
import { STEPS } from "@/features/lessons/flow";
import type { ModelPart } from "@/features/inside-real-models/interfaces/model-part.interface";
import { MODELS } from "@/features/inside-real-models/models";

const SNIPPET_FORMAT = { parser: "typescript", printWidth: 56 } as const;

const allParts = (part: ModelPart): ModelPart[] => [part, ...(part.children ?? []).flatMap(allParts)];

const lessonSnippets = STEPS.flatMap((step) => (step.code ? [[step.id, step.code] as const] : []));
const insideRealModelsSnippets = MODELS.flatMap((model) =>
  Object.values(model.modes).flatMap((root) => allParts(root).flatMap((part) => (part.code ? [[`${model.id} ${root.id} ${part.id}`, part.code] as const] : []))),
);
const snippets = [...lessonSnippets, ...insideRealModelsSnippets];

describe("code snippets", () => {
  test("every page with code is checked", () => {
    expect(lessonSnippets.length).toBeGreaterThan(0);
    expect(insideRealModelsSnippets.length).toBeGreaterThan(0);
  });

  test.each(snippets)("%s is Prettier-formatted", async (_, code) => {
    expect(await format(code, SNIPPET_FORMAT)).toBe(`${code}\n`);
  });
});

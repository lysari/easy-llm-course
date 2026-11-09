import { describe, expect, test } from "bun:test";
import { format } from "prettier";
import { STEPS } from "@/features/lessons/flow";

const SNIPPET_FORMAT = { parser: "typescript", printWidth: 56 } as const;

const snippets = STEPS.flatMap((step) => (step.code ? [[step.id, step.code] as const] : []));

describe("lesson code snippets", () => {
  test("every part with code is checked", () => {
    expect(snippets.length).toBeGreaterThan(0);
  });

  test.each(snippets)("%s is Prettier-formatted", async (_, code) => {
    expect(await format(code, SNIPPET_FORMAT)).toBe(`${code}\n`);
  });
});

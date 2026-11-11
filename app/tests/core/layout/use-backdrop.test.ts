import { describe, expect, test } from "bun:test";
import { parseBackdrop } from "@/core/layout/use-backdrop";

describe("parseBackdrop", () => {
  test("loads every saved background, including the old wash and photo values", () => {
    expect(parseBackdrop("wash")).toBe("wash");
    expect(parseBackdrop("meadow")).toBe("meadow");
    expect(parseBackdrop("sky")).toBe("sky");
    expect(parseBackdrop("citrus")).toBe("citrus");
    expect(parseBackdrop("photo")).toBe("photo");
  });

  test("falls back to the default wash when nothing or something unknown is saved", () => {
    expect(parseBackdrop(null)).toBe("wash");
    expect(parseBackdrop("")).toBe("wash");
    expect(parseBackdrop("Default")).toBe("wash");
    expect(parseBackdrop("ocean")).toBe("wash");
  });
});

import { describe, expect, test } from "bun:test";
import { parseLook } from "@/core/layout/use-look";

describe("parseLook", () => {
  test("loads every saved look", () => {
    expect(parseLook("auto")).toBe("auto");
    expect(parseLook("light")).toBe("light");
    expect(parseLook("dark")).toBe("dark");
  });

  test("falls back to auto when nothing, a removed look or something unknown is saved", () => {
    expect(parseLook(null)).toBe("auto");
    expect(parseLook("")).toBe("auto");
    expect(parseLook("fresh")).toBe("auto");
    expect(parseLook("Light")).toBe("auto");
    expect(parseLook("sepia")).toBe("auto");
  });
});

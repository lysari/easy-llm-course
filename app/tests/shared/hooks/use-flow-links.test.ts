import { describe, expect, test } from "bun:test";
import { measureLinks } from "@/shared/hooks/use-flow-links";

type Elements = Parameters<typeof measureLinks>[0];

const cardAt = (left: number, top: number, width: number, height: number) => ({ getBoundingClientRect: () => ({ left, top, width, height, right: left + width }) }) as HTMLElement;

function elements(): Elements {
  return {
    flowRef: { current: cardAt(0, 0, 1000, 600) },
    childColumnRef: { current: cardAt(500, 0, 500, 600) },
    parentCards: { current: new Map([["stage", cardAt(0, 0, 400, 100)]]) },
    childCards: {
      current: new Map([
        ["part", cardAt(500, 0, 200, 100)],
        ["sub-part", cardAt(800, 0, 150, 100)],
        ["other", cardAt(500, 200, 200, 100)],
      ]),
    },
  };
}

describe("measureLinks", () => {
  test("each line keeps the cards it runs from and to, and each dot the card it starts from", () => {
    const drawing = measureLinks(elements(), { parentId: "stage", childIds: ["part", "sub-part", "other"], nestedIds: ["sub-part"] });
    expect(drawing?.paths.map((path) => [path.fromId, path.toId])).toEqual([
      ["stage", "part"],
      ["part", "sub-part"],
      ["stage", "other"],
    ]);
    expect(drawing?.dots.map((dot) => [dot.id, dot.fromId])).toEqual([
      ["start", "stage"],
      ["sub-part", "part"],
    ]);
  });

  test("each line's colour runs from the card it starts at to its child", () => {
    const drawing = measureLinks(elements(), { parentId: "stage", childIds: ["part", "sub-part", "other"], nestedIds: ["sub-part"] });
    expect(drawing?.paths.map((path) => [path.startX, path.endX])).toEqual([
      [400, 500],
      [700, 800],
      [400, 500],
    ]);
  });

  test("draws nothing when a card is missing", () => {
    expect(measureLinks(elements(), { parentId: "stage", childIds: ["part", "gone"] })).toBeNull();
  });
});

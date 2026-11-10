import { afterAll, describe, expect, test } from "bun:test";
import { Window } from "happy-dom";
import { renderToStaticMarkup } from "react-dom/server";
import { FlowLinks } from "@/shared/components/flow-links";
import type { LinkDrawing } from "@/shared/interfaces/link.interface";

const DRAWING: LinkDrawing = {
  paths: [
    { fromId: "stage", toId: "part", d: "M 0 0 H 10", startX: 0, endX: 10 },
    { fromId: "part", toId: "sub-part", d: "M 20 0 H 30", startX: 20, endX: 30 },
    { fromId: "stage", toId: "other", d: "M 0 0 V 40 H 10", startX: 0, endX: 10 },
  ],
  dots: [
    { id: "start", fromId: "stage", x: 0, y: 0 },
    { id: "sub-part", fromId: "part", x: 20, y: 0 },
  ],
};

const window = new Window({ settings: { disableJavaScriptEvaluation: true, disableJavaScriptFileLoading: true, disableCSSFileLoading: true, disableIframePageLoading: true } });

afterAll(() => window.happyDOM.close());

function render(pathIds: string[], glowIds: string[]) {
  return new window.DOMParser().parseFromString(renderToStaticMarkup(<FlowLinks id="links" drawing={DRAWING} pathIds={pathIds} glowIds={glowIds} />), "text/html");
}

const hasGradient = (element: { getAttribute: (name: string) => string | null }) => (element.getAttribute("style") ?? "").includes("stroke:url(#links-gradient-");

const isActiveDot = (dot: { classList: { contains: (name: string) => boolean } }) => dot.classList.contains("is-active");

const coloredOf = (svg: ReturnType<typeof render>) => [...svg.querySelectorAll(".link-path")].filter(hasGradient).map((path) => path.getAttribute("d"));

const glowsOf = (svg: ReturnType<typeof render>) => [...svg.querySelectorAll(".link-glow")].map((glow) => glow.getAttribute("d"));

describe("FlowLinks", () => {
  test("every line draws in quiet, one after another, under any coloured line", () => {
    const svg = render(["stage", "part"], ["part"]);
    const quiet = [...svg.querySelectorAll(".link-path")].filter((path) => !hasGradient(path));
    expect(quiet.map((path) => [path.getAttribute("d"), path.getAttribute("style")])).toEqual([
      ["M 0 0 H 10", "animation-delay:0ms"],
      ["M 20 0 H 30", "animation-delay:70ms"],
      ["M 0 0 V 40 H 10", "animation-delay:140ms"],
    ]);
    expect([...svg.querySelectorAll(".links > path")].map(hasGradient)).toEqual([false, false, false, true, true]);
  });

  test("a card alone on the path colours none of its lines out, and nothing glows", () => {
    const svg = render(["stage"], []);
    expect(coloredOf(svg)).toEqual([]);
    expect(glowsOf(svg)).toEqual([]);
    expect([...svg.querySelectorAll(".link-dot")].map(isActiveDot)).toEqual([false, false]);
  });

  test("a line with both ends on the path is coloured, only the line into a glow card moves, and lines off the path stay quiet", () => {
    const svg = render(["stage", "part"], ["part"]);
    expect(coloredOf(svg)).toEqual(["M 0 0 H 10"]);
    expect(glowsOf(svg)).toEqual(["M 0 0 H 10"]);
    expect([...svg.querySelectorAll(".link-dot")].map(isActiveDot)).toEqual([true, false]);
  });

  test("the whole path to a nested card is coloured, and only its last line glows", () => {
    const svg = render(["stage", "part", "sub-part"], ["sub-part"]);
    expect(coloredOf(svg)).toEqual(["M 0 0 H 10", "M 20 0 H 30"]);
    expect(glowsOf(svg)).toEqual(["M 20 0 H 30"]);
    expect([...svg.querySelectorAll(".link-dot")].map(isActiveDot)).toEqual([true, true]);
  });

  test("a glow card off the path does not glow", () => {
    const svg = render(["stage"], ["other"]);
    expect(coloredOf(svg)).toEqual([]);
    expect(glowsOf(svg)).toEqual([]);
  });

  test("with no path every line is quiet and nothing glows", () => {
    const svg = render([], []);
    expect(coloredOf(svg)).toEqual([]);
    expect(glowsOf(svg)).toEqual([]);
    expect([...svg.querySelectorAll(".link-dot")].some(isActiveDot)).toBe(false);
  });

  test("each gradient line gets its own gradient from its start to its end, named after the drawing", () => {
    const gradients = [...render(["stage", "part", "sub-part"], ["sub-part"]).querySelectorAll("linearGradient")];
    expect(gradients.map((gradient) => [gradient.getAttribute("id"), gradient.getAttribute("x1"), gradient.getAttribute("x2")])).toEqual([
      ["links-gradient-part", "0", "10"],
      ["links-gradient-sub-part", "20", "30"],
    ]);
    expect(gradients.every((gradient) => gradient.getAttribute("gradientUnits") === "userSpaceOnUse")).toBe(true);
  });
});

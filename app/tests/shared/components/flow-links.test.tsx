import { afterAll, describe, expect, test } from "bun:test";
import { Window } from "happy-dom";
import { renderToStaticMarkup } from "react-dom/server";
import { FlowLinks } from "@/shared/components/flow-links";
import type { LinkDrawing } from "@/shared/interfaces/link.interface";

const DRAWING: LinkDrawing = {
  paths: [
    { id: "part", fromId: "stage", d: "M 0 0 H 10", startX: 0, endX: 10 },
    { id: "sub-part", fromId: "part", d: "M 20 0 H 30", startX: 20, endX: 30 },
  ],
  dots: [
    { id: "start", fromId: "stage", x: 0, y: 0 },
    { id: "sub-part", fromId: "part", x: 20, y: 0 },
  ],
};

const window = new Window({ settings: { disableJavaScriptEvaluation: true, disableJavaScriptFileLoading: true, disableCSSFileLoading: true, disableIframePageLoading: true } });

afterAll(() => window.happyDOM.close());

function render(activeIds: string[]) {
  return new window.DOMParser().parseFromString(renderToStaticMarkup(<FlowLinks id="links" drawing={DRAWING} activeIds={activeIds} />), "text/html");
}

const hasGradient = (element: { getAttribute: (name: string) => string | null }) => (element.getAttribute("style") ?? "").includes("stroke:url(#links-gradient-");

const isActiveDot = (dot: { classList: { contains: (name: string) => boolean } }) => dot.classList.contains("is-active");

describe("FlowLinks", () => {
  test("lines from an active card get the gradient and a glow, others stay quiet", () => {
    const svg = render(["stage"]);
    expect([...svg.querySelectorAll(".link-path")].map(hasGradient)).toEqual([true, false]);
    expect([...svg.querySelectorAll(".link-glow")].map((glow) => glow.getAttribute("d"))).toEqual(["M 0 0 H 10"]);
    expect([...svg.querySelectorAll(".link-dot")].map(isActiveDot)).toEqual([true, false]);
  });

  test("a nested line follows its own parent card", () => {
    const svg = render(["part"]);
    expect([...svg.querySelectorAll(".link-path")].map(hasGradient)).toEqual([false, true]);
    expect([...svg.querySelectorAll(".link-dot")].map(isActiveDot)).toEqual([false, true]);
  });

  test("with no active card every line is quiet and nothing glows", () => {
    const svg = render([]);
    expect([...svg.querySelectorAll(".link-path")].some(hasGradient)).toBe(false);
    expect(svg.querySelectorAll(".link-glow")).toHaveLength(0);
  });

  test("each active line gets its own gradient from its start to its end, named after the drawing", () => {
    const gradients = [...render(["stage", "part"]).querySelectorAll("linearGradient")];
    expect(gradients.map((gradient) => [gradient.getAttribute("id"), gradient.getAttribute("x1"), gradient.getAttribute("x2")])).toEqual([
      ["links-gradient-part", "0", "10"],
      ["links-gradient-sub-part", "20", "30"],
    ]);
    expect(gradients.every((gradient) => gradient.getAttribute("gradientUnits") === "userSpaceOnUse")).toBe(true);
  });
});

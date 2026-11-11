import { afterAll, describe, expect, test } from "bun:test";
import { Window } from "happy-dom";
import { prerender } from "react-dom/static";
import type { ReadabilityTrace } from "@/features/inside-real-models/interfaces/readability-trace.interface";
import { extractPage } from "@/features/inside-real-models/story/data/extract-page";
import { HowItWorks, HowItWorksStep } from "@/features/inside-real-models/story/data/how-it-works";
import { showPoints, walkStepsOf } from "@/features/inside-real-models/story/data/how-it-works-steps";
import { SAVED_PAGE } from "@/features/inside-real-models/story/data/saved-page";

const window = new Window({ settings: { disableJavaScriptEvaluation: true, disableJavaScriptFileLoading: true, disableCSSFileLoading: true, disableIframePageLoading: true } });

afterAll(() => window.happyDOM.close());

const saved = extractPage(new window.DOMParser().parseFromString(SAVED_PAGE.html, "text/html") as unknown as Document, SAVED_PAGE.html).trace;
if (!saved) throw new Error("The saved page has no walkthrough");
const trace: ReadabilityTrace = saved;
const steps = walkStepsOf(trace);
const noop = () => undefined;

async function htmlOf(element: React.ReactElement) {
  const { prelude } = await prerender(element);
  return new Response(prelude).text();
}

const stepHtml = (step: number) => htmlOf(<HowItWorksStep trace={trace} step={step} onStep={noop} onShowKept={noop} />);

const buttonOf = (html: string, label: string) => html.match(new RegExp(`<button[^>]*>${label}</button>`))?.[0] ?? "";

const liveTextOf = (html: string) => html.match(/<p class="visually-hidden" aria-live="polite">([^<]*)<\/p>/)?.[1] ?? "";

const unescape = (text: string) => text.replaceAll("&#x27;", "'").replaceAll("&quot;", '"').replaceAll("&amp;", "&");

describe("the five steps", () => {
  test("there are five, each with its own title and sentence", () => {
    expect(steps).toHaveLength(5);
    expect(new Set(steps.map((step) => step.title)).size).toBe(5);
    expect(new Set(steps.map((step) => step.sentence)).size).toBe(5);
  });

  test("their numbers come from the run", () => {
    const winner = trace.boxes[trace.winnerId];
    expect(steps[0].sentence).toContain(`${trace.boxes.length} it acts on`);
    expect(steps[1].sentence).toContain(`${trace.emptyCount} empty boxes`);
    expect(steps[3].sentence).toContain(`div.two-third-first wins with ${showPoints(winner.score?.score ?? 0)}`);
    expect(steps[3].sentence).toContain(`at least ${showPoints(trace.threshold)}`);
    expect(steps[4].sentence).toContain("Only div.two-third-first is kept");
  });
});

describe("stepping", () => {
  test("Back is off on the first step and Next on the last, both on in between", async () => {
    for (let step = 0; step < steps.length; step += 1) {
      const html = await stepHtml(step);
      expect(buttonOf(html, "‹ Back"), `step ${step + 1}`).toContain(`aria-disabled="${step === 0}"`);
      expect(buttonOf(html, "Next ›"), `step ${step + 1}`).toContain(`aria-disabled="${step === steps.length - 1}"`);
    }
  });

  test("each step shows its place, title and sentence, and only the last one links to the kept text", async () => {
    for (let step = 0; step < steps.length; step += 1) {
      const html = unescape(await stepHtml(step));
      expect(html).toContain(`Step ${step + 1} of 5`);
      expect(html).toContain(`${step + 1}. ${steps[step].title}`);
      expect(html).toContain(steps[step].sentence);
      expect(html.includes("Show the kept text")).toBe(step === steps.length - 1);
    }
  });

  test("the live region reads out the step it moves to", async () => {
    const said = await Promise.all(steps.map(async (_, step) => unescape(liveTextOf(await stepHtml(step)))));
    said.forEach((text, step) => expect(text).toBe(`Step ${step + 1} of 5: ${steps[step].title}. ${steps[step].sentence}`));
    expect(new Set(said).size).toBe(steps.length);
  });

  test("the walkthrough opens on step 1", async () => {
    const html = unescape(await htmlOf(<HowItWorks trace={trace} onShowKept={noop} />));
    expect(html).toContain("Step 1 of 5");
    expect(html).toContain(steps[0].title);
  });
});

import { describe, expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import { ModelLab } from "@/core/model/lab";
import type { ModelPart } from "@/features/inside-real-models/interfaces/model-part.interface";
import { MODELS } from "@/features/inside-real-models/models";
import { PartDrawer } from "@/features/inside-real-models/part-drawer";
import { placeOf } from "@/features/inside-real-models/real-place";
import { readSentence } from "@/features/inside-real-models/sentence";
import { viewTiny } from "@/features/inside-real-models/tiny-parts";
import { SOURCES } from "@/features/inside-real-models/trees/sources";

const INFO_TITLES = ["What it does", "Why it exists", "When you&#x27;d do this", "What goes in and out"];

const allParts = (part: ModelPart): ModelPart[] => [part, ...(part.children ?? []).flatMap(allParts)];

const missingStory = (step: ModelPart) => [...new Set(allParts(step).filter((part) => !part.story).map((part) => part.id))];

describe.each(MODELS)("$title why, watch and when", (model) => {
  const tiny = viewTiny(new ModelLab(), readSentence("").words);
  const parts = [model.modes.use, model.modes.build].flatMap(allParts);
  const told = parts.filter((part) => part.story);

  describe.each(["use", "build"] as const)("%s mode", (mode) => {
    for (const step of model.modes[mode].children ?? []) {
      test(`${step.title}: every part says why, shows it happen and says when`, () => expect(missingStory(step)).toEqual([]));
    }
  });

  test("a part with a story says what goes wrong without it, when to do it and what comes next", () => {
    expect(told.length).toBeGreaterThan(0);
    for (const { id, story } of told) {
      if (!story) throw new Error(id);
      for (const text of [story.problem, story.when, story.feeds]) expect(text, id).toMatch(/^[A-Z].{30,}\.$/);
      if (story.nextId) expect(parts.map((part) => part.id), id).toContain(story.nextId);
    }
  });

  test("its reasons are sourced claims about the real model, or say Not published", () => {
    for (const part of told) {
      const reasons = part.facts.reasons ?? [];
      expect(reasons.length, part.id).toBeGreaterThan(0);
      expect(new Set(reasons.map((reason) => reason.value)).size, part.id).toBe(reasons.length);
      for (const reason of reasons) {
        expect(SOURCES, `${part.id}: ${reason.source}`).toHaveProperty(reason.source);
        expect(reason.value.length, part.id).toBeGreaterThan(2);
      }
    }
  });

  test.each(["use", "build"] as const)("in %s mode the wide drawer puts why and when after What it does, and the demo in its own column", (mode) => {
    const root = model.modes[mode];
    for (const part of allParts(root).filter((each) => each.story)) {
      const nextPart = parts.find((each) => each.id === part.story?.nextId);
      const html = renderToStaticMarkup(<PartDrawer ref={null} root={root} modelId={model.id} modelTitle={model.title} part={part} place={placeOf(model, root, part)} tiny={tiny} nextPart={nextPart} onOpen={() => {}} onClose={() => {}} />);
      const places = INFO_TITLES.map((title) => html.indexOf(`<p class="section-title">${title}</p>`));
      const panel = html.indexOf('<div class="drawer-panel">');
      expect(html, part.id).toContain('class="drawer drawer--wide"');
      expect(places.every((place) => place >= 0 && place < panel), part.id).toBe(true);
      expect([...places].sort((a, b) => a - b), part.id).toEqual(places);
      expect(html.indexOf("Watch it happen", panel), part.id).toBeGreaterThan(panel);
      if (nextPart) expect(html, part.id).toContain(`Next step: ${nextPart.title.replaceAll("'", "&#x27;")}`);
    }
  });

  test("a part with no story keeps the narrow drawer", () => {
    const part: ModelPart = { ...(model.modes.use.children?.[0] ?? model.modes.use), story: undefined };
    const html = renderToStaticMarkup(<PartDrawer ref={null} root={model.modes.use} modelId={model.id} modelTitle={model.title} part={part} place={placeOf(model, model.modes.use, part)} tiny={tiny} nextPart={undefined} onOpen={() => {}} onClose={() => {}} />);
    expect(html).toContain('class="drawer"');
    expect(html).not.toContain("drawer-panel");
  });
});

describe("Copy one document in", () => {
  const [gpt2, llama] = MODELS;
  const gatherOf = (root: ModelPart) => allParts(root).find((part) => part.id === "gather");

  test("both models show the real page demo and lead to the quality check", () => {
    for (const model of MODELS) {
      expect(gatherOf(model.modes.build)?.story).toMatchObject({ demo: "copy-document", nextId: "filter" });
    }
  });

  test("GPT-2's reasons quote its paper, and Llama 3's say how text was copied is Not published", () => {
    const gpt2Reasons = gatherOf(gpt2.modes.build)?.facts.reasons ?? [];
    expect(gpt2Reasons.map((reason) => reason.source)).toEqual(["gpt-2-paper", "gpt-2-paper", "gpt-2-paper"]);
    expect(gpt2Reasons.map((reason) => reason.value).join(" ")).toContain("mostly unintelligible");
    const llamaReasons = gatherOf(llama.modes.build)?.facts.reasons ?? [];
    expect(llamaReasons.find((reason) => reason.label === "How text was copied out of pages")?.value).toStartWith("Not published");
  });
});

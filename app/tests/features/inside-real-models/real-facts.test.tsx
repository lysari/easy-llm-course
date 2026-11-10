import { describe, expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import { ModelLab } from "@/core/model/lab";
import type { ModelPart } from "@/features/inside-real-models/interfaces/model-part.interface";
import { MODELS } from "@/features/inside-real-models/models";
import { PartCard } from "@/features/inside-real-models/part-card";
import { PartDrawer } from "@/features/inside-real-models/part-drawer";
import { placeOf } from "@/features/inside-real-models/real-place";
import { readSentence } from "@/features/inside-real-models/sentence";
import { viewTiny } from "@/features/inside-real-models/tiny-parts";
import { SOURCES } from "@/features/inside-real-models/trees/sources";

const PLACEHOLDERS = /at a time|not covered|not given|sample/i;
const UNSTATED = /^(Not published|None: .+)$/;

const allParts = (part: ModelPart): ModelPart[] => [part, ...(part.children ?? []).flatMap(allParts)];

const partIn = (part: ModelPart, id: string) => allParts(part).find((each) => each.id === id);

function factTexts(part: ModelPart) {
  const { facts } = part;
  const headline = "text" in facts.headline ? [facts.headline.text] : [facts.headline.unit];
  return [facts.shape, facts.input, facts.output, facts.inputDetail ?? "", facts.outputDetail ?? "", ...headline, ...[...(facts.notes ?? []), ...(facts.reasons ?? [])].flatMap((note) => [note.label, note.value])];
}

function gpt2Count(vocab: number, context: number, width: number, layers: number) {
  const norm = 2 * width;
  const layer = norm + (width * 3 * width + 3 * width) + (width * width + width) + norm + (width * 4 * width + 4 * width) + (4 * width * width + width);
  return vocab * width + context * width + layers * layer + norm;
}

function llama3Count(vocab: number, width: number, layers: number, heads: number, kvHeads: number, multiplier: number, multipleOf: number) {
  const headSize = width / heads;
  const hidden = multipleOf * Math.ceil(Math.floor(multiplier * Math.floor((2 * 4 * width) / 3)) / multipleOf);
  const attention = 2 * width * width + 2 * width * kvHeads * headSize;
  return 2 * vocab * width + layers * (attention + 3 * width * hidden + 2 * width) + width;
}

describe("sources", () => {
  test("every source has a title and a web address", () => {
    for (const [id, source] of Object.entries(SOURCES)) {
      expect(source.title.length, id).toBeGreaterThan(10);
      expect(source.url, id).toMatch(/^https:\/\/\S+$/);
    }
  });
});

describe.each(MODELS)("$title real facts", (model) => {
  const trees = [model.root, model.modes.use, model.modes.build];
  const parts = trees.flatMap(allParts);

  test("every part and step names at least one source that exists", () => {
    for (const part of parts) {
      expect(part.facts.sources.length, part.id).toBeGreaterThan(0);
      for (const id of part.facts.sources) expect(SOURCES, `${part.id}: ${id}`).toHaveProperty(id);
      for (const note of [...(part.facts.notes ?? []), ...(part.facts.reasons ?? [])]) {
        expect(SOURCES, `${part.id}: ${note.source}`).toHaveProperty(note.source);
        expect(note.value.length, part.id).toBeGreaterThan(2);
      }
    }
  });

  test("no real fact uses a placeholder", () => {
    for (const part of parts) {
      for (const text of factTexts(part)) expect(text, part.id).not.toMatch(PLACEHOLDERS);
    }
  });

  test("a fact the makers never gave says so, with a note that explains it", () => {
    for (const part of parts) {
      const { headline, notes = [] } = part.facts;
      if (!("text" in headline)) continue;
      expect(headline.text, part.id).toMatch(UNSTATED);
      expect(notes.length, part.id).toBeGreaterThan(0);
    }
  });

  test("a part with no stored numbers says None to learn, never 0", () => {
    const tiny = viewTiny(new ModelLab(), readSentence("").words);
    const root = model.modes.build;
    for (const id of ["gather", "tokenizer", "pre-training"]) {
      const part = partIn(root, id);
      if (!part) throw new Error(id);
      const html = renderToStaticMarkup(<PartDrawer ref={null} root={root} modelId={model.id} modelTitle={model.title} part={part} place={placeOf(model, root, part)} tiny={tiny} nextPart={undefined} onOpen={() => {}} onClose={() => {}} />);
      expect(html, id).toContain("<span>Numbers</span><span>None to learn</span>");
      expect(html, id).not.toContain("<span>Numbers</span><span>0</span>");
    }
  });
});

describe("published numbers", () => {
  const [gpt2, llama] = MODELS;
  const headlineOf = (root: ModelPart, id: string) => partIn(root, id)?.facts.headline;

  test("GPT-2 small's count is worked out from config.json", () => {
    expect(gpt2.root.facts.weightCount).toBe(gpt2Count(50_257, 1024, 768, 12));
  });

  test("Llama 3 8B's count is worked out from Meta's settings and matches Hugging Face", () => {
    expect(llama.root.facts.weightCount).toBe(llama3Count(128_256, 4096, 32, 32, 8, 1.3, 1024));
    expect(llama.root.facts.weightCount).toBe(8_030_261_248);
  });

  test("GPT-2's WebText, word list and stop word match the paper, vocab.bpe and config", () => {
    expect(headlineOf(gpt2.modes.build, "data")).toEqual({ value: 8_000_000, unit: "documents", isOver: true });
    expect(headlineOf(gpt2.modes.build, "gather")).toEqual({ value: 45_000_000, unit: "links" });
    expect(headlineOf(gpt2.modes.build, "filter")).toEqual({ value: 3, unit: "karma or more" });
    expect(headlineOf(gpt2.modes.build, "learn-merge")).toMatchObject({ value: 50_000 });
    expect(headlineOf(gpt2.modes.use, "stop-check")).toMatchObject({ value: 50_256 });
  });

  test("GPT-2 had no post-training", () => {
    for (const part of allParts(partIn(gpt2.modes.build, "post-training") as ModelPart)) {
      expect(part.facts.headline, part.id).toEqual({ text: "None: base model only" });
    }
  });

  test("Llama 3 8B's data, post-training and stop word match the card, blog and tokenizer code", () => {
    expect(headlineOf(llama.modes.build, "data")).toEqual({ value: 15_000_000_000_000, unit: "words", isOver: true });
    expect(headlineOf(llama.modes.build, "post-training")).toEqual({ value: 10_000_000, unit: "human-marked examples", isOver: true });
    expect(headlineOf(llama.modes.use, "stop-check")).toMatchObject({ value: 128_001, isComputed: true });
    expect(partIn(llama.modes.build, "data")?.facts.notes?.find((note) => note.label === "Newest text")?.value).toContain("March 2023");
  });

  test("a word ID is never shortened on the card", () => {
    const part = partIn(llama.modes.use, "stop-check");
    if (!part) throw new Error("stop-check");
    const html = renderToStaticMarkup(<PartCard ref={null} root={llama.modes.use} modelTitle={llama.title} part={part} place={placeOf(llama, llama.modes.use, part)} index={0} isSelected={false} isOpen={false} onSelect={() => {}} onDetails={() => {}} />);
    expect(html).toContain("<b>128,001</b>");
  });
});

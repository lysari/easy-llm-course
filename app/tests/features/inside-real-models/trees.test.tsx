import { describe, expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import { MAX_CONTEXT } from "@/core/model/corpus";
import { ModelLab } from "@/core/model/lab";
import type { ModeId, ModelPart } from "@/features/inside-real-models/interfaces/model-part.interface";
import { MODELS, activeIdOf, matchingPath, openPath, pathIdsOf, pathUrl, readPath } from "@/features/inside-real-models/models";
import { PartCard } from "@/features/inside-real-models/part-card";
import { stepsTitleOf } from "@/features/inside-real-models/part-column";
import { PartDrawer } from "@/features/inside-real-models/part-drawer";
import { RUNS_LABEL, keyFactOf, placeOf, runsOf, shareOf, showAmount } from "@/features/inside-real-models/real-place";
import { readSentence } from "@/features/inside-real-models/sentence";
import { viewTiny } from "@/features/inside-real-models/tiny-parts";
import { SOURCES } from "@/features/inside-real-models/trees/sources";

const TOTALS = { "gpt-2-small": 124_439_808, "llama-3-8b": 8_030_261_248 };
const STEP_IDS: Record<ModeId, string[]> = {
  use: ["tokenize", "embedding", "layers", "output", "generate"],
  build: ["data", "tokenizer", "architecture", "pre-training", "post-training", "evaluation"],
};
const SUMMARY_LIMIT = 90;
const CHIP_LIMIT = 21;
const IN_OUT_LIMIT = 35;
const KEY_FACT_LIMIT = 100;
const TINY_COMPARISON = /tiny model|our tiny|tiny corpus|lesson model|tiny gpt/i;
const SENTENCES = ["my dog is hungry , so i give it a", "My ELEPHANT is hungry, so", ""];

const allParts = (part: ModelPart): ModelPart[] => [part, ...(part.children ?? []).flatMap(allParts)];
const escaped = (text: string) => text.replaceAll("'", "&#x27;").replaceAll('"', "&quot;");
const urlOf = (modelIndex: number, ids: string[]) => pathUrl(MODELS[modelIndex].id, readPath(MODELS[modelIndex], ids));

function trainedLab() {
  const lab = new ModelLab();
  lab.trainRounds(3);
  return lab;
}

describe.each(MODELS)("$title", (model) => {
  test("the model tree holds the official number count", () => {
    expect(model.root.facts.weightCount).toBe(TOTALS[model.id]);
  });

  describe.each(["use", "build"] as const)("%s mode", (mode) => {
    const root = model.modes[mode];
    const parts = allParts(root);

    test("the root holds the official number count", () => {
      expect(root.id).toBe(mode);
      expect(root.facts.weightCount).toBe(TOTALS[model.id]);
    });

    test("the steps come in order, each with what goes in and out", () => {
      expect(root.children?.map((step) => step.id)).toEqual(STEP_IDS[mode]);
      for (const step of root.children ?? []) {
        expect(step.facts.input.length, step.id).toBeGreaterThan(2);
        expect(step.facts.output.length, step.id).toBeGreaterThan(2);
      }
    });

    test.each(parts.filter((part) => part.children))("$id holds the sum of its parts", (part) => {
      const sum = (part.children ?? []).reduce((total, child) => total + child.facts.weightCount * child.copies, 0);
      expect(sum).toBe(part.facts.weightCount);
    });

    test("every branch ends at a smallest step", () => {
      for (const part of parts) {
        expect(Boolean(part.children?.length) !== Boolean(part.groundZero), part.id).toBe(true);
      }
    });

    test("ids are unique", () => {
      const ids = parts.map((part) => part.id);
      expect(new Set(ids).size).toBe(ids.length);
    });

    test("every part has text and a formula or code", () => {
      for (const part of parts) {
        expect(part.explanation.length, part.id).toBeGreaterThan(20);
        expect(part.why.length, part.id).toBeGreaterThan(20);
        expect(Boolean(part.formula || part.code), part.id).toBe(true);
        expect(part.copies, part.id).toBeGreaterThanOrEqual(1);
      }
    });

    test("every part says what it does in one short plain sentence", () => {
      for (const part of parts) {
        expect(part.summary.length, part.id).toBeGreaterThan(20);
        expect(part.summary.length, part.id).toBeLessThanOrEqual(SUMMARY_LIMIT);
        expect(part.summary, part.id).toMatch(/^[A-Z0-9].*\.$/);
        expect(part.summary.slice(0, -1), part.id).not.toMatch(/\. /);
      }
    });

    test("every part has what goes in and out", () => {
      for (const part of parts) {
        expect(part.facts.input.trim().length, part.id).toBeGreaterThan(2);
        expect(part.facts.output.trim().length, part.id).toBeGreaterThan(2);
      }
    });

    test("the card's in and out labels are short enough to sit on one line", () => {
      for (const { id, facts } of parts) {
        expect(facts.input.length, id).toBeLessThanOrEqual(CHIP_LIMIT);
        expect(facts.output.length, id).toBeLessThanOrEqual(CHIP_LIMIT);
        expect(facts.input.length + facts.output.length, id).toBeLessThanOrEqual(IN_OUT_LIMIT);
        if (facts.inputDetail) expect(facts.inputDetail.length, id).toBeGreaterThan(facts.input.length);
        if (facts.outputDetail) expect(facts.outputDetail.length, id).toBeGreaterThan(facts.output.length);
      }
    });

    test("every part has one real headline and one more real fact with its source", () => {
      for (const part of parts) {
        const { headline } = part.facts;
        if ("text" in headline) expect(part.facts.notes?.length, part.id).toBeGreaterThan(0);
        else {
          expect(Number.isFinite(headline.value) && headline.value >= 0, part.id).toBe(true);
          expect(headline.unit.length, part.id).toBeGreaterThan(0);
        }
        const place = placeOf(model, root, part);
        for (const isExact of [false, true]) {
          const key = keyFactOf(part, place, isExact);
          expect(key.label.length, part.id).toBeGreaterThan(2);
          expect(key.value.trim().length, part.id).toBeGreaterThan(0);
          expect(SOURCES, `${part.id}: ${key.source}`).toHaveProperty(key.source);
        }
        expect(keyFactOf(part, place, false).value.length, part.id).toBeLessThanOrEqual(KEY_FACT_LIMIT);
      }
    });

    test("a part links to the tiny model only when its demo runs on that part", () => {
      for (const part of parts) expect(Boolean(part.tiny), part.id).toBe(part.story?.demo === "tiny-part");
    });

    test.each([
      ["fresh", () => new ModelLab()],
      ["trained a little", trainedLab],
    ] as const)("cards and details have no gaps when %s", (_, makeLab) => {
      const lab = makeLab();
      for (const text of SENTENCES) {
        const sentence = readSentence(text);
        const tiny = viewTiny(lab, sentence.words);
        for (const part of parts) {
          const place = placeOf(model, root, part);
          const card = renderToStaticMarkup(<PartCard ref={null} root={root} part={part} place={place} index={0} isSelected={false} isActive={false} isOpen={false} onSelect={() => {}} onDetails={() => {}} />);
          const drawer = renderToStaticMarkup(<PartDrawer ref={null} mode={mode} root={root} modelId={model.id} modelTitle={model.title} part={part} place={place} tiny={tiny} text={text} sentence={sentence} onTextChange={() => {}} nextPart={undefined} onOpen={() => {}} onClose={() => {}} />);
          const info = drawer.split('<div class="drawer-panel">')[0];
          for (const html of [card, drawer]) expect(html, `${part.id} on "${text}"`).not.toMatch(/undefined|NaN|Infinity|\[object|\$\{/);
          expect(card, part.id).toContain(escaped(part.summary));
          expect(info, part.id).toContain(escaped(part.explanation));
          expect(card, part.id).not.toContain(`Real ${model.title}`);
          expect(card, part.id).not.toMatch(/tiny/i);
          expect(info, part.id).not.toMatch(TINY_COMPARISON);
          expect(info, part.id).toContain("<span>Goes in</span>");
        }
      }
    });
  });

  test("the steps column names the model once, step by step to use it and stage by stage to build it", () => {
    expect(stepsTitleOf(model, "use")).toBe(`${model.title}, step by step`);
    expect(stepsTitleOf(model, "build")).toBe(`${model.title}, stage by stage`);
  });

  test("tokenization and generation hold no numbers", () => {
    const steps = model.modes.use.children ?? [];
    expect(steps.find((step) => step.id === "tokenize")?.facts.weightCount).toBe(0);
    expect(steps.find((step) => step.id === "generate")?.facts.weightCount).toBe(0);
  });
});

describe("where a part sits in the real model", () => {
  const [gpt2, llama] = MODELS;
  const placeIn = (model: (typeof MODELS)[number], mode: ModeId, id: string) => {
    const root = model.modes[mode];
    const part = allParts(root).find((each) => each.id === id);
    if (!part) throw new Error(id);
    return { part, place: placeOf(model, root, part) };
  };

  test("the path and copies come from the tree", () => {
    const { place } = placeIn(gpt2, "use", "query-multiply-add");
    expect(place.path.map((part) => part.id)).toEqual(["use", "layers", "layer", "attention", "head", "query", "query-number", "query-multiply-add"]);
    expect(place.copies).toBe(12 * 12 * 64 * 768);
    expect(place.isInModel).toBe(true);
    expect(placeIn(gpt2, "use", "softmax").place.isInModel).toBe(false);
  });

  test("a multiply-and-add says how many times it runs per word", () => {
    const query = placeIn(gpt2, "build", "query-multiply-add");
    expect(keyFactOf(query.part, query.place, true)).toEqual({ label: RUNS_LABEL, value: "7,077,888 times", source: "gpt-2-config", isComputed: true });
    expect(keyFactOf(query.part, query.place, false).value).toBe("7.08M times");
    const score = placeIn(llama, "use", "score-multiply-add");
    expect(runsOf(score.part, score.place, true)).toBe("up to 1,073,741,824 times");
  });

  test("a part with stored numbers gives its share of the model", () => {
    const table = placeIn(gpt2, "use", "word-table");
    expect(shareOf(table.part, table.place, true)).toBe("38,597,376 of 124,439,808 (31%)");
    const layer = placeIn(gpt2, "use", "layer");
    expect(keyFactOf(layer.part, layer.place, false)).toMatchObject({ label: "Part of the model's 124M learned numbers", value: "68.3% for all 12" });
    const llamaTable = placeIn(llama, "use", "word-table");
    expect(keyFactOf(llamaTable.part, llamaTable.place, false).label).toBe("Part of the model's 8.03B learned numbers");
    expect(shareOf(layer.part, layer.place, true)).toBe("85,054,464 of 124,439,808 (68.3%), all 12 together");
    const architecture = placeIn(llama, "build", "architecture");
    expect(shareOf(architecture.part, architecture.place, true)).toBeNull();
  });

  test("a step outside the model shows its first published note", () => {
    const stop = placeIn(llama, "use", "stop-check");
    expect(keyFactOf(stop.part, stop.place, false)).toMatchObject({ label: "Instruct model", source: "llama-3-tokenizer", isComputed: false });
  });
});

describe("card numbers", () => {
  test("big numbers are compact on the card and exact in the drawer", () => {
    expect(showAmount(124_439_808, false)).toBe("124M");
    expect(showAmount(8_030_261_248, false)).toBe("8.03B");
    expect(showAmount(50_257, false)).toBe("50,257");
    expect(showAmount(8_030_261_248, true)).toBe("8,030,261,248");
  });
});

describe("paths", () => {
  const [gpt2, llama] = MODELS;

  test("a valid path opens every level", () => {
    expect(urlOf(0, ["use", "layers", "layer", "attention", "head", "query", "query-number"])).toBe("/inside-real-models/gpt-2-small/use/layers/layer/attention/head/query/query-number");
    expect(urlOf(0, ["build", "architecture", "model", "word-table"])).toBe("/inside-real-models/gpt-2-small/build/architecture/model/word-table");
  });

  test("a wrong id or a smallest-step part stops at the last level with parts", () => {
    expect(urlOf(0, ["use", "layers", "nope", "attention"])).toBe("/inside-real-models/gpt-2-small/use/layers");
    expect(urlOf(0, ["use", "tokenize", "lookup"])).toBe("/inside-real-models/gpt-2-small/use/tokenize");
  });

  test("no mode, or an old path, goes to the matching Use path", () => {
    expect(urlOf(0, [""])).toBe("/inside-real-models/gpt-2-small/use");
    expect(urlOf(0, ["layers", "layer", "attention"])).toBe("/inside-real-models/gpt-2-small/use/layers/layer/attention");
    expect(urlOf(0, ["position-table", "position-table-row"])).toBe("/inside-real-models/gpt-2-small/use/embedding/position-table/position-table-row");
    expect(urlOf(0, ["output-head", "output-score"])).toBe("/inside-real-models/gpt-2-small/use/output/output-head/output-score");
    expect(urlOf(1, ["layers", "nope"])).toBe("/inside-real-models/llama-3-8b/use/layers");
    expect(urlOf(1, ["nope"])).toBe("/inside-real-models/llama-3-8b/use");
  });

  test("switching mode keeps the deepest part that exists, else the root", () => {
    const attention = readPath(gpt2, ["use", "layers", "layer", "attention"]);
    expect(pathUrl(gpt2.id, matchingPath(gpt2.modes.build, attention))).toBe("/inside-real-models/gpt-2-small/build/architecture/model/layers/layer/attention");
    const back = readPath(gpt2, ["build", "architecture", "model", "word-table"]);
    expect(pathUrl(gpt2.id, matchingPath(gpt2.modes.use, back))).toBe("/inside-real-models/gpt-2-small/use/embedding/word-table");
    const generate = readPath(gpt2, ["use", "generate", "write-round"]);
    expect(pathUrl(gpt2.id, matchingPath(gpt2.modes.build, generate))).toBe("/inside-real-models/gpt-2-small/build");
  });

  test("switching model keeps the mode and part when it exists, else its nearest parent", () => {
    const query = readPath(gpt2, ["use", "layers", "layer", "attention", "head", "query"]);
    expect(pathUrl(llama.id, matchingPath(llama.modes.use, query))).toBe("/inside-real-models/llama-3-8b/use/layers/layer/attention/group/head/query");
    const gelu = readPath(gpt2, ["build", "architecture", "model", "layers", "layer", "feed-forward", "gelu"]);
    expect(pathUrl(llama.id, matchingPath(llama.modes.build, gelu))).toBe("/inside-real-models/llama-3-8b/build/architecture/model/layers/layer/feed-forward");
    const position = readPath(gpt2, ["use", "embedding", "position-table"]);
    expect(pathUrl(llama.id, matchingPath(llama.modes.use, position))).toBe("/inside-real-models/llama-3-8b/use/embedding");
  });

  test("the open card is active, else the deepest selected card", () => {
    const data = readPath(gpt2, ["build", "data"]);
    const copyIn = data[data.length - 1].children?.find((part) => part.id === "gather");
    expect(copyIn).toBeDefined();
    expect(activeIdOf(data, copyIn)).toBe("gather");
    expect(activeIdOf(data, data[1])).toBe("data");
    expect(activeIdOf(data, undefined)).toBe("data");
  });

  test("the coloured path is the selected path plus the open card", () => {
    const architecture = readPath(gpt2, ["build", "architecture", "model"]);
    const layers = architecture[architecture.length - 1].children?.find((part) => part.id === "layers");
    expect(layers).toBeDefined();
    expect(pathIdsOf(architecture, layers)).toEqual(["build", "architecture", "model", "layers"]);
    expect(pathIdsOf(architecture, architecture[1])).toEqual(["build", "architecture", "model"]);
    expect(pathIdsOf(architecture, undefined)).toEqual(["build", "architecture", "model"]);
  });

  test("openPath still walks the model tree", () => {
    expect(openPath(gpt2.root, ["layers", "layer"]).map((part) => part.id)).toEqual(["model", "layers", "layer"]);
  });
});

describe("live example", () => {
  test("known words are kept and unknown words are flagged", () => {
    const sentence = readSentence("My ELEPHANT is hungry, so");
    expect(sentence.words).toEqual(["my", "is", "hungry", ",", "so"]);
    expect(sentence.unknown).toEqual(["elephant"]);
    expect(sentence.isSample).toBe(false);
  });

  test("an empty or all-unknown sentence falls back to the sample", () => {
    expect(readSentence("").isSample).toBe(true);
    expect(readSentence("<end> zebra").unknown).toEqual(["<end>", "zebra"]);
    expect(readSentence("<end> zebra").words.length).toBeGreaterThan(0);
  });

  test("a long sentence keeps only the words that fit", () => {
    const sentence = readSentence("the dog chases the ball . ".repeat(5));
    expect(sentence.words.length).toBe(MAX_CONTEXT - 1);
    expect(sentence.isCut).toBe(true);
  });

  test("writing is seeded, so it repeats exactly", () => {
    const lab = trainedLab();
    const words = readSentence("the cat likes the").words;
    expect(viewTiny(lab, words).written).toEqual(viewTiny(lab, words).written);
  });
});

import { describe, expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import { MAX_CONTEXT } from "@/core/model/corpus";
import { ModelLab } from "@/core/model/lab";
import type { ModeId, ModelPart } from "@/features/take-apart/interfaces/model-part.interface";
import { MODELS, matchingPath, openPath, pathUrl, readPath } from "@/features/take-apart/models";
import { PartCard } from "@/features/take-apart/part-card";
import { PartDrawer } from "@/features/take-apart/part-drawer";
import { showAmount } from "@/features/take-apart/part-numbers";
import { readSentence } from "@/features/take-apart/sentence";
import { readTinyPart, viewTiny } from "@/features/take-apart/tiny-parts";

const TOTALS = { "gpt-2-small": 124_439_808, "llama-3-8b": 8_030_261_248 };
const STEP_IDS: Record<ModeId, string[]> = {
  use: ["tokenize", "embedding", "layers", "output", "generate"],
  build: ["data", "tokenizer", "architecture", "pre-training", "post-training", "evaluation"],
};
const SUMMARY_LIMIT = 90;
const CHIP_LIMIT = 21;
const IN_OUT_LIMIT = 35;
const SENTENCES = ["my dog is hungry , so i give it a", "My ELEPHANT is hungry, so", ""];

const allParts = (part: ModelPart): ModelPart[] => [part, ...(part.children ?? []).flatMap(allParts)];
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

    test("every branch ends at ground zero", () => {
      for (const part of parts) {
        expect(Boolean(part.children?.length) !== Boolean(part.groundZero), part.id).toBe(true);
      }
    });

    test("ids are unique", () => {
      const ids = parts.map((part) => part.id);
      expect(new Set(ids).size).toBe(ids.length);
    });

    test("every part has text, lesson labels and a formula or code", () => {
      for (const part of parts) {
        expect(part.explanation.length, part.id).toBeGreaterThan(20);
        expect(part.why.length, part.id).toBeGreaterThan(20);
        expect(part.lesson, part.id).toMatch(/^(Lessons? \d\d((, |–)\d\d)*|No lesson yet)$/);
        expect(part.reverseLesson, part.id).toMatch(/^RL-0[0-6]$/);
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

    test("every part has one real headline number, and a tiny one or says the tiny model lacks it", () => {
      const tiny = viewTiny(new ModelLab(), readSentence(SENTENCES[0]).words);
      for (const part of parts) {
        const { headline } = part.facts;
        if ("text" in headline) expect(part.facts.notes?.length, part.id).toBeGreaterThan(0);
        else {
          expect(Number.isFinite(headline.value) && headline.value >= 0, part.id).toBe(true);
          expect(headline.unit.length, part.id).toBeGreaterThan(0);
        }
        if (part.tiny.missing) {
          expect(part.tiny.missing.length, part.id).toBeGreaterThan(10);
          continue;
        }
        const tinyHeadline = readTinyPart(part.tiny, tiny).headline;
        if ("reason" in tinyHeadline) {
          expect(tinyHeadline.reason, part.id).toMatch(/^[A-Z].*\.$/);
          continue;
        }
        expect(Number.isFinite(tinyHeadline.value) && tinyHeadline.value >= 0, part.id).toBe(true);
        expect(tinyHeadline.unit.length, part.id).toBeGreaterThan(0);
        if (/^numbers?$/.test(tinyHeadline.unit)) expect(tinyHeadline.value, part.id).toBeGreaterThan(0);
      }
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
          const html = renderToStaticMarkup(
            <>
              <PartCard ref={null} root={root} modelTitle={model.title} part={part} tiny={tiny} index={0} isSelected={false} isOpen={false} onSelect={() => {}} onDetails={() => {}} />
              <PartDrawer ref={null} root={root} modelTitle={model.title} part={part} tiny={tiny} sentence={sentence} onClose={() => {}} />
            </>,
          );
          expect(html, `${part.id} on "${text}"`).not.toMatch(/undefined|NaN|Infinity|\[object|\$\{/);
          expect(html, part.id).toContain(part.summary.replaceAll("'", "&#x27;").replaceAll('"', "&quot;"));
          expect(html, part.id).toContain(part.tiny.missing ? "Not in the tiny model" : `Real ${model.title}`);
          for (const tinyNumbers of html.split("<dt>Tiny model").slice(1)) {
            expect(tinyNumbers.slice(0, tinyNumbers.indexOf("</dl>")), part.id).not.toMatch(/<b>0<\/b> numbers?</);
          }
        }
      }
    });
  });

  test("tokenization and generation hold no numbers", () => {
    const steps = model.modes.use.children ?? [];
    expect(steps.find((step) => step.id === "tokenize")?.facts.weightCount).toBe(0);
    expect(steps.find((step) => step.id === "generate")?.facts.weightCount).toBe(0);
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
    expect(urlOf(0, ["use", "layers", "layer", "attention", "head", "query", "query-number"])).toBe("/take-apart/gpt-2-small/use/layers/layer/attention/head/query/query-number");
    expect(urlOf(0, ["build", "architecture", "model", "word-table"])).toBe("/take-apart/gpt-2-small/build/architecture/model/word-table");
  });

  test("a wrong id or a ground-zero part stops at the last level with parts", () => {
    expect(urlOf(0, ["use", "layers", "nope", "attention"])).toBe("/take-apart/gpt-2-small/use/layers");
    expect(urlOf(0, ["use", "tokenize", "lookup"])).toBe("/take-apart/gpt-2-small/use/tokenize");
  });

  test("no mode, or an old path, goes to the matching Use path", () => {
    expect(urlOf(0, [""])).toBe("/take-apart/gpt-2-small/use");
    expect(urlOf(0, ["layers", "layer", "attention"])).toBe("/take-apart/gpt-2-small/use/layers/layer/attention");
    expect(urlOf(0, ["position-table", "position-table-row"])).toBe("/take-apart/gpt-2-small/use/embedding/position-table/position-table-row");
    expect(urlOf(0, ["output-head", "output-score"])).toBe("/take-apart/gpt-2-small/use/output/output-head/output-score");
    expect(urlOf(1, ["layers", "nope"])).toBe("/take-apart/llama-3-8b/use/layers");
    expect(urlOf(1, ["nope"])).toBe("/take-apart/llama-3-8b/use");
  });

  test("switching mode keeps the deepest part that exists, else the root", () => {
    const attention = readPath(gpt2, ["use", "layers", "layer", "attention"]);
    expect(pathUrl(gpt2.id, matchingPath(gpt2.modes.build, attention))).toBe("/take-apart/gpt-2-small/build/architecture/model/layers/layer/attention");
    const back = readPath(gpt2, ["build", "architecture", "model", "word-table"]);
    expect(pathUrl(gpt2.id, matchingPath(gpt2.modes.use, back))).toBe("/take-apart/gpt-2-small/use/embedding/word-table");
    const generate = readPath(gpt2, ["use", "generate", "write-round"]);
    expect(pathUrl(gpt2.id, matchingPath(gpt2.modes.build, generate))).toBe("/take-apart/gpt-2-small/build");
  });

  test("switching model keeps the mode and part when it exists, else its nearest parent", () => {
    const query = readPath(gpt2, ["use", "layers", "layer", "attention", "head", "query"]);
    expect(pathUrl(llama.id, matchingPath(llama.modes.use, query))).toBe("/take-apart/llama-3-8b/use/layers/layer/attention/group/head/query");
    const gelu = readPath(gpt2, ["build", "architecture", "model", "layers", "layer", "feed-forward", "gelu"]);
    expect(pathUrl(llama.id, matchingPath(llama.modes.build, gelu))).toBe("/take-apart/llama-3-8b/build/architecture/model/layers/layer/feed-forward");
    const position = readPath(gpt2, ["use", "embedding", "position-table"]);
    expect(pathUrl(llama.id, matchingPath(llama.modes.use, position))).toBe("/take-apart/llama-3-8b/use/embedding");
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

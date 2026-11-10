import { describe, expect, test } from "bun:test";
import { TEST_WORDS, VOCABULARY } from "@/core/model/corpus";
import { nextOptions } from "@/core/model/generate";
import { ModelLab } from "@/core/model/lab";
import { traceForward } from "@/core/model/tiny-gpt";
import { showPower, softmaxOf } from "@/features/inside-real-models/story/output/next-word";

const lab = new ModelLab();
lab.trainRounds(5);
const trace = traceForward(TEST_WORDS.map((word) => VOCABULARY.indexOf(word)), lab.live);
const scores = trace.logits[TEST_WORDS.length - 1];

describe("softmax worked out by hand", () => {
  test("each chance is e^score over the total, and they add up to 1", () => {
    const { rows, total } = softmaxOf(scores);
    expect(rows).toHaveLength(VOCABULARY.length);
    expect(rows.reduce((sum, row) => sum + row.chance, 0)).toBeCloseTo(1, 12);
    for (const row of rows) expect(row.chance).toBeCloseTo(Math.exp(row.score) / total, 12);
  });

  test("gives the same chances, in the same order, as the model's own guess", () => {
    const own = nextOptions(TEST_WORDS, lab.live);
    softmaxOf(scores).rows.forEach((row, i) => {
      expect(row.word).toBe(own[i].token);
      expect(row.chance).toBeCloseTo(own[i].chance, 12);
    });
  });

  test("writes big powers as whole numbers and small ones with 3 decimals", () => {
    expect(showPower(22026.47)).toBe("22,026");
    expect(showPower(0.5)).toBe("0.500");
  });
});

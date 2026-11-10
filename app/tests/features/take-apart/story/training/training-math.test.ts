import { describe, expect, test } from "bun:test";
import { TRAINING_LINES, VOCABULARY } from "@/core/model/corpus";
import { ModelLab } from "@/core/model/lab";
import { START_TRACE, adamStep, slopeCheck, wordChances } from "@/features/take-apart/story/training/training-math";

describe("the training demo's maths", () => {
  test("each word's loss is minus the log of its chance", () => {
    const lab = new ModelLab();
    const chances = wordChances(TRAINING_LINES[0].text.split(" "), lab.live);
    expect(chances.length).toBe(TRAINING_LINES[0].text.split(" ").length);
    for (const each of chances) {
      expect(each.chance).toBeGreaterThan(0);
      expect(each.loss).toBeCloseTo(-Math.log(each.chance), 12);
      expect(VOCABULARY).toContain(each.word);
    }
  });

  test("the chain-rule slope matches the slope measured by nudging the number", () => {
    const check = slopeCheck(new ModelLab().live);
    expect(check.measured).toBeCloseTo(check.slope, 6);
  });

  test("the step worked out from m and v is the step the tiny model's Adam really took", () => {
    const lab = new ModelLab();
    let trace = START_TRACE;
    for (let round = 0; round < 3; round++) {
      const before = lab.live.tokenTable.data[0];
      lab.trainRounds(1);
      trace = adamStep(trace, lab.live.tokenTable.grad[0]);
      expect(before - lab.live.tokenTable.data[0]).toBeCloseTo(trace.step, 12);
    }
  });
});

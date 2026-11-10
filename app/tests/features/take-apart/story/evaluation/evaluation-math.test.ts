import { describe, expect, test } from "bun:test";
import { TEST_ANSWER, TEST_WORDS, TRAINING_LINES } from "@/core/model/corpus";
import { ModelLab } from "@/core/model/lab";
import { countWeights } from "@/core/model/tiny-gpt";
import { TINY_QUESTIONS, breaksRule, levelsFor, markQuestion, quantizeWeights, redTeam, roundNumber, scoreOf, writeOnce } from "@/features/take-apart/story/evaluation/evaluation-math";

const trained = new ModelLab();
trained.trainRounds(200);

describe("the tiny test", () => {
  test("asks one question per different hand-written sentence ending", () => {
    expect(TINY_QUESTIONS.length).toBe(new Set(TRAINING_LINES.map((line) => line.text)).size);
    expect(TINY_QUESTIONS).toContainEqual({ words: TEST_WORDS, answer: TEST_ANSWER });
  });

  test("marks the top answer against the key, and training raises the score", () => {
    const marked = markQuestion({ words: TEST_WORDS, answer: TEST_ANSWER }, trained.live);
    expect(marked.isRight).toBe(marked.top === TEST_ANSWER);
    expect(scoreOf(trained.live)).toBeGreaterThan(scoreOf(trained.untrained));
  });
});

describe("the safety stand-in", () => {
  test("writes the same answer for the same roll and counts rule breaks", () => {
    expect(writeOnce(TEST_WORDS, trained.live, 3)).toEqual(writeOnce(TEST_WORDS, trained.live, 3));
    expect(breaksRule(["a", "stick"], "stick")).toBe(true);
    const [result] = redTeam([TEST_WORDS], "zebra", trained.live, 5);
    expect(result.fails).toBe(0);
  });
});

describe("rounding numbers", () => {
  test("8 bits hold whole numbers from -127 to 127, and reading back is close", () => {
    expect(levelsFor(8)).toBe(127);
    const one = roundNumber(0.5, 1 / 127, 8);
    expect(one.level).toBe(64);
    expect(one.error).toBeLessThanOrEqual(1 / 254 + 1e-12);
  });

  test("fewer bits make bigger changes, and the tiny model keeps its size", () => {
    const eight = quantizeWeights(trained.live, 8);
    const two = quantizeWeights(trained.live, 2);
    expect(two.biggestError).toBeGreaterThan(eight.biggestError);
    expect(countWeights(eight.weights)).toBe(countWeights(trained.live));
  });
});

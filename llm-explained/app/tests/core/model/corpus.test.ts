import { describe, expect, test } from "bun:test";
import { END_TOKEN, MAX_CONTEXT, TRAINING_SEQUENCES, VOCABULARY, joinTokens } from "@/core/model/corpus";
import { nextOptions, pickToken, writeText } from "@/core/model/generate";
import { seededRandom } from "@/core/model/random";
import { createWeights } from "@/core/model/tiny-gpt";

test("joinTokens writes normal text and hides the stop token", () => {
  expect(joinTokens(["my", "dog", ",", "a", "bone", ".", END_TOKEN])).toBe("my dog, a bone.");
});

test("every training sentence ends with the stop token and fits the context", () => {
  for (const sequence of TRAINING_SEQUENCES) {
    expect(VOCABULARY[sequence[sequence.length - 1]]).toBe(END_TOKEN);
    expect(sequence.length).toBeLessThanOrEqual(MAX_CONTEXT + 1);
  }
});

describe("writing", () => {
  const weights = createWeights();

  test("picking follows the chances", () => {
    const random = seededRandom(1);
    const options = nextOptions(["my", "dog"], weights);
    const TRIALS = 5000;
    let firstCount = 0;
    for (let i = 0; i < TRIALS; i++) if (pickToken(options, random).token === options[0].token) firstCount++;
    expect(firstCount / TRIALS).toBeCloseTo(options[0].chance, 1);
  });

  test("writing always stops within the context window", () => {
    expect(writeText(["my"], weights, seededRandom(2)).length).toBeLessThanOrEqual(MAX_CONTEXT);
  });
});

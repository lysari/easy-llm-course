import { describe, expect, test } from "bun:test";
import { Adam } from "../../src/model/adam";
import { END_TOKEN, TRAINING_SEQUENCES, VOCABULARY } from "../../src/model/corpus";
import { TARGET_STEPS } from "../../src/model/lab";
import { allTables, createWeights, lossAndGradients, measureLoss, predictNext, traceForward } from "../../src/model/tiny-gpt";

const ids = (text: string) => text.split(" ").map((token) => VOCABULARY.indexOf(token));
const chanceOf = (text: string, token: string, weights: ReturnType<typeof createWeights>) =>
  predictNext(ids(text), weights).find((option) => option.token === token)?.chance ?? 0;

const trainedWeights = createWeights();
const optimizer = new Adam(trainedWeights);
const untrainedLoss = measureLoss(TRAINING_SEQUENCES, trainedWeights);
const startedAt = performance.now();
for (let step = 0; step < TARGET_STEPS; step++) {
  lossAndGradients(TRAINING_SEQUENCES, trainedWeights);
  optimizer.update();
}
const millisecondsPerStep = (performance.now() - startedAt) / TARGET_STEPS;

describe("gradients", () => {
  test("backpropagation matches a numeric slope for every table", () => {
    const weights = createWeights(7);
    const sample = TRAINING_SEQUENCES.slice(0, 2);
    lossAndGradients(sample, weights);
    const NUDGE = 1e-5;
    for (const { name, tensor } of allTables(weights)) {
      const index = tensor.grad.reduce((best, g, i, all) => (Math.abs(g) > Math.abs(all[best]) ? i : best), 0);
      const analytic = tensor.grad[index];
      const original = tensor.data[index];
      tensor.data[index] = original + NUDGE;
      const above = measureLoss(sample, weights);
      tensor.data[index] = original - NUDGE;
      const below = measureLoss(sample, weights);
      tensor.data[index] = original;
      const numeric = (above - below) / (2 * NUDGE);
      expect(Math.abs(numeric - analytic), name).toBeLessThan(1e-5 + Math.abs(analytic) * 0.01);
    }
  });
});

describe("training", () => {
  test("starts near the random baseline and ends far below it", () => {
    expect(untrainedLoss).toBeGreaterThan(Math.log(VOCABULARY.length) * 0.7);
    expect(measureLoss(TRAINING_SEQUENCES, trainedWeights)).toBeLessThan(0.5);
  });

  test.each([
    ["my dog is hungry , so i give it a", "bone"],
    ["my cat is hungry , so i give it a", "fish"],
    ["my dog is bored , so i give it a", "ball"],
    ["my cat is bored , so i give it a", "toy"],
  ])("%s predicts %s", (prompt, expected) => {
    expect(predictNext(ids(prompt), trainedWeights)[0].token).toBe(expected);
  });

  test("the percents it learns reflect how often each ending appeared", () => {
    const bone = chanceOf("my dog is hungry , so i give it a", "bone", trainedWeights);
    const treat = chanceOf("my dog is hungry , so i give it a", "treat", trainedWeights);
    expect(bone).toBeGreaterThan(0.6);
    expect(bone).toBeLessThan(0.9);
    expect(treat).toBeGreaterThan(0.1);
    expect(treat).toBeLessThan(0.4);
  });

  test("it learns to stop", () => {
    expect(predictNext(ids("my dog is hungry , so i give it a bone ."), trainedWeights)[0].token).toBe(END_TOKEN);
  });

  test("attention can only look backwards and each row adds up to 100%", () => {
    const { attention } = traceForward(ids("my dog is hungry"), trainedWeights).blocks[0];
    attention.forEach((row, r) => {
      expect(row.reduce((sum, share) => sum + share, 0)).toBeCloseTo(1, 6);
      row.forEach((share, c) => {
        if (c > r) expect(share).toBe(0);
      });
    });
  });

  test("one step is fast enough to train live", () => {
    console.log(`untrained loss ${untrainedLoss.toFixed(3)}, trained loss ${measureLoss(TRAINING_SEQUENCES, trainedWeights).toFixed(3)}, ${millisecondsPerStep.toFixed(2)} ms/step`);
    expect(millisecondsPerStep).toBeLessThan(30);
  });
});

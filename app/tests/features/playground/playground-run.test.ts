import { describe, expect, test } from "bun:test";
import { START_CHOICES, TRAINING_SEQUENCES } from "@/core/model/corpus";
import { nextOptions } from "@/core/model/generate";
import { ModelLab, TARGET_ROUNDS } from "@/core/model/lab";
import { PlaygroundRun, TEXT_GROUPS, defaultSettings, sequencesFor } from "@/features/playground/playground-run";

const START = START_CHOICES[0];
const noAttention = { hasAttention: false, hasFeedForward: true };

function trained(change: Partial<ReturnType<typeof defaultSettings>> = {}) {
  const run = new PlaygroundRun({ ...defaultSettings(START), ...change });
  while (!run.isDone) run.trainRound();
  return run;
}

const topWord = (run: PlaygroundRun) => nextOptions(START.split(" "), run.weights)[0].token;

const DEFAULT_RUN = trained();

describe("playground settings change the real model", () => {
  test("the default settings train exactly like the lessons' model", () => {
    const lab = new ModelLab();
    lab.trainRounds(TARGET_ROUNDS);
    expect(DEFAULT_RUN.latestLoss).toBeCloseTo(lab.latestLoss as number, 10);
  });

  test("every text group has lines, and together they are the whole text", () => {
    for (const group of TEXT_GROUPS) {
      expect(sequencesFor({ textGroups: [group.name], isEvenAnswers: false }).length, group.name).toBeGreaterThan(0);
    }
    const all = sequencesFor(defaultSettings(START));
    expect(all).toEqual(TRAINING_SEQUENCES);
  });

  test("equal answers give every line one copy", () => {
    const lines = sequencesFor({ textGroups: ["hungry dog"], isEvenAnswers: true });
    expect(lines).toHaveLength(2);
  });

  test("removing text changes what it answers", () => {
    expect(topWord(DEFAULT_RUN)).toBe("bone");
    expect(topWord(trained({ textGroups: ["hungry cat", "bored dog", "bored cat", "likes", "chases"] }))).not.toBe("bone");
  });

  test("attention off removes its numbers and loses the difference between dog and cat", () => {
    const run = trained({ layers: [noAttention, noAttention] });
    expect(run.numberCount).toBeLessThan(DEFAULT_RUN.numberCount);
    expect(nextOptions(START.split(" "), run.weights)[0].chance).toBeLessThan(0.5);
  });

  test("a model with no layers still trains", () => {
    const run = trained({ layers: [], rounds: 20 });
    expect(run.history[19].loss).toBeLessThan(run.history[0].loss);
  });

  test("plain steps with the same step size learn slower than Adam", () => {
    expect(trained({ optimizer: "plain" }).latestLoss as number).toBeGreaterThan(DEFAULT_RUN.latestLoss as number);
  });

  test("a run stops at its round count", () => {
    expect(trained({ rounds: 20 }).history).toHaveLength(20);
  });
});

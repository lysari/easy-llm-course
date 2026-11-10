import { describe, expect, test } from "bun:test";
import { END_TOKEN, MAX_CONTEXT, TEST_WORDS } from "@/core/model/corpus";
import { nextOptions, pickToken, writeText } from "@/core/model/generate";
import { ModelLab } from "@/core/model/lab";
import { seededRandom } from "@/core/model/random";
import { chancesFor, rollWalk, writeRounds } from "@/features/inside-real-models/story/generate/decode";

const lab = new ModelLab();
lab.trainRounds(20);
const weights = lab.live;
const PLAIN = { temperature: 1, topK: 0, topP: 1 };
const added = (rounds: ReturnType<typeof writeRounds>) => rounds.filter((round) => round.stop === null || round.stop === "full").map((round) => round.picked.token);

describe("writing word by word", () => {
  test("with plain settings it writes exactly what the model's own writeText writes", () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      const rounds = writeRounds(TEST_WORDS, weights, PLAIN, seed, null);
      expect([...TEST_WORDS, ...added(rounds)]).toEqual(writeText(TEST_WORDS, weights, seededRandom(seed)));
    }
  });

  test("each round reads every word written so far", () => {
    const rounds = writeRounds(TEST_WORDS, weights, PLAIN, 3, null);
    rounds.forEach((round, i) => expect(round.words).toEqual([...TEST_WORDS, ...added(rounds).slice(0, i)]));
  });

  test("temperature 0 always takes the top word and rolls nothing", () => {
    const rounds = writeRounds(["the"], weights, { ...PLAIN, temperature: 0 }, 1, null);
    for (const round of rounds) {
      expect(round.roll).toBeNull();
      expect(round.picked.token).toBe(nextOptions(round.words, weights)[0].token);
    }
  });

  test("top-k keeps only that many words, and their chances add up to 1", () => {
    for (const round of writeRounds(TEST_WORDS, weights, { ...PLAIN, topK: 2 }, 9, null)) {
      expect(round.options.length).toBeLessThanOrEqual(2);
      expect(round.options.reduce((sum, option) => sum + option.chance, 0)).toBeCloseTo(1, 10);
    }
  });

  test("it stops at the stop word, at your own stop word, or when the text is full", () => {
    for (const seed of [1, 2, 3, 4, 5, 6]) {
      const rounds = writeRounds(["the"], weights, PLAIN, seed, null);
      const last = rounds[rounds.length - 1];
      if (last.stop === "stop-word") expect(last.picked.token).toBe(END_TOKEN);
      else expect([last.stop, last.words.length + 1]).toEqual(["full", MAX_CONTEXT]);
      expect(rounds.slice(0, -1).every((round) => round.stop === null)).toBe(true);
    }
    const top = chancesFor(TEST_WORDS, weights, { ...PLAIN, temperature: 0 })[0].token;
    const stopped = writeRounds(TEST_WORDS, weights, { ...PLAIN, temperature: 0 }, 1, top);
    expect(stopped).toHaveLength(1);
    expect(stopped[0].stop).toBe("your-stop");
  });
});

describe("the roll", () => {
  test("walking down the chances lands on the same word the model picks", () => {
    const options = nextOptions(TEST_WORDS, weights);
    for (const roll of [0.01, 0.3, 0.5, 0.77, 0.99]) {
      const walk = rollWalk(options, roll);
      expect(walk[walk.length - 1].option.token).toBe(pickToken(options, () => roll).token);
      expect(walk[walk.length - 1].left).toBeLessThanOrEqual(0);
      expect(walk.slice(0, -1).every((step) => step.left > 0)).toBe(true);
    }
  });
});

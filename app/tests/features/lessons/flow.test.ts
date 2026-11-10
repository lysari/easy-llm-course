import { describe, expect, test } from "bun:test";
import { readdirSync } from "node:fs";
import { join } from "node:path";
import { renderToStaticMarkup } from "react-dom/server";
import { END_TOKEN, START_CHOICES, TEST_ANSWER } from "@/core/model/corpus";
import { TARGET_ROUNDS } from "@/core/model/lab";
import { LINE_TOTAL_ROUNDS, SALARIES, YEARS, lineLoss, predictOnLine } from "@/core/model/line-model";
import { LESSON_ROUNDS, NETWORK_TOTAL_ROUNDS, SmallNetwork } from "@/core/model/small-network";
import { RUN_GLOW_STYLE, STAGES, STEPS, STEP_ORDER_MS, TRAINING_PATH, activeIdsOf, stepsOf } from "@/features/lessons/flow";
import { rollUp } from "@/features/lessons/roll-up";
import { Session } from "@/features/lessons/session";

const LONG_ENOUGH = 10_000;
const MIN_DETAILS = 2;

function finishedSession() {
  const session = new Session();
  session.runCommand("start:line");
  session.runCommand("start:network");
  session.runCommand("start:train");
  session.tick(LONG_ENOUGH);
  session.runCommand(`choose:${START_CHOICES[2]}`);
  session.runCommand("pick-many");
  session.runCommand("start:write");
  session.tick(LONG_ENOUGH);
  return session;
}

const FINISHED = finishedSession();

const states = [
  ["fresh", () => new Session()],
  ["mid-training", () => {
    const session = new Session();
    session.runCommand("start:train");
    session.tick(2);
    return session;
  }],
  ["finished", () => FINISHED],
  ["with sampling settings", () => {
    const session = finishedSession();
    session.runCommand("temperature:0.5");
    session.runCommand("topk:2");
    session.runCommand("topp:0.9");
    session.runCommand("curve:relu");
    session.runCommand("line:13,25");
    return session;
  }],
] as const;

describe.each(states)("every step when %s", (_, makeSession) => {
  const session = makeSession();
  test("has no gaps in anything it shows", () => {
    for (const step of STEPS) {
      const rows = step.rows?.(session) ?? [];
      const text = [
        step.title, step.lesson, step.goal, step.from(session), step.to(session), step.why,
        ...rows.flatMap((row) => [row.text, row.value ?? ""]),
        ...(step.formula?.lines ?? []), ...(step.formula?.legend ?? []).flatMap(({ term, meaning }) => [term, meaning]),
        renderToStaticMarkup(step.visual?.(session)), renderToStaticMarkup(step.note?.(session)),
      ].join(" ");
      expect(text, step.id).not.toMatch(/undefined|NaN|Infinity|\[object|\$\{/);
      expect(text, step.id).not.toContain(END_TOKEN);
      expect(text, step.id).not.toContain(renderToStaticMarkup(END_TOKEN));
    }
  });
});

describe("the flow", () => {
  test("every lesson from 00 to 30 has a step", () => {
    const folders = readdirSync(join(import.meta.dir, "../../../../lessons")).map((name) => name.slice(0, 2)).sort();
    const named = new Set(STEPS.flatMap((step) => step.lesson.match(/\d{2}/g) ?? []));
    expect(folders.filter((lesson) => !named.has(lesson))).toEqual([]);
  });

  test("every step has real detail, not only a label", () => {
    const session = FINISHED;
    for (const step of STEPS) {
      const details = [step.rows?.(session).length, step.visual, step.formula, step.code, step.note].filter(Boolean).length;
      expect(details, step.id).toBeGreaterThanOrEqual(MIN_DETAILS);
      expect(step.why.length, step.id).toBeGreaterThan(20);
    }
  });

  test("the stages run in order and each step belongs to one", () => {
    const stageIds = STAGES.map((stage) => stage.id);
    const order = STEPS.map((step) => stageIds.indexOf(step.stage));
    expect([...order].sort((a, b) => a - b)).toEqual(order);
    expect(order).not.toContain(-1);
  });

  test("the training path follows the real order of a training round", () => {
    const ids = STEPS.map((step) => step.id);
    const positions = TRAINING_PATH.map((id) => ids.indexOf(id));
    expect(positions).not.toContain(-1);
    expect([...positions].sort((a, b) => a - b)).toEqual(positions);
    expect(TRAINING_PATH.slice(-4)).toEqual(["loss", "backprop", "adam", "train-loop"]);
  });

  test("every stage has parts, and a status taken from them", () => {
    const session = FINISHED;
    for (const stage of STAGES) {
      expect(stepsOf(stage.id).length, stage.id).toBeGreaterThan(0);
      const text = [stage.title, stage.lesson, stage.task, stage.value(session), stage.detail(session)].join(" ");
      expect(text, stage.id).not.toMatch(/undefined|NaN|\[object/);
    }
    expect(rollUp(["done", "running", "needs"])).toBe("running");
    expect(rollUp(["done", "needs"])).toBe("needs");
    expect(rollUp(["done", "lesson"])).toBe("done");
    expect(rollUp(["lesson", "lesson"])).toBe("lesson");
    expect(rollUp(["done", "queued"])).toBe("queued");
  });
});

describe("the active card", () => {
  test("the selected stage is active and nothing is coloured until a part's drawer opens, then the path to that part is", () => {
    expect(activeIdsOf("forward", undefined, null)).toEqual({ activeCardIds: ["forward"], pathIds: ["forward"], glowIds: [] });
    expect(activeIdsOf("forward", "attention", null)).toEqual({ activeCardIds: ["attention"], pathIds: ["forward", "attention"], glowIds: ["attention"] });
  });

  test("an open nested part colours the whole chain to it, through the part before it", () => {
    expect(activeIdsOf("learn", "train-loop", null)).toEqual({ activeCardIds: ["train-loop"], pathIds: ["learn", "adam", "train-loop"], glowIds: ["train-loop"] });
  });

  test("during a training run the lines follow the running part, and the open part keeps its band", () => {
    expect(activeIdsOf("forward", undefined, "ffn")).toEqual({ activeCardIds: ["forward", "ffn"], pathIds: ["forward", "ffn"], glowIds: ["ffn"] });
    expect(activeIdsOf("forward", "attention", "repeat-layer")).toEqual({ activeCardIds: ["attention", "repeat-layer"], pathIds: ["forward", "ffn", "repeat-layer"], glowIds: ["repeat-layer"] });
    expect(activeIdsOf("data", undefined, "ffn")).toEqual({ activeCardIds: ["data", "ffn"], pathIds: ["data", "ffn"], glowIds: ["ffn"] });
  });

  test("the glow passes once per training step during a run", () => {
    expect(Object.entries(RUN_GLOW_STYLE)).toEqual([["--links-glow-duration", `${STEP_ORDER_MS}ms`]]);
  });
});

describe("statuses follow what really happens", () => {
  const step = (id: string) => STEPS.find((item) => item.id === id)!;

  test("training asks for you, runs, then finishes", () => {
    const session = new Session();
    expect(step("train-loop").status(session)).toBe("needs");
    session.runCommand("start:train");
    expect(step("train-loop").status(session)).toBe("running");
    session.tick(LONG_ENOUGH);
    expect(session.lab.round).toBe(TARGET_ROUNDS);
    expect(step("train-loop").status(session)).toBe("done");
  });

  test("using the model stays queued until it is trained", () => {
    const session = new Session();
    expect(step("generate").status(session)).toBe("queued");
    session.runCommand("start:write");
    expect(session.isRunning("write")).toBe(false);
    session.runCommand("start:train");
    session.tick(LONG_ENOUGH);
    expect(step("generate").status(session)).toBe("needs");
  });

  test("forgetting resets training and use", () => {
    const session = finishedSession();
    session.runCommand("forget");
    expect(step("train-loop").status(session)).toBe("needs");
    expect(session.lab.round).toBe(0);
    expect(session.picks).toEqual([]);
  });
});

describe("the worked numbers on the cards are real", () => {
  test("the line learns the lesson's values", () => {
    const session = new Session();
    session.runCommand("start:line");
    session.tick(LONG_ENOUGH);
    expect(session.lineRounds).toBe(LINE_TOTAL_ROUNDS);
    expect(session.learningLine.w).toBeCloseTo(12.32, 1);
    expect(session.learningLine.b).toBeCloseTo(24.29, 1);
  });

  test("the worked first step matches a real first step", () => {
    const session = new Session();
    session.runCommand("start:line");
    session.tick(1 / 2000);
    const rows = STEPS.find((item) => item.id === "descent")!.rows!(session);
    expect(Number(rows[6].value)).toBeCloseTo(session.learningLine.w, 2);
    expect(Number(rows[7].value)).toBeCloseTo(session.learningLine.b, 2);
  });

  test("the line cannot fit the bend, but the network can", () => {
    const lineScore = lineLoss(new Session().bestLine);
    const network = new SmallNetwork();
    network.trainRounds(LESSON_ROUNDS);
    expect(network.loss()).toBeGreaterThan(lineScore * 0.5);
    network.trainRounds(NETWORK_TOTAL_ROUNDS - LESSON_ROUNDS);
    expect(network.loss()).toBeLessThan(lineScore * 0.2);
  });

  test("the three lines from lesson 01 give the misses shown", () => {
    const miss = (w: number, b: number) => YEARS.reduce((sum, x, i) => sum + Math.abs(predictOnLine({ w, b }, x) - SALARIES[i]), 0);
    expect([miss(10, 30), miss(12, 26), miss(13, 25)]).toEqual([25, 21, 28]);
  });

  test("Adam's worked first step moves exactly the learning rate", () => {
    const round = new Session().adamFirstRound();
    expect(Math.abs(round.change)).toBeCloseTo(0.004, 6);
    expect(round.after).toBeCloseTo(round.value - round.change, 8);
    expect(round.mHat).toBeCloseTo(round.gradient, 8);
  });

  test("the gradient's prediction matches a measured change", () => {
    const { predicted, measured } = new Session().nudgeTest();
    expect(measured).toBeCloseTo(predicted, 2);
  });

  test("training makes the right answer the top guess", () => {
    const session = new Session();
    const before = session.answerChanceNow();
    session.runCommand("start:train");
    session.tick(LONG_ENOUGH);
    expect(session.guess()[0].token).toBe(TEST_ANSWER);
    expect(session.answerChanceNow()).toBeGreaterThan(before);
  });

  test("temperature and top-K change the percents", () => {
    const session = finishedSession();
    const normal = session.sampledOptions()[0].chance;
    session.runCommand("temperature:0.5");
    expect(session.sampledOptions()[0].chance).toBeGreaterThan(normal);
    session.runCommand("temperature:2");
    expect(session.sampledOptions()[0].chance).toBeLessThan(normal);
    session.runCommand("temperature:1");
    session.runCommand("topk:2");
    expect(session.sampledOptions()).toHaveLength(2);
    expect(session.sampledOptions().reduce((sum, option) => sum + option.chance, 0)).toBeCloseTo(1, 6);
  });

  test("writing a reply always finishes", () => {
    expect(FINISHED.isWritingDone).toBe(true);
  });
});

import type { CSSProperties } from "react";
import { END_TOKEN, TRAINING_SEQUENCES, VOCABULARY } from "@/core/model/corpus";
import { TARGET_ROUNDS } from "@/core/model/lab";
import { countWeights } from "@/core/model/tiny-gpt";
import type { FlowStage, FlowStep, StageId } from "@/features/lessons/interfaces/story.interface";
import { count } from "@/shared/utils/format";
import { forwardSteps } from "./build-steps";
import { dataSteps } from "./data-steps";
import { rollUp } from "./roll-up";
import { simpleSteps } from "./simple-steps";
import { learnSteps } from "./train-steps";
import { beyondSteps, useSteps } from "./use-steps";

export const STEPS: FlowStep[] = [...simpleSteps, ...dataSteps, ...forwardSteps, ...learnSteps, ...useSteps, ...beyondSteps];

export const stepsOf = (stage: StageId) => STEPS.filter((step) => step.stage === stage);

export const NESTED_STEPS = ["repeat-layer", "train-loop"];

export const TRAINING_PATH = ["embedding", "position", "layernorm", "attention", "ffn", "repeat-layer", "logits", "softmax", "loss", "backprop", "adam", "train-loop"];

export const STEP_ORDER_MS = 450;

export const RUN_GLOW_STYLE = { "--links-glow-duration": `${STEP_ORDER_MS}ms` } as CSSProperties;

export function activeIdsOf(stageId: StageId, openStepId: string | undefined, runningStepId: string | null) {
  const focusId = openStepId ?? stageId;
  return {
    activeCardIds: runningStepId ? [focusId, runningStepId] : [focusId],
    activeLinkIds: runningStepId ? [stageId, runningStepId] : [focusId],
  };
}

const questionTotal = TRAINING_SEQUENCES.reduce((sum, sequence) => sum + sequence.length - 1, 0);

export const STAGES: FlowStage[] = [
  {
    id: "simple",
    icon: "line",
    title: "Start simple",
    lesson: "Lessons 00–08",
    task: "Train a small model on salary data",
    status: (session) => rollUp(stepsOf("simple").map((step) => step.status(session))),
    value: ({ learningLine, lineRounds }) => (lineRounds === 0 ? "2 numbers" : `w ${learningLine.w.toFixed(1)}, b ${learningLine.b.toFixed(1)}`),
    detail: (session) => `${[session.isLineDone, session.isNetworkDone].filter(Boolean).length}/2 trained`,
  },
  {
    id: "data",
    icon: "text",
    title: "Prepare the data",
    lesson: "Lessons 09, 11, 14, 18",
    task: "Turn text into practice questions",
    status: (session) => rollUp(stepsOf("data").map((step) => step.status(session))),
    value: () => `${count(questionTotal)} questions`,
    detail: () => `${VOCABULARY.length} words`,
  },
  {
    id: "forward",
    icon: "network",
    title: "Build the model",
    lesson: "Lessons 10–15, 19, 20, 25",
    task: "Input → hidden layers → output",
    status: (session) => rollUp(stepsOf("forward").map((step) => step.status(session))),
    value: (session) => `${count(countWeights(session.lab.live))} numbers`,
    detail: (session) => (session.lab.round === 0 ? "all random" : session.lab.isTrained ? "trained" : "training"),
  },
  {
    id: "learn",
    icon: "repeat",
    title: "Train the model",
    lesson: "Lessons 11, 14, 16, 17, 22",
    task: "Repeat: guess, score, find the cause, fix",
    status: (session) => rollUp(stepsOf("learn").map((step) => step.status(session))),
    value: (session) => `loss ${(session.lab.latestLoss ?? session.startLoss).toFixed(2)}`,
    detail: (session) => `round ${session.lab.round}/${TARGET_ROUNDS} · ${Math.round(session.trainSeconds)}s`,
    action: (session) => (session.isRunning("train") || session.lab.isTrained ? null : { label: session.lab.round > 0 ? "Continue" : "Start", command: "start:train", isStrong: true }),
  },
  {
    id: "use",
    icon: "pen",
    title: "Use the model",
    lesson: "Lessons 14, 21, 23",
    task: "Give it a start, and it writes the rest",
    status: (session) => (session.lab.isTrained ? rollUp(stepsOf("use").map((step) => step.status(session))) : "queued"),
    value: (session) => (session.lab.isTrained ? `${session.written.filter((word) => word !== END_TOKEN).length} words written` : "waiting"),
    detail: (session) => (session.lab.isTrained ? "numbers stay fixed" : "train it first"),
  },
  {
    id: "beyond",
    icon: "sliders",
    title: "Make it big and helpful",
    lesson: "Lessons 22, 24, 26–30",
    task: "How real models become assistants",
    status: () => "lesson",
    value: () => `${stepsOf("beyond").length} lessons`,
    detail: () => "explained, not run",
  },
];

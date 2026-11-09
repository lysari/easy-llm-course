import { MOMENTUM_DECAY, LEARNING_RATE, VARIANCE_DECAY } from "@/core/model/adam";
import { GUESSING_LOSS, TEST_ANSWER, VOCABULARY } from "@/core/model/corpus";
import { TARGET_ROUNDS } from "@/core/model/lab";
import { countWeights } from "@/core/model/tiny-gpt";
import type { FlowStep } from "@/features/lessons/interfaces/story.interface";
import { LossPlot } from "@/shared/components/loss-plot";
import type { Status } from "@/shared/interfaces/card.interface";
import { count } from "@/shared/utils/format";
import { GradientBars, LossCurve } from "./lesson-plots";
import type { Session } from "./session";

const trainStatus = (session: Session): Status => {
  if (session.isRunning("train")) return "running";
  return session.lab.isTrained ? "done" : "needs";
};

export const learnSteps: FlowStep[] = [
  {
    id: "loss",
    stage: "learn",
    icon: "gauge",
    title: "Cross-entropy loss",
    lesson: "Lesson 11",
    goal: "Score the guess by the chance given to the right word.",
    from: () => `${VOCABULARY.length} percents + the answer`,
    to: () => "1 number: the loss",
    status: trainStatus,
    rowsTitle: "Real numbers",
    rows: (session) => {
      const chance = session.answerChanceNow();
      return [
        { text: `Chance given to "${TEST_ANSWER}"`, value: `${(chance * 100).toFixed(1)}%` },
        { text: "Loss = −ln(chance)", value: (-Math.log(chance)).toFixed(2) },
        { text: "Average over all the training text", value: (session.lab.latestLoss ?? session.startLoss).toFixed(2) },
        { text: `Random guessing = ln(${VOCABULARY.length})`, value: GUESSING_LOSS.toFixed(2) },
      ];
    },
    visual: (session) => <LossCurve chance={session.answerChanceNow()} />,
    formula: { lines: ["loss = −ln(p(right word))", "p = 100% → loss 0"] },
    code: "function crossEntropyLoss(\n  probs: number[],\n  targetIndex: number,\n) {\n  return -Math.log(probs[targetIndex] + 1e-9);\n}",
    why: "Being confidently wrong is punished hard. This is the number the whole training loop tries to lower. An untrained model starts near random guessing.",
  },
  {
    id: "backprop",
    stage: "learn",
    icon: "back",
    title: "Backprop through attention",
    lesson: "Lesson 16",
    goal: "Go right to left, and find the gradient of every number.",
    from: () => "1 loss",
    to: (session) => `${count(countWeights(session.lab.live))} gradients`,
    status: trainStatus,
    rowsTitle: "Path of the gradient, right to left",
    rows: () => [
      { text: "Loss → logits", value: "p − target" },
      { text: "logits → output table", value: "dC · Bᵀ" },
      { text: "FFN → W2, ReLU, W1", value: "chain rule" },
      { text: "attnOut → V and A", value: "dV = Aᵀ · dOut" },
      { text: "A → scores (softmax)", value: "A(i,j) (δ − A(i,l))" },
      { text: "scores → Q and K", value: "÷ √d" },
      { text: "Q, K, V → Wq, Wk, Wv", value: "chain rule" },
      { text: "→ position and token tables", value: "row by row" },
    ],
    visual: (session) => (
      <>
        <p className="section-title">Average gradient per table, first training step</p>
        <GradientBars tables={session.gradientTables()} />
      </>
    ),
    note: (session) => {
      const test = session.nudgeTest();
      return `Checked for real: one number has gradient ${test.gradient.toFixed(3)}. That predicts raising it by 0.01 changes the loss by ${test.predicted.toFixed(4)}. Measured: ${test.measured.toFixed(4)}.`;
    },
    formula: { lines: ["C = A·B → dA = dC·Bᵀ, dB = Aᵀ·dC", "∂A[i][j]/∂s[i][l] = A[i][j] (δjl − A[i][l])"] },
    why: "One backward pass gives the answer for every number at once. Trying each number separately would be far too slow.",
  },
  {
    id: "adam",
    stage: "learn",
    icon: "sliders",
    title: "Adam optimizer",
    lesson: "Lesson 17",
    goal: "Give every number its own step size.",
    from: () => "Gradients",
    to: () => "Slightly better numbers",
    status: trainStatus,
    rowsTitle: "First update of one real number",
    rows: (session) => {
      const round = session.adamFirstRound();
      return [
        { text: "Number now", value: round.value.toFixed(4) },
        { text: "Gradient g", value: round.gradient.toFixed(4) },
        { text: `1. m = ${MOMENTUM_DECAY} × 0 + 0.1 × g`, value: round.m.toFixed(5) },
        { text: `2. v = ${VARIANCE_DECAY} × 0 + 0.001 × g²`, value: round.v.toExponential(2) },
        { text: "3. m̂ = m / (1 − 0.9¹)", value: round.mHat.toFixed(4) },
        { text: "4. v̂ = v / (1 − 0.999¹)", value: round.vHat.toExponential(2) },
        { text: `5. step = ${LEARNING_RATE} × m̂ / √v̂`, value: round.change.toFixed(4) },
        { text: "New number", value: round.after.toFixed(4) },
      ];
    },
    note: (session) => `The first step is always exactly the learning rate, ${LEARNING_RATE}. Later steps vary. So far, numbers have moved ${session.averageMove().toFixed(3)} on average.`,
    formula: { lines: ["m = β1·m + (1−β1)·g", "v = β2·v + (1−β2)·g²", "θ = θ − lr · m̂ / (√v̂ + ε)"] },
    code: "m = 0.9 * m + 0.1 * g;\nv = 0.999 * v + 0.001 * g * g;\nconst mHat = m / (1 - 0.9 ** t);\nconst vHat = v / (1 - 0.999 ** t);\ntheta -= (lr * mHat) / (Math.sqrt(vHat) + 1e-8);",
    why: "m remembers direction, like momentum. v tracks how much a number bounces. Steady numbers get bigger steps, and bouncing numbers get smaller ones.",
  },
  {
    id: "train-loop",
    stage: "learn",
    icon: "repeat",
    title: "The training loop",
    lesson: "Lessons 14, 22",
    goal: `Repeat the whole path ${TARGET_ROUNDS} times.`,
    from: () => "Random numbers",
    to: (session) => (session.lab.isTrained ? "A trained model" : `Round ${session.lab.round} of ${TARGET_ROUNDS}`),
    status: trainStatus,
    actions: (session) => {
      if (session.lab.isTrained) return [{ label: "Forget everything", command: "forget" }];
      if (session.isRunning("train")) return [{ label: "Finish quickly", command: "finish", isStrong: true }];
      return [{ label: session.lab.round > 0 ? "Continue" : "Start training", command: "start:train", isStrong: true }];
    },
    visual: (session) => <LossPlot points={session.lab.history} rounds={TARGET_ROUNDS} guessingLoss={GUESSING_LOSS} />,
    rowsTitle: "Right now",
    rows: (session) => [
      { text: "Round", value: `${session.lab.round} of ${TARGET_ROUNDS}` },
      { text: "Loss on all the text", value: (session.lab.latestLoss ?? session.startLoss).toFixed(2) },
      { text: `Chance of "${TEST_ANSWER}"`, value: `${(session.answerChanceNow() * 100).toFixed(1)}%` },
      { text: "Time", value: `${Math.round(session.trainSeconds)} s` },
    ],
    note: (session) => `It writes: "${session.sampleWriting()}". The loss stops near 0.2 because the text itself is unsure: bone or treat.`,
    code: "for (let round = 0; round < 200; round++) {\n  const loss = forward(text); // guess and score\n  backward(loss); // find gradients\n  adam.update(); // fix the numbers\n}",
    why: "One step does almost nothing. Thousands of small steps add up to learning. Real models read so much they must learn general patterns, but ours just memorises.",
  },
];

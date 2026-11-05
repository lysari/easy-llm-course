import { LINE_LEARNING_RATE, LINE_TOTAL_ROUNDS, SALARIES, YEARS, lineLoss, predictOnLine } from "@/core/model/line-model";
import { LESSON_ROUNDS, NETWORK_NUMBER_COUNT, NETWORK_TOTAL_ROUNDS } from "@/core/model/small-network";
import type { FlowStep, Status } from "@/features/lessons/interfaces/story.interface";
import { count } from "@/shared/utils/format";
import { curvePlot, networkDiagram, salaryPlot } from "./lesson-plots";
import type { Session } from "./session";

const SCORED_LINE = { w: 13, b: 25 };
const LAST_DOT = 6;
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const relu = (z: number) => Math.max(0, z);
const tanh = (z: number) => Math.tanh(z);
const totalMiss = (w: number, b: number) => YEARS.reduce((sum, x, i) => sum + Math.abs(predictOnLine({ w, b }, x) - SALARIES[i]), 0);
const mse = (guesses: number[], targets: number[]) => guesses.reduce((sum, guess, i) => sum + (guess - targets[i]) ** 2, 0) / guesses.length;
const mae = (guesses: number[], targets: number[]) => guesses.reduce((sum, guess, i) => sum + Math.abs(guess - targets[i]), 0) / guesses.length;
const LOSS_TARGETS = [40, 50, 60, 70, 80];
const jobStatus = (isRunning: boolean, isDone: boolean): Status => (isRunning ? "running" : isDone ? "done" : "needs");

function firstLearningRound() {
  const errors = YEARS.map((_, i) => 0 - SALARIES[i]);
  const dwSum = errors.reduce((sum, error, i) => sum + error * YEARS[i], 0);
  const dbSum = errors.reduce((sum, error) => sum + error, 0);
  const dLdw = (2 / YEARS.length) * dwSum;
  const dLdb = (2 / YEARS.length) * dbSum;
  return { dwSum, dbSum, dLdw, dLdb, w: -LINE_LEARNING_RATE * dLdw, b: -LINE_LEARNING_RATE * dLdb };
}

const networkExplain = (session: Session) => session.network.explain(LAST_DOT);

export const simpleSteps: FlowStep[] = [
  {
    id: "model",
    stage: "simple",
    icon: "line",
    title: "What is a model?",
    lesson: "Lesson 00",
    goal: "A model is a function that predicts something useful.",
    from: () => "Years of work (x)",
    to: () => "Salary (y)",
    status: () => "done",
    rowsTitle: "Try y = 10 × x + 30",
    rows: () =>
      [1, 2, 3, 4, 5].map((x, i) => ({ text: `x = ${x} → 10 × ${x} + 30`, value: `${10 * x + 30} (true ${LOSS_TARGETS[i]}) ✓` })),
    note: () => `With wrong numbers, w = 5 and b = 10: x = 1 gives ${5 + 10}, but the true answer is 40. It is off by 25.`,
    formula: "y = w × x + b\nw (weight): how steep the line is\nb (bias): where the line starts",
    code: "function predict(x: number, w: number, b: number): number {\n  return w * x + b;\n}",
    why: "Machine learning means finding the right w and b automatically. Every model, even a very large one, follows this idea.",
  },
  {
    id: "lines",
    stage: "simple",
    icon: "line",
    title: "Linear regression",
    lesson: "Lesson 01",
    goal: "Find the straight line closest to the data.",
    from: () => "7 data points",
    to: () => "A line (w, b)",
    status: () => "done",
    rowsTitle: "Try 3 lines. Press one.",
    rows: ({ lineChoice }) => [
      { text: "w = 10, b = 30", value: `total miss ${totalMiss(10, 30)}`, command: "line:10,30", isOn: lineChoice === "10,30" },
      { text: "w = 12, b = 26", value: `total miss ${totalMiss(12, 26)}`, command: "line:12,26", isOn: lineChoice === "12,26" },
      { text: "w = 13, b = 25", value: `total miss ${totalMiss(13, 25)}`, command: "line:13,25", isOn: lineChoice === "13,25" },
    ],
    visual: ({ chosenLine }) => salaryPlot({ model: (x) => predictOnLine(chosenLine, x), showMisses: true }),
    note: () => "Dots are the true salaries. Dashed lines are the misses.",
    code: "for (const { w, b } of candidates) {\n  let totalError = 0;\n  for (let i = 0; i < x.length; i++) {\n    totalError += Math.abs(predict(x[i], w, b) - y[i]);\n  }\n}",
    why: "A line can also predict new values it never saw. But it can only ever be straight.",
  },
  {
    id: "mse",
    stage: "simple",
    icon: "gauge",
    title: "Loss function",
    lesson: "Lesson 02",
    goal: "Measure how wrong the model is, as one number.",
    from: () => "Guesses and true values",
    to: () => "1 number: the loss",
    status: () => "done",
    rowsTitle: "For w = 13, b = 25",
    rows: () => [
      ...YEARS.map((x, i) => {
        const guess = predictOnLine(SCORED_LINE, x);
        const miss = guess - SALARIES[i];
        return { text: `x = ${x}: guess ${guess}, true ${SALARIES[i]}`, value: `miss ${miss}, squared ${miss * miss}` };
      }),
      { text: "Average of the squares (MSE)", value: lineLoss(SCORED_LINE).toFixed(1) },
    ],
    note: () =>
      `Why square? Compare one big error, with guesses 140, 50, 60, 70, 80: MSE ${mse([140, 50, 60, 70, 80], LOSS_TARGETS).toFixed(0)}, but plain average miss (MAE) only ${mae([140, 50, 60, 70, 80], LOSS_TARGETS).toFixed(0)}. Squaring punishes big errors more.`,
    formula: "MSE = (1/n) × Σ (guess − true)²",
    code: "function mse(predictions: number[], targets: number[]): number {\n  const n = predictions.length;\n  let sum = 0;\n  for (let i = 0; i < n; i++) {\n    sum += (predictions[i] - targets[i]) ** 2;\n  }\n  return sum / n;\n}",
    why: "Loss is the compass. Training has one goal: make this number smaller.",
  },
  {
    id: "descent",
    stage: "simple",
    icon: "down",
    title: "Gradient descent",
    lesson: "Lesson 03",
    goal: "Change w and b, a little at a time, to lower the loss.",
    from: () => "w = 0, b = 0",
    to: ({ learningLine }) => `w = ${learningLine.w.toFixed(2)}, b = ${learningLine.b.toFixed(2)}`,
    status: (session) => jobStatus(session.isRunning("line"), session.isLineDone),
    actions: (session) => (session.isLineDone || session.isRunning("line") ? [] : [{ label: "Run 10,000 steps", command: "start:line", isStrong: true }]),
    rowsTitle: "The first step, worked out",
    rows: () => {
      const round = firstLearningRound();
      return [
        { text: "1. Guess with w = 0, b = 0", value: "every guess is 0" },
        { text: "2. Errors (guess − true)", value: SALARIES.map((y) => -y).join(", ") },
        { text: "3. Σ error × x", value: count(round.dwSum) },
        { text: "4. Σ error", value: count(round.dbSum) },
        { text: "5. dL/dw = (2/7) × Σ error × x", value: round.dLdw.toFixed(1) },
        { text: "6. dL/db = (2/7) × Σ error", value: round.dLdb.toFixed(1) },
        { text: `7. New w = 0 − ${LINE_LEARNING_RATE} × dL/dw`, value: round.w.toFixed(2) },
        { text: `8. New b = 0 − ${LINE_LEARNING_RATE} × dL/db`, value: round.b.toFixed(2) },
      ];
    },
    visual: ({ learningLine }) => salaryPlot({ model: (x) => predictOnLine(learningLine, x) }),
    note: ({ lineRounds, learningLine }) => `Step ${count(lineRounds)} of ${count(LINE_TOTAL_ROUNDS)}. Loss now: ${lineLoss(learningLine).toFixed(1)}.`,
    formula: "dL/dw = (2/n) × Σ (guess − true) × x\ndL/db = (2/n) × Σ (guess − true)\nw = w − lr × dL/dw\nb = b − lr × dL/db",
    code: "function step(): void {\n  const n = x.length;\n  let dw = 0;\n  let db = 0;\n  for (let i = 0; i < n; i++) {\n    const error = predict(x[i]) - y[i];\n    dw += error * x[i];\n    db += error;\n  }\n  w -= lr * (2 / n) * dw;\n  b -= lr * (2 / n) * db;\n}",
    why: "Like hiking down a hill blindfolded: feel the slope and take a small step downhill. The learning rate (lr) is the step size.",
  },
  {
    id: "class",
    stage: "simple",
    icon: "grid",
    title: "OOP model",
    lesson: "Lesson 04",
    goal: "Put the model and its training into one class.",
    from: () => "Loose functions",
    to: () => "One object",
    status: () => "done",
    rows: () => [
      { text: "Data it keeps", value: "w, b, x, y, lr" },
      { text: "train()", value: "the 10,000 steps" },
      { text: "predict(x)", value: "w × x + b" },
      { text: "loss()", value: "MSE" },
    ],
    code: "class LinearRegression {\n  private w = 0;\n  private b = 0;\n\n  constructor(private x: number[], private y: number[],\n              private lr = 0.01, iterations = 10000) {\n    this.train(iterations);\n  }\n\n  predict(xv: number): number {\n    return this.w * xv + this.b;\n  }\n}",
    why: "No new math here. Several separate models can now exist together, each with its own w and b.",
  },
  {
    id: "activation",
    stage: "simple",
    icon: "merge",
    title: "Activation functions",
    lesson: "Lesson 05",
    goal: "Add a curve, so the model can bend.",
    from: () => "A straight line",
    to: () => "A number passed through a curve",
    status: () => "done",
    rowsTitle: "Press a curve to see it",
    rows: ({ curveChoice }) => [
      { text: "Sigmoid: squeezes to 0–1", value: `z = 2 → ${sigmoid(2).toFixed(3)}`, command: "curve:sigmoid", isOn: curveChoice === "sigmoid" },
      { text: "ReLU: negative becomes 0", value: `z = −2 → ${relu(-2)}, z = 2 → ${relu(2)}`, command: "curve:relu", isOn: curveChoice === "relu" },
      { text: "Tanh: squeezes to −1 to 1", value: `z = 2 → ${tanh(2).toFixed(3)}`, command: "curve:tanh", isOn: curveChoice === "tanh" },
    ],
    visual: ({ curveChoice }) => {
      if (curveChoice === "relu") return curvePlot(relu, 0, 5);
      if (curveChoice === "tanh") return curvePlot(tanh, -1, 1);
      return curvePlot(sigmoid, 0, 1);
    },
    note: () => {
      const slope = (z: number) => sigmoid(z) * (1 - sigmoid(z));
      return `The problem with sigmoid: its slope at z = 10 is only ${slope(10).toFixed(6)}, so learning nearly stops. That is the vanishing gradient. ReLU's slope is 0 or 1.`;
    },
    formula: "sigmoid(z) = 1 / (1 + e^(−z))\nrelu(z) = max(0, z)\ntanh(z) = (e^z − e^(−z)) / (e^z + e^(−z))",
    code: "function sigmoid(z: number): number {\n  return 1 / (1 + Math.exp(-z));\n}\nfunction relu(z: number): number {\n  return Math.max(0, z);\n}",
    why: "Without a curve, stacked layers collapse into one straight line, however many you add.",
  },
  {
    id: "network",
    stage: "simple",
    icon: "network",
    title: "Neural network",
    lesson: "Lesson 06",
    goal: "Join neurons into layers.",
    from: () => "x",
    to: () => "Salary guess",
    status: () => "done",
    visual: () => networkDiagram(),
    rowsTitle: "Forward pass, x = 7, with the current numbers",
    rows: (session) => {
      const { hidden, out } = networkExplain(session);
      return [
        { text: "h1 = sigmoid(w1 × 7 + b1)", value: hidden[0].toFixed(3) },
        { text: "h2 = sigmoid(w2 × 7 + b2)", value: hidden[1].toFixed(3) },
        { text: "h3 = sigmoid(w3 × 7 + b3)", value: hidden[2].toFixed(3) },
        { text: "out = sigmoid(wO1·h1 + wO2·h2 + wO3·h3 + bOut)", value: out.toFixed(3) },
        { text: "Salary = out × 115", value: (out * 115).toFixed(1) },
      ];
    },
    note: () => `One neuron is: weighted sum, plus bias, then a curve. This network has ${NETWORK_NUMBER_COUNT} numbers to learn: 3 × (w, b) + 3 output weights + 1 output bias.`,
    formula: "neuron(x) = sigmoid(w · x + b)\nlayer = many neurons side by side",
    code: "function neuron(inputs: number[], weights: number[], bias: number) {\n  const z = inputs.reduce((sum, xi, i) => sum + xi * weights[i], bias);\n  return sigmoid(z);\n}",
    why: "The middle layer is the hidden layer. More neurons can make more complex shapes.",
  },
  {
    id: "backprop-small",
    stage: "simple",
    icon: "back",
    title: "Backpropagation",
    lesson: "Lesson 07",
    goal: "Find how every weight affected the loss, working backwards.",
    from: () => "1 loss",
    to: () => `${NETWORK_NUMBER_COUNT} updated numbers`,
    status: (session) => jobStatus(session.isRunning("network"), session.isNetworkDone),
    actions: (session) => (session.isNetworkDone || session.isRunning("network") ? [] : [{ label: "Train the network", command: "start:network", isStrong: true }]),
    rowsTitle: "One backward pass, x = 7, real numbers",
    rows: (session) => {
      const round = networkExplain(session);
      return [
        { text: "1. Forward: save h1, h2, h3, out", value: `out = ${round.out.toFixed(3)}` },
        { text: "2. Error = target − out", value: `${round.target.toFixed(3)} − ${round.out.toFixed(3)} = ${round.error.toFixed(3)}` },
        { text: "3. δ_out = error × out × (1 − out)", value: round.deltaOut.toFixed(4) },
        { text: "4. Change to wO1 = lr × δ_out × h1", value: round.outputChange.toFixed(4) },
        { text: "5. Back to hidden: δ_h1 = δ_out × wO1 × h1(1 − h1)", value: round.deltaHidden.toFixed(5) },
      ];
    },
    visual: ({ network }) => salaryPlot({ model: (x) => network.predict(x) }),
    note: ({ network, bestLine }) => {
      const compare = `Round ${count(network.rounds)} of ${count(NETWORK_TOTAL_ROUNDS)}. Loss ${network.loss().toFixed(1)}. The best straight line has ${lineLoss(bestLine).toFixed(1)}.`;
      return network.rounds > 0 && network.rounds <= LESSON_ROUNDS ? `${compare} Lesson 07 stops at 50,000 rounds, where it only ties the line.` : compare;
    },
    formula: "dL/d(out) = −2 × (y − out)\nd(out)/dz = out × (1 − out)\ndL/dwO1 = δ_out × h1",
    code: "const error = yi - out;\nconst dOut = error * sigmoidDerivative(out);\nwO1 += lr * dOut * h1;\nconst dH1 = dOut * wO1 * sigmoidDerivative(h1);\nw1 += lr * dH1 * xi;",
    why: "This is the chain rule, applied from the output back to the input. It is how every neural network learns.",
  },
  {
    id: "matrix",
    stage: "simple",
    icon: "grid",
    title: "Matrix math",
    lesson: "Lesson 08",
    goal: "Do thousands of sums at once.",
    from: () => "Loops over numbers",
    to: () => "One table multiply",
    status: () => "done",
    visual: () => `<div class="grid-sum">
      <table><tr><td>1</td><td>2</td><td>3</td></tr><tr><td>4</td><td>5</td><td>6</td></tr></table> ×
      <table><tr><td>7</td><td>8</td></tr><tr><td>9</td><td>10</td></tr><tr><td>11</td><td>12</td></tr></table> =
      <table><tr><td>58</td><td>64</td></tr><tr><td>139</td><td>154</td></tr></table>
    </div>`,
    rowsTitle: "How each result is made",
    rows: () => [
      { text: "Top left: 1×7 + 2×9 + 3×11", value: "58" },
      { text: "Top right: 1×8 + 2×10 + 3×12", value: "64" },
      { text: "Bottom left: 4×7 + 5×9 + 6×11", value: "139" },
      { text: "Bottom right: 4×8 + 5×10 + 6×12", value: "154" },
      { text: "Shape rule", value: "(2×3) × (3×2) = (2×2)" },
      { text: "Dot product: [1,2,3] · [4,5,6]", value: "32" },
    ],
    formula: "One layer: output = activation(X @ W + b)",
    code: "for (let i = 0; i < rows; i++)\n  for (let j = 0; j < cols; j++)\n    for (let k = 0; k < inner; k++)\n      C[i][j] += A[i][k] * B[k][j];",
    why: "Every layer of a big model is this: multiply tables, add, then a curve.",
  },
];

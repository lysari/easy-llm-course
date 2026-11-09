import type { ModelPart, TinyPartId } from "@/features/take-apart/interfaces/model-part.interface";
import { count } from "@/shared/utils/format";

interface PartText {
  id: string;
  icon: string;
  title: string;
  summary: string;
  copies: number;
  lesson: string;
  reverseLesson: string;
  explanation: string;
  why: string;
  missing?: string;
}

interface ProjectionSpec extends PartText {
  inSize: number;
  outSize: number;
  hasBias: boolean;
  input: string;
  output: string;
  inputDetail?: string;
  outputDetail?: string;
  tinyPart: TinyPartId;
}

interface TableSpec extends PartText {
  rows: number;
  cols: number;
  rowName: string;
  tinyPart: TinyPartId;
}

interface NormSpec extends PartText {
  size: number;
  hasShift: boolean;
  tinyPart: TinyPartId;
}

const NO_BIAS = "Our tiny model has no biases. Its numbers come from the multiply-and-adds alone.";
const NO_SCALE = "Our tiny model's norm has no stored scale or shift numbers. It only centres and rescales.";

export function projectionPart(spec: ProjectionSpec): ModelPart {
  const { id, inSize, outSize, hasBias, tinyPart, missing } = spec;
  const sizeIn = count(inSize);
  const sizeOut = count(outSize);
  const multiplyAdd: ModelPart = {
    id: `${id}-multiply-add`,
    icon: "plus",
    title: "One multiply-and-add",
    summary: "Multiplies one input number by one stored number and adds it to a running sum.",
    copies: inSize,
    facts: { shape: "1 × 1", weightCount: 1, headline: { value: inSize, unit: "per output number" }, input: "1 input, 1 stored", output: "the running sum", inputDetail: "1 input number and 1 stored number", outputDetail: "the running sum, a little bigger or smaller" },
    tiny: { part: tinyPart, level: "multiply-add", missing },
    lesson: "Lessons 00, 01",
    reverseLesson: "RL-06",
    explanation: `Ground zero. One input number times one stored number, added to a running sum. This happens ${sizeIn} times to make one output number.`,
    why: "There is nothing smaller inside. The whole part is this one line, repeated. No step here knows what a word means.",
    formula: { lines: ["sum = sum + x × w"], legend: [{ term: "x", meaning: "one input number" }, { term: "w", meaning: "one stored number, set by training" }] },
    code: "sum = sum + x * w;",
    groundZero: "multiply-add",
  };
  const bias: ModelPart = {
    id: `${id}-bias`,
    icon: "plus",
    title: "Bias",
    summary: "Adds one stored number at the end, so the output can move up or down.",
    copies: 1,
    facts: { shape: "1", weightCount: 1, headline: { value: 1, unit: "number" }, input: "the finished sum", output: "sum plus 1 number", outputDetail: "the sum plus one stored number" },
    tiny: { part: tinyPart, level: "multiply-add", missing: NO_BIAS },
    lesson: "Lesson 01",
    reverseLesson: "RL-06",
    explanation: "Ground zero. One stored number added at the end, the b in y = w × x + b. It lets the output move up or down even when every input is zero.",
    why: "It is a single learned number, nothing more. Training nudges it like every other number.",
    formula: { lines: ["out = sum + b"] },
    code: "const out = sum + b;",
    groundZero: "stored-number",
  };
  const number: ModelPart = {
    id: `${id}-number`,
    icon: "hash",
    title: "One output number",
    summary: "Makes one output number: each input times its own stored number, all added up.",
    copies: outSize,
    facts: { shape: `${sizeIn} × 1`, weightCount: inSize + (hasBias ? 1 : 0), headline: { value: inSize + (hasBias ? 1 : 0), unit: "numbers" }, input: `${sizeIn} numbers`, output: "1 number" },
    tiny: { part: tinyPart, level: "number", missing },
    lesson: "Lesson 08",
    reverseLesson: "RL-06",
    explanation: `One of the ${sizeOut} numbers that come out. It is a dot product: each of the ${sizeIn} input numbers times its own stored number, all added up${hasBias ? ", plus one bias" : ""}.`,
    why: "A matrix multiply is just many of these dot products side by side. Each one only multiplies and adds.",
    formula: { lines: [`out = x1 × w1 + x2 × w2 + … + x${inSize} × w${inSize}${hasBias ? " + b" : ""}`] },
    code: `let sum = ${hasBias ? "b" : "0"};\nfor (let i = 0; i < ${inSize}; i++) {\n  sum = sum + x[i] * w[i];\n}`,
    children: hasBias ? [multiplyAdd, bias] : [multiplyAdd],
  };
  return {
    id,
    icon: spec.icon,
    title: spec.title,
    summary: spec.summary,
    copies: spec.copies,
    facts: { shape: `${sizeIn} × ${sizeOut}`, weightCount: inSize * outSize + (hasBias ? outSize : 0), headline: { value: inSize * outSize + (hasBias ? outSize : 0), unit: "numbers" }, input: spec.input, output: spec.output, inputDetail: spec.inputDetail, outputDetail: spec.outputDetail },
    tiny: { part: tinyPart, level: "whole", missing },
    lesson: spec.lesson,
    reverseLesson: spec.reverseLesson,
    explanation: spec.explanation,
    why: spec.why,
    formula: { lines: [hasBias ? "out = x × W + b" : "out = x × W"], legend: [{ term: "W", meaning: `${sizeIn} × ${sizeOut} stored numbers` }] },
    code: `const out: number[] = [];\nfor (let j = 0; j < ${outSize}; j++) {\n  out.push(dot(x, column(W, j))${hasBias ? " + b[j]" : ""});\n}`,
    children: [number],
  };
}

export function tablePart(spec: TableSpec): ModelPart {
  const { id, rows, cols, rowName, tinyPart, missing } = spec;
  const storedNumber: ModelPart = {
    id: `${id}-number`,
    icon: "hash",
    title: "One stored number",
    summary: "One number in the table, set by training, that means nothing on its own.",
    copies: cols,
    facts: { shape: "1", weightCount: 1, headline: { value: rows * cols, unit: "in the table" }, input: "nothing, only read", output: "1 number", inputDetail: "nothing, it is only read" },
    tiny: { part: tinyPart, level: "number", missing },
    lesson: "Lessons 03, 10",
    reverseLesson: "RL-06",
    explanation: "Ground zero. One stored number in the table. It started random, and training nudged it a tiny bit, round after round.",
    why: "There is no word inside, only this number. Meaning comes from where many such numbers sit next to each other.",
    formula: { lines: ["w = w − step size × slope"] },
    code: "const w = table[row][i];",
    groundZero: "stored-number",
  };
  const row: ModelPart = {
    id: `${id}-row`,
    icon: "grid",
    title: `One ${rowName}'s row`,
    summary: `The list of numbers for one ${rowName}, copied out each time that ${rowName} comes up.`,
    copies: rows,
    facts: { shape: `1 × ${count(cols)}`, weightCount: cols, headline: { value: cols, unit: "numbers" }, input: `the ${rowName}'s id`, output: `${count(cols)} numbers` },
    tiny: { part: tinyPart, level: "row", missing },
    lesson: "Lesson 10",
    reverseLesson: "RL-02",
    explanation: `The ${count(cols)} numbers for one ${rowName}. Looking it up is just copying this row out of the table.`,
    why: "The row is a point in a space with many directions. Rows that are used in similar places end up close together.",
    formula: { lines: [`row = table[id], ${count(cols)} numbers`] },
    code: "const row: number[] = table[id];",
    children: [storedNumber],
  };
  return {
    id,
    icon: spec.icon,
    title: spec.title,
    summary: spec.summary,
    copies: spec.copies,
    facts: { shape: `${count(rows)} × ${count(cols)}`, weightCount: rows * cols, headline: { value: rows * cols, unit: "numbers" }, input: `a ${rowName} id`, output: `${count(cols)} numbers` },
    tiny: { part: tinyPart, level: "whole", missing },
    lesson: spec.lesson,
    reverseLesson: spec.reverseLesson,
    explanation: spec.explanation,
    why: spec.why,
    formula: { lines: [`table shape: ${count(rows)} × ${count(cols)}`] },
    code: "const vectors = ids.map((id) => table[id]);",
    children: [row],
  };
}

export function normPart(spec: NormSpec): ModelPart {
  const { id, size, hasShift, tinyPart } = spec;
  const number: ModelPart = {
    id: `${id}-number`,
    icon: "hash",
    title: "One output number",
    summary: hasShift ? "Scales one number by its own stored number, then shifts it by another." : "Scales one number by its own stored number.",
    copies: size,
    facts: { shape: "1", weightCount: hasShift ? 2 : 1, headline: hasShift ? { value: 2, unit: "numbers" } : { value: 1, unit: "number" }, input: "1 rescaled number", output: "1 number" },
    tiny: { part: tinyPart, level: "number", missing: NO_SCALE },
    lesson: "Lessons 13, 20",
    reverseLesson: "RL-06",
    explanation: hasShift
      ? "Ground zero. One rescaled number times its own scale number, plus its own shift number. One multiply-and-add with 2 stored numbers."
      : "Ground zero. One rescaled number times its own scale number. A single multiply with 1 stored number.",
    why: "Training picks the scale (and shift) so each of the numbers can grow back to the size that helps the next part.",
    formula: { lines: [hasShift ? "y = x̂ × g + b" : "y = x̂ × g"] },
    code: hasShift ? "const y = xHat * g + b;" : "const y = xHat * g;",
    groundZero: "multiply-add",
  };
  return {
    id,
    icon: spec.icon,
    title: spec.title,
    summary: spec.summary,
    copies: spec.copies,
    facts: { shape: hasShift ? `2 × ${count(size)}` : `1 × ${count(size)}`, weightCount: size * (hasShift ? 2 : 1), headline: { value: size * (hasShift ? 2 : 1), unit: "numbers" }, input: `${count(size)} numbers`, output: `${count(size)} steady numbers`, outputDetail: `${count(size)} numbers, in a steady range` },
    tiny: { part: tinyPart, level: "whole", missing: spec.missing },
    lesson: spec.lesson,
    reverseLesson: spec.reverseLesson,
    explanation: spec.explanation,
    why: spec.why,
    formula: hasShift
      ? { lines: ["x̂ = (x − mean) / √(variance + ε)", "y = x̂ × g + b"] }
      : { lines: ["x̂ = x / √(mean(x²) + ε)", "y = x̂ × g"] },
    code: hasShift
      ? "const xHat = x.map((v) => (v - mean) / spread);\nconst y = xHat.map((v, i) => v * g[i] + b[i]);"
      : "const xHat = x.map((v) => v / rms);\nconst y = xHat.map((v, i) => v * g[i]);",
    children: [number],
  };
}

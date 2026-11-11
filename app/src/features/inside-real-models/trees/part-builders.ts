import type { ModelPart, PartStory, TinyPartId } from "@/features/inside-real-models/interfaces/model-part.interface";
import type { RealFacts, RealNote, StepFacts } from "@/features/inside-real-models/interfaces/real-facts.interface";
import type { ConceptName } from "@/shared/interfaces/concept.interface";
import { count } from "@/shared/utils/format";
import type { SourceId } from "./sources";

interface PartText {
  id: string;
  icon: ConceptName;
  title: string;
  summary: string;
  copies: number;
  explanation: string;
  why: string;
  sources: SourceId[];
  story: PartStory;
  reasons: RealNote[];
  codeReasons: RealNote[];
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
  missing?: string;
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

interface InOutDetail {
  inputDetail?: string;
  outputDetail?: string;
}

export const TOKEN_TERM = "We call each piece a \"word\" here. The real name is token: a whole word, part of a word, or a symbol.";

export const stepFacts = (facts: StepFacts, input: string, output: string, detail: InOutDetail = {}): RealFacts => ({
  shape: facts.shape,
  weightCount: 0,
  headline: facts.headline,
  input,
  output,
  ...detail,
  sources: facts.sources,
  notes: facts.notes,
  reasons: facts.reasons,
});

export function projectionPart(spec: ProjectionSpec): ModelPart {
  const { id, inSize, outSize, hasBias, tinyPart, missing, sources, codeReasons } = spec;
  const sizeIn = count(inSize);
  const sizeOut = count(outSize);
  const multiplyAdd: ModelPart = {
    id: `${id}-multiply-add`,
    icon: "multiply-add",
    title: "One multiply-and-add",
    summary: "Multiplies one input number by one stored number and adds it to a running sum.",
    copies: inSize,
    facts: { shape: "1 × 1", weightCount: 1, headline: { value: inSize, unit: "per output number", isComputed: true }, input: "1 input, 1 stored", output: "the running sum", inputDetail: "1 input number and 1 stored number", outputDetail: "the running sum, a little bigger or smaller", sources, reasons: codeReasons },
    tiny: { part: tinyPart, level: "multiply-add", missing },
    explanation: `One input number times one stored number, added to a running sum. This happens ${sizeIn} times to make one output number.`,
    why: "There is nothing smaller inside. The whole part is this one line, repeated. No step here knows what a word means.",
    story: {
      problem: `This is the bottom of the ${spec.title.toLowerCase()}. A table of numbers can't do anything by itself; this one multiply and one add is how it touches the word's numbers. Without it nothing would come out at all.`,
      demo: "tiny-part",
      when: `Nobody writes it by hand. It runs ${sizeIn} times for every output number, every time the model reads a word, and chips are built to do it fast.`,
      feeds: hasBias ? "After the last one, the running sum gets the bias added, and becomes one output number." : "After the last one, the running sum is one finished output number.",
      nextId: hasBias ? `${id}-bias` : undefined,
    },
    formula: { lines: ["sum = sum + x × w"], legend: [{ term: "x", meaning: "one input number" }, { term: "w", meaning: "one stored number, set by training" }] },
    code: "sum = sum + x * w;",
    groundZero: "multiply-add",
  };
  const bias: ModelPart = {
    id: `${id}-bias`,
    icon: "bias",
    title: "Bias",
    summary: "Adds one stored number at the end, so the output can move up or down.",
    copies: 1,
    facts: { shape: "1", weightCount: 1, headline: { value: 1, unit: "number", isComputed: true }, input: "the finished sum", output: "sum plus 1 number", outputDetail: "the sum plus one stored number", sources, reasons: codeReasons },
    tiny: { part: tinyPart, level: "multiply-add" },
    explanation: "One stored number added at the end, the b in y = w × x + b. It lets the output move up or down even when every input is zero.",
    why: "It is a single learned number, nothing more. Training nudges it like every other number.",
    story: {
      problem: "Without a bias, a word whose numbers were all 0 could only ever give 0. The bias lets each output number start from its own level, up or down.",
      demo: "tiny-part",
      when: "You choose whether a table has biases when you write the architecture. GPT-2 has them; Llama 3 leaves them out of every table.",
      feeds: "The sum plus the bias is the finished output number, which goes on with the others.",
    },
    formula: { lines: ["out = sum + b"] },
    code: "const out = sum + b;",
    groundZero: "stored-number",
  };
  const number: ModelPart = {
    id: `${id}-number`,
    icon: "number",
    title: "One output number",
    summary: "Makes one output number: each input times its own stored number, all added up.",
    copies: outSize,
    facts: { shape: `${sizeIn} × 1`, weightCount: inSize + (hasBias ? 1 : 0), headline: { value: inSize + (hasBias ? 1 : 0), unit: "numbers", isComputed: true }, input: `${sizeIn} numbers`, output: "1 number", sources, reasons: codeReasons },
    tiny: { part: tinyPart, level: "number", missing },
    explanation: `One of the ${sizeOut} numbers that come out. It is a dot product: each of the ${sizeIn} input numbers times its own stored number, all added up${hasBias ? ", plus one bias" : ""}.`,
    why: "A matrix multiply is just many of these dot products side by side. Each one only multiplies and adds.",
    story: {
      problem: `The whole table is too big to follow at once. One output number is the smallest piece that still does its job: it measures how well the word's ${sizeIn} numbers match one column of stored numbers. Without these, the part would hand nothing on.`,
      demo: "tiny-part",
      when: `It happens each time the ${spec.title.toLowerCase()} runs, for every word. You only choose how many come out (${sizeOut}) when you set the model's sizes.`,
      feeds: `Side by side, the ${sizeOut} output numbers are the ${spec.title.toLowerCase()}'s result.`,
      nextId: `${id}-multiply-add`,
    },
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
    facts: { shape: `${sizeIn} × ${sizeOut}`, weightCount: inSize * outSize + (hasBias ? outSize : 0), headline: { value: inSize * outSize + (hasBias ? outSize : 0), unit: "numbers", isComputed: true }, input: spec.input, output: spec.output, inputDetail: spec.inputDetail, outputDetail: spec.outputDetail, sources, reasons: spec.reasons },
    tiny: { part: tinyPart, level: "whole", missing },
    explanation: spec.explanation,
    why: spec.why,
    story: spec.story,
    formula: { lines: [hasBias ? "out = x × W + b" : "out = x × W"], legend: [{ term: "W", meaning: `${sizeIn} × ${sizeOut} stored numbers` }] },
    code: `const out: number[] = [];\nfor (let j = 0; j < ${outSize}; j++) {\n  out.push(dot(x, column(W, j))${hasBias ? " + b[j]" : ""});\n}`,
    children: [number],
  };
}

export function tablePart(spec: TableSpec): ModelPart {
  const { id, rows, cols, rowName, tinyPart, sources, codeReasons } = spec;
  const storedNumber: ModelPart = {
    id: `${id}-number`,
    icon: "number",
    title: "One stored number",
    summary: "One number in the table, set by training, that means nothing on its own.",
    copies: cols,
    facts: { shape: "1", weightCount: 1, headline: { value: rows * cols, unit: "in the table", isComputed: true }, input: "nothing, only read", output: "1 number", inputDetail: "nothing, it is only read", sources, reasons: codeReasons },
    tiny: { part: tinyPart, level: "number" },
    explanation: "One stored number in the table. It started random, and training nudged it a tiny bit, round after round.",
    why: "There is no word inside, only this number. Meaning comes from where many such numbers sit next to each other.",
    story: {
      problem: "Each stored number is one dial that training can turn. Without dials there would be nothing to learn: the model's whole knowledge is the settings of numbers like this one.",
      demo: "tiny-part",
      when: "It is set at random when the model is first written, then nudged a little every training round. After training it never changes again.",
      feeds: `It is read with the rest of its row whenever that ${rowName} comes up.`,
    },
    formula: { lines: ["w = w − step size × slope"] },
    code: "const w = table[row][i];",
    groundZero: "stored-number",
  };
  const row: ModelPart = {
    id: `${id}-row`,
    icon: "vector",
    title: `One ${rowName}'s row`,
    summary: `The list of numbers for one ${rowName}, copied out each time that ${rowName} comes up.`,
    copies: rows,
    facts: { shape: `1 × ${count(cols)}`, weightCount: cols, headline: { value: cols, unit: "numbers", isComputed: true }, input: `the ${rowName}'s id`, output: `${count(cols)} numbers`, sources, reasons: codeReasons },
    tiny: { part: tinyPart, level: "row" },
    explanation: `The ${count(cols)} numbers for one ${rowName}. Looking it up is just copying this row out of the table.`,
    why: "The row is a point in a space with many directions. Training moves it; the row holds no word, only numbers.",
    story: {
      problem: `A ${rowName} number on its own can't be multiplied into anything useful: ${rowName === "word" ? "ID 5 is not \"more\" than ID 4" : "place 5 is not just place 4 plus one"}. Its row turns that one number into ${count(cols)} numbers the layers can work with.`,
      demo: "tiny-part",
      when: `Every time the model reads text, once for each ${rowName}. Looking it up is only copying the row out.`,
      feeds: "The copied row is what the first layer reads.",
      nextId: `${id}-number`,
    },
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
    facts: { shape: `${count(rows)} × ${count(cols)}`, weightCount: rows * cols, headline: { value: rows * cols, unit: "numbers", isComputed: true }, input: `a ${rowName} id`, output: `${count(cols)} numbers`, sources, reasons: spec.reasons },
    tiny: { part: tinyPart, level: "whole" },
    explanation: spec.explanation,
    why: spec.why,
    story: spec.story,
    formula: { lines: [`table shape: ${count(rows)} × ${count(cols)}`] },
    code: "const vectors = ids.map((id) => table[id]);",
    children: [row],
  };
}

export function normPart(spec: NormSpec): ModelPart {
  const { id, size, hasShift, tinyPart, sources, codeReasons } = spec;
  const number: ModelPart = {
    id: `${id}-number`,
    icon: "number",
    title: "One output number",
    summary: hasShift ? "Scales one number by its own stored number, then shifts it by another." : "Scales one number by its own stored number.",
    copies: size,
    facts: { shape: "1", weightCount: hasShift ? 2 : 1, headline: hasShift ? { value: 2, unit: "numbers", isComputed: true } : { value: 1, unit: "number", isComputed: true }, input: "1 rescaled number", output: "1 number", sources, reasons: codeReasons },
    tiny: { part: tinyPart, level: "number" },
    explanation: hasShift
      ? "One rescaled number times its own scale number, plus its own shift number. One multiply-and-add with 2 stored numbers."
      : "One rescaled number times its own scale number. A single multiply with 1 stored number.",
    why: "Training picks the scale (and shift) so each of the numbers can grow back to the size that helps the next part.",
    story: {
      problem: "Rescaling makes every number the same size, which can throw away useful differences. Each number's own stored scale lets training give some of that size back where it helps.",
      demo: "tiny-part",
      when: `It runs on every one of the ${count(size)} numbers, for every word, each time the norm runs.`,
      feeds: "The scaled number goes into the next part with the others.",
    },
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
    facts: { shape: hasShift ? `2 × ${count(size)}` : `1 × ${count(size)}`, weightCount: size * (hasShift ? 2 : 1), headline: { value: size * (hasShift ? 2 : 1), unit: "numbers", isComputed: true }, input: `${count(size)} numbers`, output: `${count(size)} steady numbers`, outputDetail: `${count(size)} numbers, in a steady range`, sources, reasons: spec.reasons },
    tiny: { part: tinyPart, level: "whole" },
    explanation: spec.explanation,
    why: spec.why,
    story: spec.story,
    formula: hasShift
      ? { lines: ["x̂ = (x − mean) / √(variance + ε)", "y = x̂ × g + b"] }
      : { lines: ["x̂ = x / √(mean(x²) + ε)", "y = x̂ × g"] },
    code: hasShift
      ? "const xHat = x.map((v) => (v - mean) / spread);\nconst y = xHat.map((v, i) => v * g[i] + b[i]);"
      : "const xHat = x.map((v) => v / rms);\nconst y = xHat.map((v, i) => v * g[i]);",
    children: [number],
  };
}

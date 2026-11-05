import type { ForwardTrace, RankedToken, TinyGptWeights } from "../interfaces/model.interface";
import { END_TOKEN, MAX_CONTEXT, TRAINING_SEQUENCES, VOCABULARY, joinTokens } from "../model/corpus";
import { nextOptions, pickToken, writeText } from "../model/generate";
import { ModelLab } from "../model/lab";
import { LINE_TOTAL_STEPS, type Line, stepLine, trainedLine } from "../model/line-model";
import { NETWORK_TOTAL_ROUNDS, SmallNetwork } from "../model/small-network";
import { seededRandom } from "../model/random";
import { LEARNING_RATE, MOMENTUM_DECAY, VARIANCE_DECAY } from "../model/adam";
import { allTables, copyWeights, lossAndGradients, measureLoss, predictNext, traceForward } from "../model/tiny-gpt";

export const TEST_WORDS = "my dog is hungry , so i give it a".split(" ");
export const TEST_ANSWER = "bone";
export const START_CHOICES = [
  "my dog is hungry , so i give it a",
  "my cat is hungry , so i give it a",
  "my dog is bored , so i give it a",
  "my cat is bored , so i give it a",
  "the cat likes the",
];

export type JobName = "line" | "network" | "train" | "write";

const JOB_RATES: Record<JobName, number> = { line: 2000, network: 80000, train: 20, write: 2.5 };
const LESSON_LINES: Record<string, Line> = { "10,30": { w: 10, b: 30 }, "12,26": { w: 12, b: 26 }, "13,25": { w: 13, b: 25 } };
const QUICK_FINISH_RATE = 400;
const SAMPLE_SEED = 11;
const PICK_MANY = 20;
const NUDGE = 0.01;

export class Session {
  lineChoice = "10,30";
  curveChoice = "sigmoid";
  temperature = 1;
  topK = 0;
  topP = 1;
  readonly bestLine = trainedLine();
  learningLine: Line = { w: 0, b: 0 };
  lineSteps = 0;
  network = new SmallNetwork();

  lab = new ModelLab();
  trainSeconds = 0;
  trainRate = JOB_RATES.train;
  readonly testIds = TEST_WORDS.map((word) => VOCABULARY.indexOf(word));

  chosenStart: string[] | null = null;
  picks: string[] = [];
  written: string[] = [];
  isWritingDone = false;
  private writtenDetails: { word: string; top: string; topChance: number }[] = [];

  readonly startLoss = measureLoss(TRAINING_SEQUENCES, this.lab.untrained);

  private firstGradientWeights: TinyGptWeights | null = null;
  private readonly running = new Set<JobName>();
  private readonly owed: Record<JobName, number> = { line: 0, network: 0, train: 0, write: 0 };

  get hasRunningJobs() {
    return this.running.size > 0;
  }

  isRunning(job: JobName) {
    return this.running.has(job);
  }

  get chosenLine() {
    return LESSON_LINES[this.lineChoice];
  }

  get isLineDone() {
    return this.lineSteps >= LINE_TOTAL_STEPS;
  }

  get isNetworkDone() {
    return this.network.rounds >= NETWORK_TOTAL_ROUNDS;
  }

  get startWords() {
    return this.chosenStart ?? START_CHOICES[0].split(" ");
  }

  runCommand(command: string) {
    const [name, value = ""] = command.split(":");
    if (name === "start") this.start(value as JobName);
    if (name === "line") this.lineChoice = value;
    if (name === "finish") this.trainRate = QUICK_FINISH_RATE;
    if (name === "forget") this.forgetTraining();
    if (name === "choose") this.chooseStart(value.split(" "));
    if (name === "curve") this.curveChoice = value;
    if (name === "temperature") this.setSampling(() => (this.temperature = Number(value)));
    if (name === "topk") this.setSampling(() => (this.topK = Number(value)));
    if (name === "topp") this.setSampling(() => (this.topP = Number(value)));
    if (name === "pick-many") for (let i = 0; i < PICK_MANY; i++) this.pickWord(Math.random);
    if (name === "pick-clear") this.picks = [];
  }

  tick(seconds: number) {
    for (const job of this.running) {
      const rate = job === "train" ? this.trainRate : JOB_RATES[job];
      this.owed[job] += seconds * rate;
      const count = Math.floor(this.owed[job]);
      this.owed[job] -= count;
      this.advance(job, count, seconds);
    }
  }

  guess(weights: TinyGptWeights = this.lab.live): RankedToken[] {
    return predictNext(this.testIds, weights);
  }

  trace(weights: TinyGptWeights = this.lab.live): ForwardTrace {
    return traceForward(this.testIds, weights);
  }

  options() {
    return nextOptions(this.startWords, this.lab.live);
  }

  sampleWriting() {
    return joinTokens(writeText(["my", "dog"], this.lab.live, seededRandom(SAMPLE_SEED)));
  }

  averageMove() {
    const before = allTables(this.lab.untrained);
    const after = allTables(this.lab.live);
    let total = 0;
    let count = 0;
    after.forEach(({ tensor }, t) => {
      tensor.data.forEach((value, i) => {
        total += Math.abs(value - before[t].tensor.data[i]);
        count++;
      });
    });
    return total / count;
  }

  sampledOptions(): RankedToken[] {
    const options = this.options();
    const scaled = options.map((option) => ({ ...option, raw: Math.exp((option.score - options[0].score) / this.temperature) }));
    let kept = this.topK > 0 ? scaled.slice(0, this.topK) : scaled;
    const total = kept.reduce((sum, option) => sum + option.raw, 0);
    let withChance = kept.map((option) => ({ ...option, chance: option.raw / total }));
    if (this.topP < 1) {
      let running = 0;
      kept = withChance.filter((option) => {
        const keep = running < this.topP;
        running += option.chance;
        return keep;
      });
      const nucleus = kept.reduce((sum, option) => sum + (option as RankedToken).chance, 0);
      withChance = (kept as typeof withChance).map((option) => ({ ...option, chance: option.chance / nucleus }));
    }
    return withChance.map(({ token, id, score, chance }) => ({ token, id, score, chance }));
  }

  pickCounts(): [string, number][] {
    const counts = new Map<string, number>();
    for (const pick of this.picks) counts.set(pick, (counts.get(pick) ?? 0) + 1);
    return [...counts].sort((a, b) => b[1] - a[1]);
  }

  writtenSteps() {
    return this.writtenDetails;
  }

  answerChanceNow() {
    return this.guess().find((option) => option.token === TEST_ANSWER)?.chance ?? 0;
  }

  similarity(wordA: string, wordB: string, weights: TinyGptWeights) {
    const rows = weights.tokenTable;
    const a = rows.row(VOCABULARY.indexOf(wordA));
    const b = rows.row(VOCABULARY.indexOf(wordB));
    const dot = a.reduce((sum, value, i) => sum + value * b[i], 0);
    const size = (v: number[]) => Math.sqrt(v.reduce((sum, value) => sum + value * value, 0));
    return dot / (size(a) * size(b));
  }

  gradientTables() {
    return allTables(this.firstGradients()).map(({ name, tensor }) => ({
      name,
      size: tensor.grad.reduce((sum, g) => sum + Math.abs(g), 0) / tensor.grad.length,
    }));
  }

  nudgeTest() {
    const weights = this.firstGradients();
    const table = weights.outputTable;
    const column = VOCABULARY.indexOf(TEST_ANSWER);
    let index = column;
    for (let r = 0; r < table.rows; r++) {
      const candidate = r * table.cols + column;
      if (Math.abs(table.grad[candidate]) > Math.abs(table.grad[index])) index = candidate;
    }
    const before = measureLoss(TRAINING_SEQUENCES, weights);
    const trial = copyWeights(weights);
    trial.outputTable.data[index] += NUDGE;
    return { gradient: table.grad[index], predicted: table.grad[index] * NUDGE, measured: measureLoss(TRAINING_SEQUENCES, trial) - before };
  }

  adamFirstStep() {
    const weights = this.firstGradients();
    const table = weights.outputTable;
    const column = VOCABULARY.indexOf(TEST_ANSWER);
    let index = column;
    for (let r = 0; r < table.rows; r++) {
      const candidate = r * table.cols + column;
      if (Math.abs(table.grad[candidate]) > Math.abs(table.grad[index])) index = candidate;
    }
    const gradient = table.grad[index];
    const m = (1 - MOMENTUM_DECAY) * gradient;
    const v = (1 - VARIANCE_DECAY) * gradient * gradient;
    const mHat = m / (1 - MOMENTUM_DECAY);
    const vHat = v / (1 - VARIANCE_DECAY);
    const change = (LEARNING_RATE * mHat) / Math.sqrt(vHat);
    return { value: table.data[index], gradient, m, v, mHat, vHat, change, after: table.data[index] - change };
  }

  pickWord(random: () => number) {
    this.picks.push(pickToken(this.sampledOptions(), random).token);
  }

  writeNextWord(random: () => number) {
    if (this.isWritingDone) return;
    const words = [...this.startWords, ...this.written];
    const options = nextOptions(words, this.lab.live);
    const picked = pickToken(options, random).token;
    this.written.push(picked);
    this.writtenDetails.push({ word: picked, top: options[0].token, topChance: options[0].chance });
    this.isWritingDone = picked === END_TOKEN || words.length + 1 >= MAX_CONTEXT;
  }

  private start(job: JobName) {
    if (job === "line" && this.isLineDone) return;
    if (job === "network" && this.isNetworkDone) return;
    if (job === "train" && this.lab.isTrained) return;
    if (job === "write") {
      if (!this.lab.isTrained) return;
      this.written = [];
      this.writtenDetails = [];
      this.isWritingDone = false;
    }
    this.owed[job] = 0;
    this.running.add(job);
  }

  private advance(job: JobName, count: number, seconds: number) {
    if (job === "line") {
      for (let i = 0; i < count && !this.isLineDone; i++) {
        stepLine(this.learningLine);
        this.lineSteps++;
      }
      if (this.isLineDone) this.running.delete(job);
    }
    if (job === "network") {
      this.network.trainRounds(Math.min(count, NETWORK_TOTAL_ROUNDS - this.network.rounds));
      if (this.isNetworkDone) this.running.delete(job);
    }
    if (job === "train") {
      this.trainSeconds += seconds;
      this.lab.trainSteps(count);
      if (this.lab.isTrained) this.running.delete(job);
    }
    if (job === "write") {
      for (let i = 0; i < count; i++) this.writeNextWord(Math.random);
      if (this.isWritingDone) this.running.delete(job);
    }
  }

  private forgetTraining() {
    this.running.delete("train");
    this.running.delete("write");
    this.lab = new ModelLab();
    this.trainSeconds = 0;
    this.trainRate = JOB_RATES.train;
    this.chosenStart = null;
    this.picks = [];
    this.written = [];
    this.writtenDetails = [];
    this.isWritingDone = false;
  }

  private setSampling(change: () => void) {
    change();
    this.picks = [];
  }

  private firstGradients() {
    if (!this.firstGradientWeights) {
      this.firstGradientWeights = copyWeights(this.lab.untrained);
      lossAndGradients(TRAINING_SEQUENCES, this.firstGradientWeights);
    }
    return this.firstGradientWeights;
  }

  private chooseStart(start: string[]) {
    this.chosenStart = start;
    this.picks = [];
    this.written = [];
    this.writtenDetails = [];
    this.isWritingDone = false;
    this.running.delete("write");
  }
}

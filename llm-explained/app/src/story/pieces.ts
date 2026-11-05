import type { LossPoint } from "../interfaces/model.interface";
import { END_TOKEN } from "../model/corpus";
import { SALARIES, YEARS } from "../model/line-model";

const PLOT = { width: 640, height: 300, left: 54, right: 16, top: 12, bottom: 46 };

/** Escapes text for HTML. Use it for every word placed in markup. */
export const escapeHtml = (text: string) =>
  text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/** Shows the stop word in plain language and leaves all other words as they are. */
export const showWord = (word: string) => (word === END_TOKEN ? "stop" : word);

/** Formats a 0-1 chance as a short percent, for example 74%. */
export const percent = (chance: number) => (chance < 0.01 ? "<1%" : `${Math.round(chance * 100)}%`);

/** Writes a count with thousands separators. */
export const count = (value: number) => value.toLocaleString("en-US");

/** Draws one card per word. Pass `numbers` to show a value under each and `newFrom` to mark later words as new. */
export function wordCards(words: string[], options: { numbers?: (number | string)[]; newFrom?: number } = {}) {
  const cards = words.map((word, i) => {
    const isNew = options.newFrom !== undefined && i >= options.newFrom;
    const number = options.numbers ? `<small>${options.numbers[i]}</small>` : "";
    return `<span class="word${isNew ? " is-new" : ""}">${escapeHtml(showWord(word))}${number}</span>`;
  });
  return `<div class="words">${cards.join("")}</div>`;
}

/** Draws guesses as bars. `answer` is the word marked strongly; it is always shown, even outside the top `limit`. */
export function chanceBars(options: { token: string; chance: number }[], answer: string, limit = 4) {
  const top = options.slice(0, limit);
  const answerOption = options.find((option) => option.token === answer);
  const shown = answerOption && !top.includes(answerOption) ? [...top, answerOption] : top;
  const rows = shown.map((option) => {
    const width = Math.max(1, Math.round(option.chance * 100));
    return `<div class="chance${option.token === answer ? " is-key" : ""}">
      <span>${escapeHtml(showWord(option.token))}</span>
      <span class="chance-track"><i class="chance-fill" style="width:${width}%"></i></span>
      <span>${percent(option.chance)}</span>
    </div>`;
  });
  return `<div class="chances">${rows.join("")}</div>`;
}

/** Writes the first few numbers of a long list as short text, for example "[0.12, -0.40, …]". */
export const shortList = (values: number[], limit = 4) => `[${values.slice(0, limit).map((value) => value.toFixed(2)).join(", ")}, …]`;

/** Plots the salary dots from the lessons. Pass `model` to draw its prediction and `showMisses` to draw each dot's miss. */
export function salaryPlot(options: { model?: (x: number) => number; showMisses?: boolean } = {}) {
  const X_MAX = 8;
  const Y_MAX = 140;
  const { width, height, left, right, top, bottom } = PLOT;
  const xAt = (x: number) => left + (x / X_MAX) * (width - left - right);
  const yAt = (y: number) => top + (1 - y / Y_MAX) * (height - top - bottom);
  const grid = [0, 70, 140]
    .map((y) => `<line x1="${left}" x2="${width - right}" y1="${yAt(y)}" y2="${yAt(y)}" class="plot-axis"/><text x="${left - 10}" y="${yAt(y) + 5}" text-anchor="end" class="plot-text">${y}</text>`)
    .join("");
  const ticks = YEARS.map((x) => `<text x="${xAt(x)}" y="${height - bottom + 20}" text-anchor="middle" class="plot-text">${x}</text>`).join("");
  const { model } = options;
  const samples = Array.from({ length: 81 }, (_, i) => (i / 80) * X_MAX);
  const line = model
    ? `<polyline clip-path="url(#plot-area)" class="plot-line" points="${samples.map((x) => `${xAt(x).toFixed(1)},${yAt(model(x)).toFixed(1)}`).join(" ")}"/>`
    : "";
  const misses = model && options.showMisses
    ? YEARS.map((x, i) => `<line clip-path="url(#plot-area)" class="plot-miss" x1="${xAt(x)}" x2="${xAt(x)}" y1="${yAt(SALARIES[i])}" y2="${yAt(model(x))}"/>`).join("")
    : "";
  const dots = YEARS.map((x, i) => `<circle cx="${xAt(x)}" cy="${yAt(SALARIES[i])}" r="7" class="plot-dot"><title>${x} years: salary ${SALARIES[i]}</title></circle>`).join("");
  return `<svg class="plot" viewBox="0 0 ${width} ${height}" role="img" aria-label="Salary against years of work">
    <defs><clipPath id="plot-area"><rect x="${left}" y="${top}" width="${width - left - right}" height="${height - top - bottom}"/></clipPath></defs>
    ${grid}${ticks}
    <text x="${(left + width - right) / 2}" y="${height - 4}" text-anchor="middle" class="plot-text">years of work → salary</text>
    ${misses}${line}${dots}
  </svg>`;
}

/** Plots a small curve function, used for the activation functions. */
export function curvePlot(curve: (z: number) => number, yMin: number, yMax: number) {
  const { width, height, left, right, top, bottom } = PLOT;
  const Z_MAX = 5;
  const xAt = (z: number) => left + ((z + Z_MAX) / (2 * Z_MAX)) * (width - left - right);
  const yAt = (y: number) => top + (1 - (y - yMin) / (yMax - yMin)) * (height - top - bottom);
  const samples = Array.from({ length: 101 }, (_, i) => -Z_MAX + (i / 100) * 2 * Z_MAX);
  const grid = [yMin, yMax].map((y) => `<line x1="${left}" x2="${width - right}" y1="${yAt(y)}" y2="${yAt(y)}" class="plot-axis"/><text x="${left - 10}" y="${yAt(y) + 5}" text-anchor="end" class="plot-text">${y}</text>`).join("");
  const ticks = [-5, 0, 5].map((z) => `<text x="${xAt(z)}" y="${height - bottom + 20}" text-anchor="middle" class="plot-text">${z}</text>`).join("");
  return `<svg class="plot" viewBox="0 0 ${width} ${height}" role="img" aria-label="Curve for inputs from minus 5 to 5">
    ${grid}${ticks}
    <text x="${(left + width - right) / 2}" y="${height - 4}" text-anchor="middle" class="plot-text">number going in (z)</text>
    <polyline class="plot-line" points="${samples.map((z) => `${xAt(z).toFixed(1)},${yAt(curve(z)).toFixed(1)}`).join(" ")}"/>
  </svg>`;
}

/** Draws the lessons' small network: one input, three neurons in the hidden layer, one output. */
export function networkDiagram() {
  const hiddenY = [50, 125, 200];
  const links = [
    ...hiddenY.map((y) => `<line class="net-link" x1="110" y1="125" x2="320" y2="${y}"/>`),
    ...hiddenY.map((y) => `<line class="net-link" x1="320" y1="${y}" x2="530" y2="125"/>`),
  ].join("");
  const hidden = hiddenY.map((y, i) => `<circle class="net-node net-node--hidden" cx="320" cy="${y}" r="30"/><text class="net-text" x="320" y="${y + 6}">h${i + 1}</text>`).join("");
  return `<svg class="plot" viewBox="0 0 640 275" role="img" aria-label="A network: one input, three hidden neurons and one output">
    ${links}
    <circle class="net-node" cx="110" cy="125" r="30"/><text class="net-text" x="110" y="131">x</text>
    ${hidden}
    <circle class="net-node" cx="530" cy="125" r="30"/><text class="net-text" x="530" y="131">out</text>
    <text class="net-text" x="110" y="265">Input</text><text class="net-text" x="320" y="265">Hidden layer</text><text class="net-text" x="530" y="265">Output</text>
  </svg>`;
}

/** Draws how loss grows as the chance given to the right word falls, marking the current chance. */
export function lossCurve(chance: number) {
  const { width, height, left, right, top, bottom } = PLOT;
  const MAX_LOSS = 5;
  const xAt = (p: number) => left + p * (width - left - right);
  const yAt = (loss: number) => top + (1 - Math.min(loss, MAX_LOSS) / MAX_LOSS) * (height - top - bottom);
  const samples = Array.from({ length: 100 }, (_, i) => (i + 1) / 100);
  const grid = [0, MAX_LOSS].map((y) => `<line x1="${left}" x2="${width - right}" y1="${yAt(y)}" y2="${yAt(y)}" class="plot-axis"/><text x="${left - 10}" y="${yAt(y) + 5}" text-anchor="end" class="plot-text">${y}</text>`).join("");
  return `<svg class="plot" viewBox="0 0 ${width} ${height}" role="img" aria-label="Loss for each chance given to the right word">
    ${grid}
    <text x="${(left + width - right) / 2}" y="${height - 4}" text-anchor="middle" class="plot-text">chance given to the right word: 0% to 100%</text>
    <polyline class="plot-line" points="${samples.map((p) => `${xAt(p).toFixed(1)},${yAt(-Math.log(p)).toFixed(1)}`).join(" ")}"/>
    <circle class="plot-dot" cx="${xAt(chance)}" cy="${yAt(-Math.log(chance))}" r="8"/>
  </svg>`;
}

/** Plots the loss over training rounds, with a dashed line at the level of pure random guessing. Pass `earlier` to draw an older run behind it. */
export function lossPlot(points: LossPoint[], rounds: number, guessingLoss: number, options: { earlier?: LossPoint[]; width?: number; height?: number } = {}) {
  const { left, right, top, bottom } = PLOT;
  const { earlier = [], width = PLOT.width, height = PLOT.height } = options;
  const yMax = Math.ceil(Math.max(guessingLoss, ...points.map((point) => point.loss), ...earlier.map((point) => point.loss)));
  const xAt = (round: number) => left + (round / rounds) * (width - left - right);
  const yAt = (loss: number) => top + (1 - loss / yMax) * (height - top - bottom);
  const grid = [0, yMax]
    .map((y) => `<line x1="${left}" x2="${width - right}" y1="${yAt(y)}" y2="${yAt(y)}" class="plot-axis"/><text x="${left - 10}" y="${yAt(y) + 5}" text-anchor="end" class="plot-text">${y}</text>`)
    .join("");
  const polyline = (series: LossPoint[], className: string) =>
    series.length > 1 ? `<polyline class="${className}" points="${series.map((point) => `${xAt(point.step).toFixed(1)},${yAt(point.loss).toFixed(1)}`).join(" ")}"/>` : "";
  const line = polyline(earlier, "plot-line plot-line--earlier") + polyline(points, "plot-line");
  return `<svg class="plot" viewBox="0 0 ${width} ${height}" role="img" aria-label="Loss over ${rounds} training rounds">
    ${grid}
    <line x1="${left}" x2="${width - right}" y1="${yAt(guessingLoss)}" y2="${yAt(guessingLoss)}" class="plot-baseline"/>
    <text x="${width - right}" y="${yAt(guessingLoss) - 8}" text-anchor="end" class="plot-text">random guessing</text>
    <text x="${left}" y="${height - 16}" class="plot-text">round 0</text>
    <text x="${width - right}" y="${height - 16}" text-anchor="end" class="plot-text">round ${rounds}</text>
    ${line}
  </svg>`;
}

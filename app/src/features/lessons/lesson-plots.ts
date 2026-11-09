import { SALARIES, YEARS } from "@/core/model/line-model";
import { PLOT } from "@/shared/utils/charts";
import { showWord } from "@/shared/utils/format";
import { escapeHtml } from "@/shared/utils/html";

export function wordCards(words: string[], options: { numbers?: (number | string)[]; newFrom?: number } = {}) {
  const cards = words.map((word, i) => {
    const isNew = options.newFrom !== undefined && i >= options.newFrom;
    const number = options.numbers ? `<small>${options.numbers[i]}</small>` : "";
    return `<span class="word${isNew ? " is-new" : ""}">${escapeHtml(showWord(word))}${number}</span>`;
  });
  return `<div class="words">${cards.join("")}</div>`;
}

export const shortList = (values: number[], limit = 4) => `[${values.slice(0, limit).map((value) => value.toFixed(2)).join(", ")}, …]`;

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

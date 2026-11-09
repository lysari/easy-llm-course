import type { LossPoint } from "@/shared/interfaces/model.interface";
import { percent, showWord } from "./format";
import { escapeHtml } from "./html";

export const PLOT = { width: 640, height: 300, left: 54, right: 16, top: 12, bottom: 46 };

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
    series.length > 1 ? `<polyline class="${className}" points="${series.map((point) => `${xAt(point.round).toFixed(1)},${yAt(point.loss).toFixed(1)}`).join(" ")}"/>` : "";
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

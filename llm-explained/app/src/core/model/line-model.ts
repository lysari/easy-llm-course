import type { Line } from "@/shared/interfaces/line.interface";

export const YEARS = [1, 2, 3, 4, 5, 6, 7];
export const SALARIES = [40, 50, 60, 70, 80, 100, 115];
export const LINE_LEARNING_RATE = 0.01;
export const LINE_TOTAL_ROUNDS = 10000;

export const predictOnLine = (line: Line, x: number) => line.w * x + line.b;

export function lineLoss(line: Line): number {
  const total = YEARS.reduce((sum, x, i) => sum + (predictOnLine(line, x) - SALARIES[i]) ** 2, 0);
  return total / YEARS.length;
}

export function trainLineRound(line: Line) {
  let dw = 0;
  let db = 0;
  YEARS.forEach((x, i) => {
    const error = predictOnLine(line, x) - SALARIES[i];
    dw += error * x;
    db += error;
  });
  line.w -= LINE_LEARNING_RATE * (2 / YEARS.length) * dw;
  line.b -= LINE_LEARNING_RATE * (2 / YEARS.length) * db;
}

export function trainedLine(): Line {
  const line = { w: 0, b: 0 };
  for (let i = 0; i < LINE_TOTAL_ROUNDS; i++) trainLineRound(line);
  return line;
}

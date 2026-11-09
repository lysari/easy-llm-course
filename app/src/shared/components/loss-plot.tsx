import { Fragment } from "react";
import type { LossPoint } from "@/shared/interfaces/model.interface";

export const PLOT = { width: 640, height: 300, left: 54, right: 16, top: 12, bottom: 46 };

interface LossPlotProps {
  points: LossPoint[];
  rounds: number;
  guessingLoss: number;
  earlier?: LossPoint[];
  width?: number;
  height?: number;
}

export function LossPlot({ points, rounds, guessingLoss, earlier = [], width = PLOT.width, height = PLOT.height }: LossPlotProps) {
  const { left, right, top, bottom } = PLOT;
  const yMax = Math.ceil(Math.max(guessingLoss, ...points.map((point) => point.loss), ...earlier.map((point) => point.loss)));
  const xAt = (round: number) => left + (round / rounds) * (width - left - right);
  const yAt = (loss: number) => top + (1 - loss / yMax) * (height - top - bottom);
  const polyline = (series: LossPoint[], className: string) =>
    series.length > 1 && <polyline className={className} points={series.map((point) => `${xAt(point.round).toFixed(1)},${yAt(point.loss).toFixed(1)}`).join(" ")} />;
  return (
    <svg className="plot" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Loss over ${rounds} training rounds`}>
      {[0, yMax].map((y) => (
        <Fragment key={y}>
          <line x1={left} x2={width - right} y1={yAt(y)} y2={yAt(y)} className="plot-axis" />
          <text x={left - 10} y={yAt(y) + 5} textAnchor="end" className="plot-text">{y}</text>
        </Fragment>
      ))}
      <line x1={left} x2={width - right} y1={yAt(guessingLoss)} y2={yAt(guessingLoss)} className="plot-baseline" />
      <text x={width - right} y={yAt(guessingLoss) - 8} textAnchor="end" className="plot-text">random guessing</text>
      <text x={left} y={height - 16} className="plot-text">round 0</text>
      <text x={width - right} y={height - 16} textAnchor="end" className="plot-text">{`round ${rounds}`}</text>
      {polyline(earlier, "plot-line plot-line--earlier")}
      {polyline(points, "plot-line")}
    </svg>
  );
}

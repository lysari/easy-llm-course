import { Fragment } from "react";
import { SALARIES, YEARS } from "@/core/model/line-model";
import { PLOT } from "@/shared/components/loss-plot";
import { showWord } from "@/shared/utils/format";

interface WordCardsProps {
  words: string[];
  numbers?: (number | string)[];
  newFrom?: number;
}

interface SalaryPlotProps {
  model?: (x: number) => number;
  showMisses?: boolean;
}

interface CurvePlotProps {
  curve: (z: number) => number;
  yMin: number;
  yMax: number;
}

export function WordCards({ words, numbers, newFrom }: WordCardsProps) {
  return (
    <div className="words">
      {words.map((word, i) => (
        <span key={i} className={`word${newFrom !== undefined && i >= newFrom ? " is-new" : ""}`}>
          {showWord(word)}
          {numbers && <small>{numbers[i]}</small>}
        </span>
      ))}
    </div>
  );
}

export const shortList = (values: number[], limit = 4) => `[${values.slice(0, limit).map((value) => value.toFixed(2)).join(", ")}, …]`;

export function SalaryPlot({ model, showMisses }: SalaryPlotProps) {
  const X_MAX = 8;
  const Y_MAX = 140;
  const { width, height, left, right, top, bottom } = PLOT;
  const xAt = (x: number) => left + (x / X_MAX) * (width - left - right);
  const yAt = (y: number) => top + (1 - y / Y_MAX) * (height - top - bottom);
  const samples = Array.from({ length: 81 }, (_, i) => (i / 80) * X_MAX);
  return (
    <svg className="plot" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Salary against years of work">
      <defs>
        <clipPath id="plot-area">
          <rect x={left} y={top} width={width - left - right} height={height - top - bottom} />
        </clipPath>
      </defs>
      {[0, 70, 140].map((y) => (
        <Fragment key={y}>
          <line x1={left} x2={width - right} y1={yAt(y)} y2={yAt(y)} className="plot-axis" />
          <text x={left - 10} y={yAt(y) + 5} textAnchor="end" className="plot-text">{y}</text>
        </Fragment>
      ))}
      {YEARS.map((x) => (
        <text key={x} x={xAt(x)} y={height - bottom + 20} textAnchor="middle" className="plot-text">{x}</text>
      ))}
      <text x={(left + width - right) / 2} y={height - 4} textAnchor="middle" className="plot-text">years of work → salary</text>
      {model && showMisses && YEARS.map((x, i) => (
        <line key={x} clipPath="url(#plot-area)" className="plot-miss" x1={xAt(x)} x2={xAt(x)} y1={yAt(SALARIES[i])} y2={yAt(model(x))} />
      ))}
      {model && <polyline clipPath="url(#plot-area)" className="plot-line" points={samples.map((x) => `${xAt(x).toFixed(1)},${yAt(model(x)).toFixed(1)}`).join(" ")} />}
      {YEARS.map((x, i) => (
        <circle key={x} cx={xAt(x)} cy={yAt(SALARIES[i])} r={7} className="plot-dot">
          <title>{`${x} years: salary ${SALARIES[i]}`}</title>
        </circle>
      ))}
    </svg>
  );
}

export function CurvePlot({ curve, yMin, yMax }: CurvePlotProps) {
  const { width, height, left, right, top, bottom } = PLOT;
  const Z_MAX = 5;
  const xAt = (z: number) => left + ((z + Z_MAX) / (2 * Z_MAX)) * (width - left - right);
  const yAt = (y: number) => top + (1 - (y - yMin) / (yMax - yMin)) * (height - top - bottom);
  const samples = Array.from({ length: 101 }, (_, i) => -Z_MAX + (i / 100) * 2 * Z_MAX);
  return (
    <svg className="plot" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Curve for inputs from minus 5 to 5">
      {[yMin, yMax].map((y) => (
        <Fragment key={y}>
          <line x1={left} x2={width - right} y1={yAt(y)} y2={yAt(y)} className="plot-axis" />
          <text x={left - 10} y={yAt(y) + 5} textAnchor="end" className="plot-text">{y}</text>
        </Fragment>
      ))}
      {[-5, 0, 5].map((z) => (
        <text key={z} x={xAt(z)} y={height - bottom + 20} textAnchor="middle" className="plot-text">{z}</text>
      ))}
      <text x={(left + width - right) / 2} y={height - 4} textAnchor="middle" className="plot-text">number going in (z)</text>
      <polyline className="plot-line" points={samples.map((z) => `${xAt(z).toFixed(1)},${yAt(curve(z)).toFixed(1)}`).join(" ")} />
    </svg>
  );
}

export function NetworkDiagram() {
  const hiddenY = [50, 125, 200];
  return (
    <svg className="plot" viewBox="0 0 640 275" role="img" aria-label="A network: one input, three hidden neurons and one output">
      {hiddenY.map((y) => <line key={`in-${y}`} className="net-link" x1="110" y1="125" x2="320" y2={y} />)}
      {hiddenY.map((y) => <line key={`out-${y}`} className="net-link" x1="320" y1={y} x2="530" y2="125" />)}
      <circle className="net-node" cx="110" cy="125" r="30" />
      <text className="net-text" x="110" y="131">x</text>
      {hiddenY.map((y, i) => (
        <Fragment key={y}>
          <circle className="net-node net-node--hidden" cx="320" cy={y} r="30" />
          <text className="net-text" x="320" y={y + 6}>{`h${i + 1}`}</text>
        </Fragment>
      ))}
      <circle className="net-node" cx="530" cy="125" r="30" />
      <text className="net-text" x="530" y="131">out</text>
      <text className="net-text" x="110" y="265">Input</text>
      <text className="net-text" x="320" y="265">Hidden layer</text>
      <text className="net-text" x="530" y="265">Output</text>
    </svg>
  );
}

export function LossCurve({ chance }: { chance: number }) {
  const { width, height, left, right, top, bottom } = PLOT;
  const MAX_LOSS = 5;
  const xAt = (p: number) => left + p * (width - left - right);
  const yAt = (loss: number) => top + (1 - Math.min(loss, MAX_LOSS) / MAX_LOSS) * (height - top - bottom);
  const samples = Array.from({ length: 100 }, (_, i) => (i + 1) / 100);
  return (
    <svg className="plot" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Loss for each chance given to the right word">
      {[0, MAX_LOSS].map((y) => (
        <Fragment key={y}>
          <line x1={left} x2={width - right} y1={yAt(y)} y2={yAt(y)} className="plot-axis" />
          <text x={left - 10} y={yAt(y) + 5} textAnchor="end" className="plot-text">{y}</text>
        </Fragment>
      ))}
      <text x={(left + width - right) / 2} y={height - 4} textAnchor="middle" className="plot-text">chance given to the right word: 0% to 100%</text>
      <polyline className="plot-line" points={samples.map((p) => `${xAt(p).toFixed(1)},${yAt(-Math.log(p)).toFixed(1)}`).join(" ")} />
      <circle className="plot-dot" cx={xAt(chance)} cy={yAt(-Math.log(chance))} r="8" />
    </svg>
  );
}

export function GradientBars({ tables }: { tables: { name: string; size: number }[] }) {
  const largest = Math.max(...tables.map((table) => table.size));
  return (
    <div className="chances">
      {tables.map(({ name, size }) => (
        <div key={name} className="chance" title={`${name}: ${size.toFixed(4)}`}>
          <span>{name.replace("feed-forward", "FFN").replace("attention", "attn").replace("Layer ", "L")}</span>
          <span className="chance-track">
            <i className="chance-fill" style={{ width: `${Math.round((size / largest) * 100)}%` }} />
          </span>
          <span>{size.toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}

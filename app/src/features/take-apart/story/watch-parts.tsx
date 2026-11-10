import type { ReactNode } from "react";
import { showWord } from "@/shared/utils/format";

interface NumberRowProps {
  label: string;
  values: number[];
  picked?: number;
  onPick?: (index: number) => void;
}

export function NumberRow({ label, values, picked, onPick }: NumberRowProps) {
  const biggest = Math.max(...values.map(Math.abs), 1e-9);
  return (
    <figure className="number-row">
      <figcaption className="card-lesson">{`${label} · ${values.length} number${values.length === 1 ? "" : "s"}`}</figcaption>
      <ol className="number-cells">
        {values.map((value, i) => {
          const cell = (
            <>
              <i className="number-bar" style={{ height: `${Math.max(4, Math.round((Math.abs(value) / biggest) * 100))}%` }} />
              <span>{Number.isInteger(value) ? value : value.toFixed(2)}</span>
            </>
          );
          const className = `number-cell${value < 0 ? " is-negative" : ""}${i === picked ? " is-picked" : ""}`;
          return (
            <li key={i}>
              {onPick ? (
                <button type="button" className={className} aria-pressed={i === picked} aria-label={`Number ${i + 1}: ${value.toFixed(3)}`} onClick={() => onPick(i)}>
                  {cell}
                </button>
              ) : (
                <span className={className} title={value.toFixed(4)}>
                  {cell}
                </span>
              )}
            </li>
          );
        })}
      </ol>
    </figure>
  );
}

export function BeforeAfter({ before, after, beforeLabel = "Before", afterLabel = "After" }: { before: ReactNode; after: ReactNode; beforeLabel?: string; afterLabel?: string }) {
  return (
    <div className="before-after">
      <div className="before-after-side">
        <p className="section-title">{beforeLabel}</p>
        {before}
      </div>
      <span className="before-after-arrow" aria-hidden="true">
        →
      </span>
      <div className="before-after-side">
        <p className="section-title">{afterLabel}</p>
        {after}
      </div>
    </div>
  );
}

export function HonestNote({ children }: { children: ReactNode }) {
  return <p className="watch-honest">{children}</p>;
}

export function WordPicker({ label, words, picked, onPick }: { label: string; words: string[]; picked: number; onPick: (index: number) => void }) {
  return (
    <div className="word-picker" role="group" aria-label={label}>
      <span className="card-lesson">{label}</span>
      {words.map((word, i) => (
        <button key={`${word}-${i}`} type="button" className={`word-chip${i === picked ? " is-picked" : ""}`} aria-pressed={i === picked} onClick={() => onPick(i)}>
          {showWord(word)}
        </button>
      ))}
    </div>
  );
}

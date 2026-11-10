import "./architecture.css";
import { useId, useState } from "react";
import type { DemoProps } from "@/features/take-apart/interfaces/model-part.interface";
import { BeforeAfter, HonestNote, NumberRow, WordPicker } from "@/features/take-apart/story/watch-parts";
import { count } from "@/shared/utils/format";
import { type Terms, tinyStep } from "./tiny-steps";

const EXTRA_RANGE = { min: -2, max: 2, step: 0.05 };

const fixed = (value: number) => value.toFixed(4);

function AllTerms({ terms }: { terms: Terms }) {
  return (
    <div className="watch-box">
      <p className="section-title">{terms.pairs.length === 1 ? "The one multiply" : `${count(terms.pairs.length)} multiply-and-adds, all of them`}</p>
      <ol className="term-list">
        {terms.pairs.map(([a, b], i) => (
          <li key={i}>
            <span>{`${fixed(a)} × ${fixed(b)}`}</span>
            <b>{fixed(a * b)}</b>
          </li>
        ))}
      </ol>
      <p className="note">{`Sum${terms.scale === 1 ? "" : ` × ${fixed(terms.scale)}`}${terms.add === 0 ? "" : ` + ${fixed(terms.add)}`} = ${fixed(terms.total)}`}</p>
    </div>
  );
}

function OneTerm({ terms, index, onIndex }: { terms: Terms; index: number; onIndex: (index: number) => void }) {
  const i = Math.min(index, terms.pairs.length - 1);
  const sumBefore = terms.pairs.slice(0, i).reduce((sum, [a, b]) => sum + a * b, 0);
  const [a, b] = terms.pairs[i];
  return (
    <div className="watch-box">
      <div className="watch-controls">
        <button type="button" className="action" onClick={() => onIndex(i - 1)} disabled={i === 0}>
          Back
        </button>
        <button type="button" className="action action--strong" onClick={() => onIndex(i + 1)} disabled={i === terms.pairs.length - 1}>
          Next multiply-and-add
        </button>
        <span className="card-lesson">{`${count(i + 1)} of ${count(terms.pairs.length)}`}</span>
      </div>
      <BeforeAfter
        beforeLabel={`Running sum after ${count(i)}`}
        before={<p className="term-big">{fixed(sumBefore)}</p>}
        afterLabel={`+ ${terms.leftName} × ${terms.rightName}`}
        after={<p className="term-big">{`${fixed(sumBefore)} + ${fixed(a)} × ${fixed(b)} = ${fixed(sumBefore + a * b)}`}</p>}
      />
      {i === terms.pairs.length - 1 && <p className="note">{`Done: the full sum${terms.scale === 1 ? "" : ` × ${fixed(terms.scale)}`} is ${fixed(terms.total)}.`}</p>}
    </div>
  );
}

export function TinyPartDemo({ part, modelId, tiny }: DemoProps) {
  const [word, setWord] = useState(tiny.words.length - 1);
  const [out, setOut] = useState(0);
  const [extra, setExtra] = useState(part.id.endsWith("-bias") ? 0 : 1);
  const [term, setTerm] = useState(0);
  const pickId = useId();
  const extraId = useId();
  const shown = tinyStep(part, modelId, tiny, { word, out, extra });
  const isStepping = part.id.endsWith("multiply-add");
  const hasExtra = part.id.endsWith("-bias") || part.id === "gate-times-up" || part.id === "swiglu-number";

  return (
    <div className="watch">
      <WordPicker label="Watch this word of your sentence:" words={tiny.words} picked={Math.min(word, tiny.words.length - 1)} onPick={(i) => { setWord(i); setTerm(0); }} />
      {shown.pick && (
        <div className="watch-slider">
          <label htmlFor={pickId}>{`${shown.pick.label} ${(out % shown.pick.count) + 1} of ${shown.pick.count}`}</label>
          <input id={pickId} type="range" min={0} max={shown.pick.count - 1} value={out % shown.pick.count} onChange={(event) => { setOut(Number(event.target.value)); setTerm(0); }} />
        </div>
      )}
      {hasExtra && (
        <div className="watch-slider">
          <label htmlFor={extraId}>{`${part.id.endsWith("-bias") ? "Bias b" : "Up number"}: ${fixed(extra)}`}</label>
          <input id={extraId} type="range" min={EXTRA_RANGE.min} max={EXTRA_RANGE.max} step={EXTRA_RANGE.step} value={extra} onChange={(event) => setExtra(Number(event.target.value))} />
        </div>
      )}
      <BeforeAfter
        before={shown.before.map((list) => <NumberRow key={list.label} label={list.label} values={list.values} picked={isStepping && list.values.length === shown.terms?.pairs.length ? Math.min(term, list.values.length - 1) : undefined} />)}
        after={shown.after.map((list) => <NumberRow key={list.label} label={list.label} values={list.values} />)}
      />
      <p className="formula-line">{shown.formula}</p>
      {shown.terms && isStepping && <OneTerm terms={shown.terms} index={term} onIndex={setTerm} />}
      {shown.terms && !isStepping && <AllTerms terms={shown.terms} />}
      <HonestNote>{`${shown.note} The real ${part.title} in this model: ${part.facts.shape}.`}</HonestNote>
    </div>
  );
}

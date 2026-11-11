import { useId, useState } from "react";
import type { ReadabilityTrace } from "@/features/inside-real-models/interfaces/readability-trace.interface";
import { walkStepsOf } from "./how-it-works-steps";
import { PageBoxes } from "./page-boxes";
import { ScoreDetails } from "./score-details";

interface HowItWorksStepProps {
  trace: ReadabilityTrace;
  step: number;
  onStep: (step: number) => void;
  onShowKept: () => void;
}

export function HowItWorksStep({ trace, step, onStep, onShowKept }: HowItWorksStepProps) {
  const headingId = useId();
  const steps = walkStepsOf(trace);
  const shown = steps[step];
  const isFirst = step === 0;
  const isLast = step === steps.length - 1;
  const position = `Step ${step + 1} of ${steps.length}`;

  return (
    <section className="how" aria-labelledby={headingId}>
      <div className="how-head">
        <h4 id={headingId} className="card-title">
          How it works, step by step
        </h4>
        <div className="how-controls">
          <span className="card-lesson how-position">{position}</span>
          <button type="button" className="action" aria-disabled={isFirst} onClick={() => !isFirst && onStep(step - 1)}>
            ‹ Back
          </button>
          <button type="button" className="action action--strong" aria-disabled={isLast} onClick={() => !isLast && onStep(step + 1)}>
            Next ›
          </button>
        </div>
      </div>
      <ol className="how-dots" aria-hidden="true">
        {steps.map((each, i) => (
          <li key={each.title} className={i === step ? "is-current" : i < step ? "is-done" : undefined} />
        ))}
      </ol>
      <div className="how-step">
        <h5 className="how-title">{`${step + 1}. ${shown.title}`}</h5>
        <p className="how-sentence">{shown.sentence}</p>
      </div>
      <p className="visually-hidden" aria-live="polite">{`${position}: ${shown.title}. ${shown.sentence}`}</p>
      {step === 2 && <ScoreDetails trace={trace} />}
      <PageBoxes trace={trace} step={step} />
      {isLast && (
        <button type="button" className="action action--strong how-kept" onClick={onShowKept}>
          Show the kept text
        </button>
      )}
    </section>
  );
}

export function HowItWorks({ trace, onShowKept }: { trace: ReadabilityTrace; onShowKept: () => void }) {
  const [shown, setShown] = useState({ trace, step: 0 });
  const step = shown.trace === trace ? shown.step : 0;
  return <HowItWorksStep trace={trace} step={step} onStep={(next) => setShown({ trace, step: next })} onShowKept={onShowKept} />;
}

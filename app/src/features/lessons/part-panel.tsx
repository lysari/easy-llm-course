import { type RefObject, useRef } from "react";
import type { FlowStep } from "@/features/lessons/interfaces/story.interface";
import { usePanelPlace } from "./hooks/use-panel-place";
import type { Session } from "./session";

interface PartPanelProps {
  step: FlowStep;
  session: Session;
  partCards: RefObject<Map<string, HTMLElement>>;
  width: number;
  onCommand: (command: string) => void;
  onClick: () => void;
}

export function PartPanel({ step, session, partCards, width, onCommand, onClick }: PartPanelProps) {
  const panelRef = useRef<HTMLElement>(null);
  const place = usePanelPlace(panelRef, partCards, step.id, width);
  const rows = step.rows?.(session) ?? [];
  const note = step.note?.(session);

  return (
    <section
      className="panel"
      id="panel"
      aria-label={`${step.title} details`}
      ref={panelRef}
      style={place}
      onClick={(event) => {
        event.stopPropagation();
        onClick();
      }}
    >
      <div className="panel-inner" id="panel-inner">
        {rows.length > 0 && (
          <section className="section">
            <p className="section-title">{step.rowsTitle ?? "Step by step"}</p>
            <div className="tasks">
              {rows.map(({ text, value, command, isOn }, i) =>
                command ? (
                  <button key={i} type="button" className={`task${isOn ? " is-on" : ""}`} aria-pressed={Boolean(isOn)} onClick={() => onCommand(command)}>
                    <span>{text}</span>
                    <span>{value}</span>
                  </button>
                ) : (
                  <div key={i} className="task">
                    <span>{text}</span>
                    <span>{value}</span>
                  </div>
                ),
              )}
            </div>
          </section>
        )}
        {step.visual && <section className="section">{step.visual(session)}</section>}
        {note && <p className="note">{note}</p>}
        {step.formula && (
          <section className="section">
            <p className="section-title">Formula</p>
            <pre className="code">{step.formula}</pre>
          </section>
        )}
        {step.code && (
          <section className="section">
            <p className="section-title">Code</p>
            <pre className="code">{step.code}</pre>
          </section>
        )}
        <p className="why">{step.why}</p>
      </div>
    </section>
  );
}

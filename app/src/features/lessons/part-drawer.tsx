import type { Ref } from "react";
import type { FlowStep } from "@/features/lessons/interfaces/story.interface";
import { Icon } from "@/shared/components/icon";
import { CodeBlock } from "./code-block";
import { FormulaBlock } from "./formula-block";
import { DRAWER_ID, StatusPill } from "./lesson-cards";
import type { Session } from "./session";

interface PartDrawerProps {
  ref: Ref<HTMLElement>;
  step: FlowStep;
  session: Session;
  onCommand: (command: string) => void;
  onClose: () => void;
}

export function PartDrawer({ ref, step, session, onCommand, onClose }: PartDrawerProps) {
  const rows = step.rows?.(session) ?? [];
  const note = step.note?.(session);

  return (
    <aside className="drawer" id={DRAWER_ID} ref={ref} tabIndex={-1} aria-label={`${step.title} details`}>
      <header className="card-head drawer-head">
        <span className="tile"><Icon name={step.icon} /></span>
        <div>
          <h2 className="card-title">{step.title}</h2>
          <p className="card-lesson">{step.lesson}</p>
        </div>
        <StatusPill status={step.status(session)} />
        <button type="button" className="drawer-close" aria-label="Close details" onClick={onClose}>
          <Icon name="close" />
        </button>
      </header>
      <div className="drawer-body" key={step.id}>
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
        {step.formula && <FormulaBlock formula={step.formula} />}
        {step.code && <CodeBlock code={step.code} />}
        <p className="why">{step.why}</p>
      </div>
    </aside>
  );
}

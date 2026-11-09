import type { Ref } from "react";
import type { FlowStep } from "@/features/lessons/interfaces/story.interface";
import { CodeBlock } from "@/shared/components/code-block";
import { Drawer } from "@/shared/components/drawer";
import { FormulaBlock } from "@/shared/components/formula-block";
import { DRAWER_ID } from "./lesson-cards";
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
    <Drawer ref={ref} id={DRAWER_ID} icon={step.icon} title={step.title} lesson={step.lesson} status={step.status(session)} bodyKey={step.id} onClose={onClose}>
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
    </Drawer>
  );
}

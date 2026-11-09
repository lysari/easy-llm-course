import type { KeyboardEvent, ReactNode, Ref } from "react";
import type { CardAction, FlowStage, FlowStep, Status } from "@/features/lessons/interfaces/story.interface";
import { Icon } from "@/shared/components/icon";
import { STAGGER_MS } from "./flow-links";
import type { Session } from "./session";

const STATUS_TEXT: Record<Status, string> = { running: " Running", done: "✓ Done", needs: " Needs you", queued: "• Queued", lesson: "◦ Lesson only" };
const STATUS_ICONS: Partial<Record<Status, ReactNode>> = {
  running: <i className="spinner" />,
  needs: (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" aria-hidden="true">
      <circle cx="6" cy="6" r="5" />
      <path d="M6 3.2V6l1.8 1.2" />
    </svg>
  ),
};

interface CardBodyProps {
  icon: string;
  title: string;
  lesson: string;
  status: Status;
  task: string;
  foot: ReactNode;
  actions: CardAction[];
  onCommand: (command: string) => void;
}

interface StageCardProps {
  ref: Ref<HTMLElement>;
  stage: FlowStage;
  session: Session;
  isSelected: boolean;
  onSelect: () => void;
  onCommand: (command: string) => void;
}

interface PartCardProps {
  ref: Ref<HTMLElement>;
  step: FlowStep;
  session: Session;
  index: number;
  isNested: boolean;
  isActive: boolean;
  onSelect: () => void;
  onCommand: (command: string) => void;
}

function activateOnKey(event: KeyboardEvent<HTMLElement>) {
  if (event.target !== event.currentTarget || (event.key !== "Enter" && event.key !== " ")) return;
  event.preventDefault();
  event.currentTarget.click();
}

function CardBody({ icon, title, lesson, status, task, foot, actions, onCommand }: CardBodyProps) {
  return (
    <>
      <div className="card-head">
        <span className="tile"><Icon name={icon} /></span>
        <div>
          <h3 className="card-title">{title}</h3>
          <p className="card-lesson">{lesson}</p>
        </div>
        <span className={`pill pill--${status}`}>{STATUS_ICONS[status]}{STATUS_TEXT[status]}</span>
      </div>
      <p className="card-task">{task}</p>
      <div className="card-foot">
        <span className="foot-value">{foot}</span>
        {actions.length > 0 && (
          <span className="foot-actions">
            {actions.map((action, i) => (
              <button
                key={i}
                type="button"
                className={`action${action.isStrong ? " action--strong" : ""}`}
                onClick={(event) => {
                  event.stopPropagation();
                  onCommand(action.command);
                }}
              >
                {action.label}
              </button>
            ))}
          </span>
        )}
      </div>
    </>
  );
}

export function StageCard({ ref, stage, session, isSelected, onSelect, onCommand }: StageCardProps) {
  const action = stage.action?.(session);
  return (
    <article
      ref={ref}
      className={`card${isSelected ? "" : " is-dim"}`}
      id={`stage-${stage.id}`}
      tabIndex={0}
      aria-label={stage.title}
      aria-current={isSelected}
      onClick={(event) => {
        event.stopPropagation();
        onSelect();
      }}
      onKeyDown={activateOnKey}
    >
      <CardBody
        icon={stage.icon}
        title={stage.title}
        lesson={stage.lesson}
        status={stage.status(session)}
        task={stage.task}
        foot={<><b>{stage.value(session)}</b>{` · ${stage.detail(session)}`}</>}
        actions={action ? [action] : []}
        onCommand={onCommand}
      />
    </article>
  );
}

export function PartCard({ ref, step, session, index, isNested, isActive, onSelect, onCommand }: PartCardProps) {
  return (
    <article
      ref={ref}
      className={`card is-entering${isNested ? " is-nested" : ""}${isActive ? " is-active" : ""}`}
      id={`part-${step.id}`}
      tabIndex={0}
      aria-label={step.title}
      style={{ animationDelay: `${index * STAGGER_MS}ms` }}
      onClick={(event) => {
        event.stopPropagation();
        onSelect();
      }}
      onKeyDown={activateOnKey}
    >
      <CardBody
        icon={step.icon}
        title={step.title}
        lesson={step.lesson}
        status={step.status(session)}
        task={step.goal}
        foot={<>{`${step.from(session)} → `}<b>{step.to(session)}</b></>}
        actions={step.actions?.(session) ?? []}
        onCommand={onCommand}
      />
    </article>
  );
}

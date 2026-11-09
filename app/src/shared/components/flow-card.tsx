import type { HTMLAttributes, KeyboardEvent, ReactNode, Ref } from "react";
import type { CardAction, Status } from "@/shared/interfaces/card.interface";
import { Icon } from "./icon";
import { StatusPill } from "./status-pill";
import "./flow-card.css";

interface FlowCardProps extends Omit<HTMLAttributes<HTMLElement>, "title" | "onKeyDown"> {
  ref: Ref<HTMLElement>;
  id: string;
  icon: string;
  title: string;
  lesson: string;
  status: Status;
  task: string;
  foot: ReactNode;
  children?: ReactNode;
  actions: CardAction[];
  onCommand: (command: string) => void;
}

function activateOnKey(event: KeyboardEvent<HTMLElement>) {
  if (event.target !== event.currentTarget || (event.key !== "Enter" && event.key !== " ")) return;
  event.preventDefault();
  event.currentTarget.click();
}

export function FlowCard({ ref, id, className, icon, title, lesson, status, task, foot, children, actions, onCommand, ...article }: FlowCardProps) {
  return (
    <article ref={ref} className={className ? `card ${className}` : "card"} id={id} tabIndex={0} aria-label={title} {...article} onKeyDown={activateOnKey}>
      <div className="card-head">
        <span className="tile"><Icon name={icon} /></span>
        <div>
          <h3 className="card-title">{title}</h3>
          <p className="card-lesson">{lesson}</p>
        </div>
        <StatusPill status={status} />
      </div>
      <p className="card-task">{task}</p>
      {children}
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
    </article>
  );
}

import type { Ref } from "react";
import type { FlowStage, FlowStep } from "@/features/lessons/interfaces/story.interface";
import { FlowCard } from "@/shared/components/flow-card";
import { STAGGER_MS } from "@/shared/components/flow-links";
import type { Session } from "./session";

export const DRAWER_ID = "part-drawer";

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
  isOpen: boolean;
  onToggle: () => void;
  onCommand: (command: string) => void;
}

export function StageCard({ ref, stage, session, isSelected, onSelect, onCommand }: StageCardProps) {
  const action = stage.action?.(session);
  return (
    <FlowCard
      ref={ref}
      className={isSelected ? undefined : "is-dim"}
      id={`stage-${stage.id}`}
      aria-current={isSelected}
      onClick={onSelect}
      icon={stage.icon}
      title={stage.title}
      lesson={stage.lesson}
      status={stage.status(session)}
      task={stage.task}
      foot={<><b>{stage.value(session)}</b>{` · ${stage.detail(session)}`}</>}
      actions={action ? [action] : []}
      onCommand={onCommand}
    />
  );
}

export function PartCard({ ref, step, session, index, isNested, isActive, isOpen, onToggle, onCommand }: PartCardProps) {
  return (
    <FlowCard
      ref={ref}
      className={`is-entering${isNested ? " is-nested" : ""}${isActive ? " is-active" : ""}${isOpen ? " is-open" : ""}`}
      id={`part-${step.id}`}
      aria-expanded={isOpen}
      aria-controls={isOpen ? DRAWER_ID : undefined}
      style={{ animationDelay: `${index * STAGGER_MS}ms` }}
      onClick={onToggle}
      icon={step.icon}
      title={step.title}
      lesson={step.lesson}
      status={step.status(session)}
      task={step.goal}
      foot={<>{`${step.from(session)} → `}<b>{step.to(session)}</b></>}
      actions={step.actions?.(session) ?? []}
      onCommand={onCommand}
    />
  );
}

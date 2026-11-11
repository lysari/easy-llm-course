import type { ReactNode } from "react";
import type { Session } from "@/features/lessons/session";
import type { CardAction, Status } from "@/shared/interfaces/card.interface";
import type { ConceptName } from "@/shared/interfaces/concept.interface";
import type { Formula } from "@/shared/interfaces/formula.interface";

export type StageId = "simple" | "data" | "forward" | "learn" | "use" | "beyond";

export interface TaskRow {
  text: string;
  value?: string;
  command?: string;
  isOn?: boolean;
}

export interface FlowStep {
  id: string;
  stage: StageId;
  icon: ConceptName;
  title: string;
  lesson: string;
  goal: string;
  from: (session: Session) => string;
  to: (session: Session) => string;
  status: (session: Session) => Status;
  rowsTitle?: string;
  rows?: (session: Session) => TaskRow[];
  visual?: (session: Session) => ReactNode;
  note?: (session: Session) => ReactNode;
  formula?: Formula;
  code?: string;
  why: string;
  actions?: (session: Session) => CardAction[];
}

export interface FlowStage {
  id: StageId;
  icon: ConceptName;
  title: string;
  lesson: string;
  task: string;
  status: (session: Session) => Status;
  value: (session: Session) => string;
  detail: (session: Session) => string;
  action?: (session: Session) => CardAction | null;
}

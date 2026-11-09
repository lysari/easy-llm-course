export type Status = "running" | "done" | "needs" | "queued" | "lesson";

export interface CardAction {
  label: string;
  command: string;
  isStrong?: boolean;
}

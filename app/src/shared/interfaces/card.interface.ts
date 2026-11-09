export type Status = "running" | "done" | "needs" | "queued" | "lesson" | "ground-zero";

export interface CardAction {
  label: string;
  command: string;
  isStrong?: boolean;
}

import type { Status } from "@/shared/interfaces/card.interface";

export function rollUp(statuses: Status[]): Status {
  if (statuses.includes("running")) return "running";
  if (statuses.includes("needs")) return "needs";
  if (statuses.every((status) => status === "lesson")) return "lesson";
  return statuses.every((status) => status === "done" || status === "lesson") ? "done" : "queued";
}

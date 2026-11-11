import type { ReactNode } from "react";
import type { Status } from "@/shared/interfaces/card.interface";

const STATUS_TEXT: Record<Status, string> = { running: "Running", done: "Done", needs: "Your turn", queued: "Train first", lesson: "Read only", "ground-zero": "Smallest step" };
const STATUS_HINTS: Record<Status, string> = {
  running: "This is running right now, in your browser.",
  done: "Ready to explore: nothing to wait for or press first.",
  needs: "Waiting for you: press a button on this card or inside it to run it.",
  queued: "Train Tiny GPT first. This waits until its training is done.",
  lesson: "This is explained here, but not run in your browser.",
  "ground-zero": "Nothing smaller inside: one multiply, one stored number or one simple action.",
};
const STATUS_MARKS: Record<Status, ReactNode> = {
  running: <i className="spinner" />,
  done: "✓ ",
  needs: (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" aria-hidden="true">
      <circle cx="6" cy="6" r="5" />
      <path d="M6 3.2V6l1.8 1.2" />
    </svg>
  ),
  queued: "• ",
  lesson: "◦ ",
  "ground-zero": "◆ ",
};

export function StatusPill({ status }: { status: Status }) {
  return (
    <span className={`pill pill--${status}`} title={STATUS_HINTS[status]}>
      <span className="pill-mark" aria-hidden="true">{STATUS_MARKS[status]}</span>
      {STATUS_TEXT[status]}
      <span className="visually-hidden">{`. ${STATUS_HINTS[status]}`}</span>
    </span>
  );
}

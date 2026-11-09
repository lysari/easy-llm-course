import type { ReactNode } from "react";
import type { Status } from "@/shared/interfaces/card.interface";

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

export function StatusPill({ status }: { status: Status }) {
  return <span className={`pill pill--${status}`}>{STATUS_ICONS[status]}{STATUS_TEXT[status]}</span>;
}

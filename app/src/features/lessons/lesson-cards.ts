import type { CardAction, FlowStep, Status } from "@/features/lessons/interfaces/story.interface";
import { escapeHtml } from "@/shared/utils/html";
import { iconSvg } from "@/shared/utils/icons";
import type { Session } from "./session";

const STATUS_NAMES: Record<Status, string> = { running: "Running", done: "Done", needs: "Needs you", queued: "Queued", lesson: "Lesson only" };
const STATUS_MARKS: Record<Status, string> = {
  running: `<i class="spinner"></i>`,
  done: "✓",
  needs: `<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" aria-hidden="true"><circle cx="6" cy="6" r="5"/><path d="M6 3.2V6l1.8 1.2"/></svg>`,
  queued: "•",
  lesson: "◦",
};

export function cardHtml(card: { icon: string; title: string; lesson: string; status: Status; task: string; foot: string; actions: CardAction[] }) {
  const buttons = card.actions
    .map((action) => `<button type="button" class="action${action.isStrong ? " action--strong" : ""}" data-command="${escapeHtml(action.command)}">${escapeHtml(action.label)}</button>`)
    .join("");
  return `<div class="card-head">
      <span class="tile">${iconSvg(card.icon)}</span>
      <div><h3 class="card-title">${escapeHtml(card.title)}</h3><p class="card-lesson">${escapeHtml(card.lesson)}</p></div>
      <span class="pill pill--${card.status}">${STATUS_MARKS[card.status]} ${STATUS_NAMES[card.status]}</span>
    </div>
    <p class="card-task">${escapeHtml(card.task)}</p>
    <div class="card-foot">
      <span class="foot-value">${card.foot}</span>
      ${buttons ? `<span class="foot-actions">${buttons}</span>` : ""}
    </div>`;
}

export function stepCardHtml(step: FlowStep, session: Session) {
  return cardHtml({
    icon: step.icon,
    title: step.title,
    lesson: step.lesson,
    status: step.status(session),
    task: step.goal,
    foot: `${escapeHtml(step.from(session))} → <b>${escapeHtml(step.to(session))}</b>`,
    actions: step.actions?.(session) ?? [],
  });
}

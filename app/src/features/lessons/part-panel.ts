import type { FlowStep } from "@/features/lessons/interfaces/story.interface";
import { escapeHtml } from "@/shared/utils/html";
import type { Session } from "./session";

export function panelHtml(step: FlowStep, session: Session) {
  const rows = step.rows?.(session) ?? [];
  const items = rows.map(({ text, value, command, isOn }) => {
    const inner = `<span>${escapeHtml(text)}</span><span>${escapeHtml(value ?? "")}</span>`;
    return command
      ? `<button type="button" class="task${isOn ? " is-on" : ""}" data-command="${escapeHtml(command)}" aria-pressed="${Boolean(isOn)}">${inner}</button>`
      : `<div class="task">${inner}</div>`;
  });
  const note = step.note?.(session);
  return `${rows.length > 0 ? `<section class="section"><p class="section-title">${escapeHtml(step.rowsTitle ?? "Step by step")}</p><div class="tasks">${items.join("")}</div></section>` : ""}
    ${step.visual ? `<section class="section">${step.visual(session)}</section>` : ""}
    ${note ? `<p class="note">${note}</p>` : ""}
    ${step.formula ? `<section class="section"><p class="section-title">Formula</p><pre class="code">${escapeHtml(step.formula)}</pre></section>` : ""}
    ${step.code ? `<section class="section"><p class="section-title">Code</p><pre class="code">${escapeHtml(step.code)}</pre></section>` : ""}
    <p class="why">${escapeHtml(step.why)}</p>`;
}

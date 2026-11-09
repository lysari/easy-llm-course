import { STAGES } from "@/features/lessons/flow";
import { byId, escapeHtml } from "@/shared/utils/html";
import { SPARKLE_SVG, iconSvg } from "@/shared/utils/icons";

export function sidebarHtml() {
  return `<aside class="sidebar">
      <p class="brand">${SPARKLE_SVG} Tiny GPT</p>
      <nav class="side-nav" aria-label="Pages">
        <a class="side-link side-link--page" id="nav-lessons" href="#lessons">${iconSvg("book")} Lessons</a>
        <div class="side-stages" id="side-stages">
          ${STAGES.map((stage) => `<button type="button" class="side-link" id="nav-${stage.id}" data-stage="${stage.id}">${iconSvg(stage.icon)} ${escapeHtml(stage.title)}</button>`).join("")}
        </div>
        <a class="side-link side-link--page" id="nav-playground" href="#playground">${iconSvg("play")} Playground</a>
      </nav>
    </aside>`;
}

export function showSidebarPage(isPlayground: boolean) {
  byId("side-stages").hidden = isPlayground;
  for (const [id, isOn] of [["nav-lessons", !isPlayground], ["nav-playground", isPlayground]] as const) {
    byId(id).classList.toggle("is-on", isOn);
    if (isOn) byId(id).setAttribute("aria-current", "page");
    else byId(id).removeAttribute("aria-current");
  }
}

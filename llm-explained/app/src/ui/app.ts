import type { CardAction, FlowStep, StageId, Status } from "../interfaces/story.interface";
import { NESTED_STEPS, STAGES, STEPS, TRAINING_PATH, stepsOf } from "../story/flow";
import { escapeHtml } from "../story/pieces";
import { Session } from "../story/session";
import { SPARKLE_SVG, iconSvg } from "./icons";
import { createPlayground } from "./playground";

const SELECT_DELAY_MS = 120;
const CLOSE_DELAY_MS = 260;
const LEAVE_MS = 110;
const STAGGER_MS = 70;
const NARROW_WIDTH = 960;
const CORNER = 10;
const STEP_ORDER_MS = 450;
const STATUS_NAMES: Record<Status, string> = { running: "Running", done: "Done", needs: "Needs you", queued: "Queued", lesson: "Lesson only" };
const STATUS_MARKS: Record<Status, string> = {
  running: `<i class="spinner"></i>`,
  done: "✓",
  needs: `<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" aria-hidden="true"><circle cx="6" cy="6" r="5"/><path d="M6 3.2V6l1.8 1.2"/></svg>`,
  queued: "•",
  lesson: "◦",
};

const byId = (id: string) => {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing element: ${id}`);
  return element;
};

const setHtml = (element: Element, html: string) => {
  if (element.getAttribute("data-html") === html) return;
  element.setAttribute("data-html", html);
  element.innerHTML = html;
};

function cardHtml(card: { icon: string; title: string; lesson: string; status: Status; task: string; foot: string; actions: CardAction[] }) {
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

function stepCardHtml(step: FlowStep, session: Session) {
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

function panelHtml(step: FlowStep, session: Session) {
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

export function startApp() {
  const session = new Session();
  let stageId: StageId = "learn";
  let openStepId: string | null = null;
  let isPinned = false;
  let lastFrame = 0;
  let isLooping = false;
  let selectTimer = 0;
  let openTimer = 0;
  let closeTimer = 0;

  const isNarrow = () => window.innerWidth <= NARROW_WIDTH;
  const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
  const openStep = () => stepsOf(stageId).find((step) => step.id === openStepId);

  function renderShell() {
    byId("app").innerHTML = `<aside class="sidebar">
      <p class="brand">${SPARKLE_SVG} Tiny GPT</p>
      <nav class="side-nav" aria-label="Pages">
        <a class="side-link side-link--page" id="nav-lessons" href="#lessons">${iconSvg("book")} Lessons</a>
        <div class="side-stages" id="side-stages">
          ${STAGES.map((stage) => `<button type="button" class="side-link" id="nav-${stage.id}" data-stage="${stage.id}">${iconSvg(stage.icon)} ${escapeHtml(stage.title)}</button>`).join("")}
        </div>
        <a class="side-link side-link--page" id="nav-playground" href="#playground">${iconSvg("play")} Playground</a>
      </nav>
    </aside>
    <main class="page" id="lessons">
      <div class="fleet">
        <header class="top">
          <h1 class="page-title">Lessons</h1>
          <p class="top-stats" id="top-stats"></p>
        </header>
        <div class="flow" id="flow">
          <svg class="links" id="links" aria-hidden="true"></svg>
          <div class="column" id="stages">
            ${STAGES.map((stage) => `<article class="card" id="stage-${stage.id}" data-stage="${stage.id}" tabindex="0" aria-label="${escapeHtml(stage.title)}"></article>`).join("")}
          </div>
          <div class="column part-column" id="parts"></div>
        </div>
      </div>
    </main>
    <main class="page page--playground" id="playground" hidden></main>`;
  }

  function showPage() {
    const isPlayground = location.hash === "#playground";
    byId("lessons").hidden = isPlayground;
    byId("playground").hidden = !isPlayground;
    byId("side-stages").hidden = isPlayground;
    for (const [id, isOn] of [["nav-lessons", !isPlayground], ["nav-playground", isPlayground]] as const) {
      byId(id).classList.toggle("is-on", isOn);
      if (isOn) byId(id).setAttribute("aria-current", "page");
      else byId(id).removeAttribute("aria-current");
    }
    if (isPlayground) {
      closePanel();
      playground.open();
    } else {
      playground.close();
      drawLinks();
    }
  }

  function renderHeader() {
    const statuses = STEPS.map((step) => step.status(session));
    const running = statuses.filter((status) => status === "running").length;
    const needs = statuses.filter((status) => status === "needs").length;
    setHtml(byId("top-stats"), `<span>${STAGES.length} stages / ${STEPS.length} parts</span>
      <span>${running} running</span><span>${needs} need you</span>
      <strong>Loss ${(session.lab.latestLoss ?? session.startLoss).toFixed(2)}</strong>`);
  }

  function renderCards() {
    for (const stage of STAGES) {
      const element = byId(`stage-${stage.id}`);
      const action = stage.action?.(session);
      setHtml(element, cardHtml({
        icon: stage.icon,
        title: stage.title,
        lesson: stage.lesson,
        status: stage.status(session),
        task: stage.task,
        foot: `<b>${escapeHtml(stage.value(session))}</b> · ${escapeHtml(stage.detail(session))}`,
        actions: action ? [action] : [],
      }));
      element.classList.toggle("is-dim", stage.id !== stageId);
      element.setAttribute("aria-current", String(stage.id === stageId));
      byId(`nav-${stage.id}`).classList.toggle("is-selected", stage.id === stageId);
    }
    const activeIndex = session.isRunning("train") ? Math.floor(performance.now() / STEP_ORDER_MS) % TRAINING_PATH.length : -1;
    for (const step of stepsOf(stageId)) {
      const element = document.getElementById(`part-${step.id}`);
      if (!element) continue;
      setHtml(element, stepCardHtml(step, session));
      element.classList.toggle("is-active", activeIndex >= 0 && TRAINING_PATH[activeIndex] === step.id);
    }
  }

  function renderPanel() {
    document.getElementById("panel")?.remove();
    const step = openStep();
    if (!step) return;
    const card = byId(`part-${step.id}`);
    const panel = document.createElement("section");
    panel.className = "panel";
    panel.id = "panel";
    panel.setAttribute("aria-label", `${step.title} details`);
    panel.innerHTML = `<div class="panel-inner" id="panel-inner"></div>`;
    panel.style.top = `${card.offsetTop + card.offsetHeight + 8}px`;
    if (isNarrow()) card.after(panel);
    else byId("parts").append(panel);
    panel.style.left = NESTED_STEPS.includes(step.id) ? `${card.offsetLeft + card.offsetWidth - panel.offsetWidth}px` : `${card.offsetLeft}px`;
    setHtml(byId("panel-inner"), panelHtml(step, session));
  }

  function refreshPanelBody() {
    const inner = document.getElementById("panel-inner");
    const step = openStep();
    if (inner && step) setHtml(inner, panelHtml(step, session));
  }

  function pathBetween(x1: number, y1: number, trunk: number, x2: number, y2: number) {
    if (Math.abs(y2 - y1) < 1) return `M ${x1} ${y1} H ${x2}`;
    const r = Math.min(CORNER, Math.abs(y2 - y1) / 2);
    const direction = y2 > y1 ? 1 : -1;
    return `M ${x1} ${y1} H ${trunk - r} Q ${trunk} ${y1} ${trunk} ${y1 + direction * r} V ${y2 - direction * r} Q ${trunk} ${y2} ${trunk + r} ${y2} H ${x2}`;
  }

  function drawLinks() {
    const links = byId("links");
    if (isNarrow()) {
      links.innerHTML = "";
      return;
    }
    const flow = byId("flow").getBoundingClientRect();
    const parent = byId(`stage-${stageId}`).getBoundingClientRect();
    const startX = parent.right - flow.left;
    const startY = parent.top + parent.height / 2 - flow.top;
    const trunk = (startX + byId("parts").getBoundingClientRect().left - flow.left) / 2;
    const paths: string[] = [];
    const dots: string[] = [`<circle class="link-dot" cx="${startX}" cy="${startY}" r="4"/>`];
    let previous: DOMRect | null = null;
    let delay = 0;
    for (const step of stepsOf(stageId)) {
      const box = byId(`part-${step.id}`).getBoundingClientRect();
      const y2 = box.top + box.height / 2 - flow.top;
      const endX = box.left - flow.left;
      if (NESTED_STEPS.includes(step.id) && previous) {
        const x = previous.right - flow.left;
        paths.push(`<path class="link-path" pathLength="1" style="animation-delay:${delay}ms" d="M ${x} ${y2} H ${endX}"/>`);
        dots.push(`<circle class="link-dot" cx="${x}" cy="${y2}" r="4"/>`);
      } else {
        paths.push(`<path class="link-path" pathLength="1" style="animation-delay:${delay}ms" d="${pathBetween(startX, startY, trunk, endX, y2)}"/>`);
      }
      previous = box;
      delay += STAGGER_MS;
    }
    links.innerHTML = paths.join("") + dots.join("");
  }

  function placeParts() {
    if (isNarrow()) byId(`stage-${stageId}`).after(byId("parts"));
    else byId("flow").append(byId("parts"));
  }

  function renderParts() {
    const column = byId("parts");
    column.classList.remove("is-leaving");
    placeParts();
    const steps = stepsOf(stageId);
    const partCard = (step: FlowStep) =>
      `<article class="card is-entering${NESTED_STEPS.includes(step.id) ? " is-nested" : ""}" id="part-${step.id}" data-part="${step.id}" tabindex="0" aria-label="${escapeHtml(step.title)}" style="animation-delay:${steps.indexOf(step) * STAGGER_MS}ms"></article>`;
    column.innerHTML = steps
      .map((step, i) => {
        if (NESTED_STEPS.includes(step.id)) return "";
        const next = steps[i + 1];
        const subCard = next && NESTED_STEPS.includes(next.id) ? partCard(next) : "";
        return `<div class="part-row">${partCard(step)}${subCard}</div>`;
      })
      .join("");
    renderCards();
    drawLinks();
  }

  function selectStage(id: StageId) {
    if (id === stageId) return;
    stageId = id;
    openStepId = null;
    isPinned = false;
    document.getElementById("panel")?.remove();
    byId("links").innerHTML = "";
    renderCards();
    if (reducedMotion()) {
      renderParts();
      return;
    }
    byId("parts").classList.add("is-leaving");
    window.setTimeout(renderParts, LEAVE_MS);
  }

  function openPanel(id: string, pin: boolean) {
    window.clearTimeout(closeTimer);
    isPinned = pin || (isPinned && id === openStepId);
    if (openStepId === id && document.getElementById("panel")) return;
    openStepId = id;
    renderPanel();
  }

  function closePanel() {
    isPinned = false;
    openStepId = null;
    renderPanel();
  }

  function closePanelSoon() {
    if (isPinned) return;
    window.clearTimeout(closeTimer);
    closeTimer = window.setTimeout(closePanel, CLOSE_DELAY_MS);
  }

  function renderAll() {
    renderHeader();
    renderCards();
    refreshPanelBody();
  }

  function frame(now: number) {
    if (!session.hasRunningJobs) {
      isLooping = false;
      renderAll();
      return;
    }
    session.tick((now - lastFrame) / 1000);
    lastFrame = now;
    renderAll();
    requestAnimationFrame(frame);
  }

  function ensureLoop() {
    if (isLooping || !session.hasRunningJobs) return;
    isLooping = true;
    lastFrame = performance.now();
    requestAnimationFrame(frame);
  }

  document.addEventListener("click", (event) => {
    const target = event.target as HTMLElement;
    const command = target.closest<HTMLElement>("[data-command]");
    if (command) {
      session.runCommand(command.dataset.command as string);
      renderAll();
      ensureLoop();
      return;
    }
    const stageCard = target.closest<HTMLElement>("[data-stage]");
    if (stageCard) {
      selectStage(stageCard.dataset.stage as StageId);
      return;
    }
    const partCard = target.closest<HTMLElement>("[data-part]");
    if (partCard) {
      const id = partCard.dataset.part as string;
      if (isPinned && openStepId === id) closePanel();
      else openPanel(id, true);
      return;
    }
    if (!target.closest("#panel") && isPinned) closePanel();
  });

  document.addEventListener("keydown", (event) => {
    const target = event.target as HTMLElement;
    if (event.key === "Escape" && openStepId) {
      closePanel();
      return;
    }
    if ((event.key !== "Enter" && event.key !== " ") || !target.matches(".card")) return;
    event.preventDefault();
    target.click();
  });

  document.addEventListener("click", (event) => {
    if (matchMedia("(hover: none)").matches) return;
    const target = event.target as HTMLElement;
    const stageCard = target.closest<HTMLElement>("[data-stage]");
    if (stageCard) {
      window.clearTimeout(selectTimer);
      selectTimer = window.setTimeout(() => selectStage(stageCard.dataset.stage as StageId), SELECT_DELAY_MS);
      return;
    }
    window.clearTimeout(selectTimer);
    const partCard = target.closest<HTMLElement>("[data-part]");
    if (partCard) {
      window.clearTimeout(openTimer);
      window.clearTimeout(closeTimer);
      if (isPinned && openStepId !== partCard.dataset.part) return;
      openTimer = window.setTimeout(() => openPanel(partCard.dataset.part as string, false), SELECT_DELAY_MS);
      return;
    }
    window.clearTimeout(openTimer);
    if (target.closest("#panel")) {
      window.clearTimeout(closeTimer);
      return;
    }
    closePanelSoon();
  });

  window.addEventListener("resize", () => {
    placeParts();
    drawLinks();
    renderPanel();
  });

  renderShell();
  renderHeader();
  renderParts();
  const playground = createPlayground(byId("playground"));
  window.addEventListener("hashchange", showPage);
  showPage();
}

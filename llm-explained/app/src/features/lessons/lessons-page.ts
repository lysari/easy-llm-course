import "./lessons.css";
import type { FlowStep, StageId } from "@/features/lessons/interfaces/story.interface";
import { byId, escapeHtml, setHtml } from "@/shared/utils/html";
import { NESTED_STEPS, STAGES, STEPS, TRAINING_PATH, stepsOf } from "./flow";
import { STAGGER_MS, drawLinks, isNarrow } from "./flow-links";
import { cardHtml, stepCardHtml } from "./lesson-cards";
import { panelHtml } from "./part-panel";
import { Session } from "./session";

const SELECT_DELAY_MS = 120;
const CLOSE_DELAY_MS = 260;
const LEAVE_MS = 150;
const STEP_ORDER_MS = 450;

export function lessonsPageHtml() {
  return `<main class="page" id="lessons">
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
    </main>`;
}

export function createLessonsPage() {
  const session = new Session();
  let stageId: StageId = "learn";
  let openStepId: string | null = null;
  let isPinned = false;
  let lastFrame = 0;
  let isLooping = false;
  let selectTimer = 0;
  let openTimer = 0;
  let closeTimer = 0;

  const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
  const openStep = () => stepsOf(stageId).find((step) => step.id === openStepId);

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
    drawLinks(stageId);
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
      if (!target.closest("[data-command]") || (isPinned && openStepId !== partCard.dataset.part)) return;
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
    drawLinks(stageId);
    renderPanel();
  });

  renderHeader();
  renderParts();

  return {
    open() {
      drawLinks(stageId);
    },
    close: closePanel,
  };
}

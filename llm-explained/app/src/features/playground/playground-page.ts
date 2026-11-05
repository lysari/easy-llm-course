import "./playground.css";
import { START_CHOICES } from "@/core/model/corpus";
import type { PlaygroundState, RowKey } from "@/features/playground/interfaces/playground.interface";
import { setHtml } from "@/shared/utils/html";
import { builderHtml } from "./builder";
import { createMenu } from "./menu";
import { PlaygroundRun, defaultSettings } from "./playground-run";
import { resultHtml } from "./result";

const FRAME_BUDGET_MS = 12;
const START_WORDS = START_CHOICES[0];

export function createPlayground(root: HTMLElement) {
  const state: PlaygroundState = {
    settings: defaultSettings(START_WORDS),
    addedRows: [],
    isAutoTrain: true,
    run: null,
    isStale: true,
    earlier: [],
    earlierLoss: null,
    writeSeed: 1,
  };
  let isLooping = false;

  function renderBuilder() {
    root.querySelector("#pg-builder")!.innerHTML = builderHtml(state);
  }

  function renderResult() {
    setHtml(root.querySelector("#pg-result")!, resultHtml(state));
  }

  function frame() {
    state.run?.trainFor(FRAME_BUDGET_MS);
    renderResult();
    if (state.run && !state.run.isDone) requestAnimationFrame(frame);
    else isLooping = false;
  }

  function startTraining() {
    if (state.run?.isDone && !state.run.isBroken) {
      state.earlier = state.run.history.map((point) => ({ ...point }));
      state.earlierLoss = state.run.latestLoss;
    }
    state.run = new PlaygroundRun(state.settings);
    state.isStale = false;
    if (isLooping) return;
    isLooping = true;
    requestAnimationFrame(frame);
  }

  function changed(retrains: boolean) {
    renderBuilder();
    if (retrains) state.isStale = true;
    if (retrains && state.isAutoTrain) startTraining();
    renderResult();
  }

  const menu = createMenu(root, state, changed);

  function runAction(action: string) {
    const [name, value] = action.split(":");
    if (name === "train") startTraining();
    if (name === "write-again") {
      state.writeSeed++;
      renderResult();
    }
    if (name === "reset") {
      state.settings = defaultSettings(START_WORDS);
      state.addedRows = [];
      changed(true);
    }
    if (name === "toggle") {
      state.isAutoTrain = !state.isAutoTrain;
      const toggle = root.querySelector("#pg-toggle")!;
      toggle.setAttribute("aria-checked", String(state.isAutoTrain));
      if (state.isAutoTrain && state.isStale) startTraining();
      renderResult();
    }
    if (name === "remove-text") {
      state.settings.textGroups = state.settings.textGroups.filter((group) => group !== value);
      changed(true);
    }
    if (name === "add-layer") {
      state.settings.layers = [...state.settings.layers, { hasAttention: true, hasFeedForward: true }];
      changed(true);
    }
    if (name === "remove-layer") {
      state.settings.layers = state.settings.layers.filter((_, i) => i !== Number(value));
      changed(true);
    }
    if (name === "remove-row") {
      const key = value as RowKey;
      state.addedRows = state.addedRows.filter((row) => row !== key);
      state.settings = { ...state.settings, [key]: defaultSettings(START_WORDS)[key] };
      changed(false);
    }
  }

  root.innerHTML = `<div class="pg">
    <div class="pg-main">
      <header class="pg-head">
        <div>
          <h1 class="page-title">Playground</h1>
          <p class="pg-sub">A real model trains in your browser · Change a setting and see what happens</p>
        </div>
        <label class="pg-switch"><span id="pg-switch-label">Auto train</span>
          <button type="button" class="switch" id="pg-toggle" role="switch" aria-checked="true" aria-labelledby="pg-switch-label" data-do="toggle"></button>
        </label>
      </header>
      <div id="pg-builder"></div>
    </div>
    <aside class="pg-result" id="pg-result" aria-label="Result"></aside>
  </div>`;

  root.addEventListener("click", (event) => {
    const target = event.target as HTMLElement;
    if (target.closest("#pg-menu")) return;
    menu.toggle(target.closest<HTMLElement>("[data-menu]"));
    const action = target.closest<HTMLElement>("[data-do]");
    if (action) runAction(action.dataset.do as string);
  });

  renderBuilder();
  renderResult();

  return {
    open() {
      if (!state.run) startTraining();
    },
    close() {
      menu.close();
    },
  };
}

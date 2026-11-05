import type { LossPoint } from "../interfaces/model.interface";
import { joinTokens } from "../model/corpus";
import { applySampling, nextOptions, writeText } from "../model/generate";
import { GUESSING_LOSS, PlaygroundRun, TEXT_GROUPS, defaultSettings } from "../model/playground";
import type { PlaygroundSettings } from "../model/playground";
import { seededRandom } from "../model/random";
import { chanceBars, count, escapeHtml, lossPlot, showWord } from "../story/pieces";
import { START_CHOICES } from "../story/session";
import { iconSvg } from "./icons";

type Tint = "green" | "orange" | "blue" | "grey";
type RowKey = "isEvenAnswers" | "vectorSize" | "rounds" | "learningRate" | "optimizer" | "start" | "temperature" | "topK" | "topP";

interface Row {
  icon: string;
  tint: Tint;
  label: string;
  verb: string;
  lesson: string;
  help: string;
  retrains: boolean;
  options: { label: string; value: PlaygroundSettings[RowKey] }[];
}

interface MenuItem {
  label: string;
  isOn: boolean;
  apply: () => void;
}

const FRAME_BUDGET_MS = 12;
const MAX_LAYERS = 4;
const PLOT_SIZE = { width: 400, height: 220 };
const PLOT_CEILING = GUESSING_LOSS * 3;
const START_WORDS = START_CHOICES[0];
const OPTIONAL_ROWS: RowKey[] = ["temperature", "topK", "topP"];

const numbered = (values: number[], unit: string) => values.map((value) => ({ label: `${value} ${unit}`, value }));

const ROWS: Record<RowKey, Row> = {
  isEvenAnswers: {
    icon: "order", tint: "orange", label: "Usual answer", verb: "appears", lesson: "Lesson 14", retrains: true,
    help: "How often the text uses its usual answer. The model copies this pattern.",
    options: [{ label: "3 times more", value: false }, { label: "Just as often", value: true }],
  },
  vectorSize: {
    icon: "grid", tint: "blue", label: "Each word", verb: "is", lesson: "Lesson 10", retrains: true,
    help: "How many numbers describe each word. More numbers hold more, but train slower.",
    options: numbered([6, 12, 24, 48], "numbers"),
  },
  rounds: {
    icon: "repeat", tint: "blue", label: "Repeat", verb: "for", lesson: "Lesson 14", retrains: true,
    help: "One round is: guess, score, find the cause, fix.",
    options: numbered([20, 50, 200, 500], "rounds"),
  },
  learningRate: {
    icon: "down", tint: "orange", label: "Step size", verb: "is", lesson: "Lesson 03", retrains: true,
    help: "How far the numbers move in one round. Too small learns slowly. Too big jumps past the answer.",
    options: [0.0004, 0.004, 0.04, 0.4].map((value) => ({ label: String(value), value })),
  },
  optimizer: {
    icon: "sliders", tint: "grey", label: "Fix numbers", verb: "with", lesson: "Lesson 17", retrains: true,
    help: "Adam gives every number its own step size. Plain steps use one size for all.",
    options: [{ label: "Adam", value: "adam" }, { label: "Plain steps", value: "plain" }],
  },
  start: {
    icon: "pen", tint: "green", label: "Start with", verb: "", lesson: "Lesson 14", retrains: false,
    help: "The words you give the model. It continues from here.",
    options: START_CHOICES.map((value) => ({ label: joinTokens(value.split(" ")), value })),
  },
  temperature: {
    icon: "gauge", tint: "orange", label: "Temperature", verb: "is", lesson: "Lesson 23", retrains: false,
    help: "Low picks the top word more often. High takes more risks.",
    options: [0.5, 1, 2].map((value) => ({ label: String(value), value })),
  },
  topK: {
    icon: "hash", tint: "blue", label: "Top-K", verb: "keeps", lesson: "Lesson 23", retrains: false,
    help: "Only pick from this many top words.",
    options: [{ label: "All words", value: 0 }, ...numbered([2, 3, 5], "words")],
  },
  topP: {
    icon: "percent", tint: "grey", label: "Top-P", verb: "keeps", lesson: "Lesson 23", retrains: false,
    help: "Only pick from the top words that add up to this chance.",
    options: [1, 0.9, 0.75, 0.5].map((value) => ({ label: `${Math.round(value * 100)}%`, value })),
  },
};

const chevron = `<span class="chip-chevron">${iconSvg("chevron")}</span>`;
const tile = (icon: string, tint: Tint) => `<span class="chip-tile chip-tile--${tint}">${iconSvg(icon)}</span>`;
const label = (icon: string, tint: Tint, text: string) => `<span class="chip">${tile(icon, tint)} ${escapeHtml(text)}</span>`;
const valueChip = (menu: string, text: string, isMono = false) =>
  `<button type="button" class="chip chip--button${isMono ? " chip--wide" : ""}" data-menu="${menu}" aria-haspopup="menu" aria-expanded="false"><span class="chip-text">${escapeHtml(text)}</span>${chevron}</button>`;
const bracket = (children: string[]) =>
  `<div class="rules${children.length > 1 ? " rules--joined" : ""}">${children.length > 1 ? `<span class="and-pill">And</span>` : ""}${children.join("")}</div>`;
const addButton = (attribute: string, icon: string, text: string, isDisabled = false) =>
  `<button type="button" class="add-button" ${attribute}${isDisabled ? " disabled" : ""}>${iconSvg(icon)} ${text}</button>`;

export function createPlayground(root: HTMLElement) {
  let settings = defaultSettings(START_WORDS);
  let addedRows: RowKey[] = [];
  let isAutoTrain = true;
  let run: PlaygroundRun | null = null;
  let isStale = true;
  let earlier: LossPoint[] = [];
  let earlierLoss: number | null = null;
  let writeSeed = 1;
  let isLooping = false;
  let openMenu: string | null = null;

  const rowValue = (key: RowKey) => settings[key];
  const optionLabel = (key: RowKey) => ROWS[key].options.find((option) => option.value === rowValue(key))?.label ?? "";

  function ruleRow(key: RowKey, isRemovable = false) {
    const row = ROWS[key];
    return `<div class="rule">
      ${label(row.icon, row.tint, row.label)}
      ${row.verb ? `<span class="rule-word">${row.verb}</span>` : ""}
      ${valueChip(`set:${key}`, optionLabel(key), key === "start")}
      <span class="rule-end">
        <span class="rule-lesson">${row.lesson}</span>
        ${isRemovable ? `<button type="button" class="icon-button" data-do="remove-row:${key}" aria-label="Remove ${escapeHtml(row.label)}">${iconSvg("trash")}</button>` : ""}
      </span>
    </div>`;
  }

  function textRow() {
    const canRemove = settings.textGroups.length > 1;
    const tags = settings.textGroups
      .map((name) => `<span class="tag">${escapeHtml(name)}${canRemove ? `<button type="button" class="tag-remove" data-do="remove-text:${name}" aria-label="Remove ${escapeHtml(name)}">${iconSvg("close")}</button>` : ""}</span>`)
      .join("");
    const canAdd = settings.textGroups.length < TEXT_GROUPS.length;
    return `<div class="rule">
      ${label("text", "green", "Learn from")}
      <span class="tags">${tags}${canAdd ? `<button type="button" class="icon-button" data-menu="add-text" aria-haspopup="menu" aria-expanded="false" aria-label="Add text">${iconSvg("plus")}</button>` : ""}</span>
      <span class="rule-end"><span class="rule-lesson">Lesson 09</span></span>
    </div>`;
  }

  function layerGroup(index: number) {
    const layer = settings.layers[index];
    const parts = Number(layer.hasAttention) + Number(layer.hasFeedForward);
    const partRow = (part: "attention" | "feed", icon: string, tint: Tint, text: string, isOn: boolean, lesson: string) => `<div class="rule">
        ${label(icon, tint, text)}
        <span class="rule-word">is</span>
        ${valueChip(`layer:${index}:${part}`, isOn ? "On" : "Off")}
        <span class="rule-end"><span class="rule-lesson">${lesson}</span></span>
      </div>`;
    return `<section class="rule-group" aria-label="Layer ${index + 1}">
      <header class="rule-group-head">
        <span><b>Layer ${index + 1}</b> · ${parts} of 2 parts</span>
        <button type="button" class="icon-button" data-do="remove-layer:${index}" aria-label="Remove layer ${index + 1}">${iconSvg("trash")}</button>
      </header>
      ${bracket([partRow("attention", "eye", "blue", "Attention", layer.hasAttention, "Lesson 12"), partRow("feed", "merge", "orange", "Feed-forward", layer.hasFeedForward, "Lesson 15")])}
    </section>`;
  }

  function renderBuilder() {
    const unusedRows = OPTIONAL_ROWS.filter((key) => !addedRows.includes(key));
    const layers = settings.layers.map((_, i) => layerGroup(i));
    const useRows = [ruleRow("start"), ...addedRows.map((key) => ruleRow(key, true))];
    root.querySelector("#pg-builder")!.innerHTML = `<p class="pg-section">Build</p>
      ${bracket([textRow(), ruleRow("isEvenAnswers"), ruleRow("vectorSize"), ...layers])}
      <div class="add-row">${addButton(`data-do="add-layer"`, "layers", "Add layer", settings.layers.length >= MAX_LAYERS)}</div>
      <p class="pg-section">Run</p>
      <p class="pg-badge pg-badge--orange">${iconSvg("repeat")} While training</p>
      ${bracket([ruleRow("rounds"), ruleRow("learningRate"), ruleRow("optimizer")])}
      <p class="pg-badge pg-badge--green">${iconSvg("check")} After training</p>
      ${bracket(useRows)}
      <div class="add-row">${unusedRows.length > 0 ? addButton(`data-menu="add-row" aria-haspopup="menu" aria-expanded="false"`, "plus", "Add setting") : ""}</div>`;
  }

  function statusPill() {
    if (run?.isBroken) return `<span class="pill pill--needs">Numbers blew up</span>`;
    if (isStale && !isAutoTrain) return `<span class="pill pill--needs">Settings changed</span>`;
    if (!run) return `<span class="pill pill--queued">• Not trained</span>`;
    return run.isDone ? `<span class="pill pill--done">✓ Trained</span>` : `<span class="pill pill--running"><i class="spinner"></i> Training</span>`;
  }

  function resultNote(activeRun: PlaygroundRun) {
    const loss = activeRun.latestLoss ?? GUESSING_LOSS;
    if (activeRun.isBroken) return "The numbers grew too big to count. Try a smaller step size.";
    if (loss >= GUESSING_LOSS) return "Worse than random guessing, so it has learned nothing useful yet.";
    return `Loss went from ${activeRun.history[0].loss.toFixed(2)} to ${loss.toFixed(2)}. Lower means it matches the text better.`;
  }

  function renderResult() {
    const rounds = Math.max(settings.rounds, earlier.length);
    const history = (run?.history ?? []).filter((point) => Number.isFinite(point.loss)).map((point) => ({ ...point, loss: Math.min(point.loss, PLOT_CEILING) }));
    const canPredict = run && run.history.length > 0 && !run.isBroken;
    const words = settings.start.split(" ");
    const ranked = canPredict ? applySampling(nextOptions(words, run!.weights), settings) : [];
    const written = canPredict && run!.isDone ? joinTokens(writeText(words, run!.weights, seededRandom(writeSeed), settings)) : null;
    const lastWord = words[words.length - 1];
    const html = `<header class="pg-result-head"><h2>Result</h2>${statusPill()}</header>
      <dl class="pg-stats">
        <div><dt>Loss</dt><dd>${run?.latestLoss != null && Number.isFinite(run.latestLoss) ? run.latestLoss.toFixed(2) : "–"}</dd></div>
        <div><dt>Last run</dt><dd>${earlierLoss === null ? "–" : earlierLoss.toFixed(2)}</dd></div>
        <div><dt>Round</dt><dd>${run ? run.history.length : 0}/${settings.rounds}</dd></div>
        <div><dt>Numbers</dt><dd>${run ? count(run.numberCount) : "–"}</dd></div>
      </dl>
      ${lossPlot(history, rounds, GUESSING_LOSS, { earlier, ...PLOT_SIZE })}
      ${earlier.length > 0 ? `<p class="pg-legend"><span class="pg-key pg-key--now">This run</span><span class="pg-key pg-key--earlier">Last run</span></p>` : ""}
      ${run ? `<p class="note">${resultNote(run)}</p>` : ""}
      <section class="section"><p class="section-title">Word after “${escapeHtml(showWord(lastWord))}”</p>${ranked.length > 0 ? chanceBars(ranked, ranked[0].token) : `<p class="note">Train the model first.</p>`}</section>
      <section class="section"><p class="section-title">It writes</p><p class="quote">${written ? escapeHtml(written) : "Waiting for training to finish."}</p></section>
      <footer class="pg-actions">
        <button type="button" class="action action--strong" data-do="train"${isStale ? "" : " disabled"}>Train</button>
        <button type="button" class="action" data-do="write-again"${written ? "" : " disabled"}>Write again</button>
        <button type="button" class="action" data-do="reset">Reset all</button>
      </footer>`;
    const element = root.querySelector("#pg-result")!;
    if (element.getAttribute("data-html") === html) return;
    element.setAttribute("data-html", html);
    element.innerHTML = html;
  }

  function frame() {
    run?.trainFor(FRAME_BUDGET_MS);
    renderResult();
    if (run && !run.isDone) requestAnimationFrame(frame);
    else isLooping = false;
  }

  function startTraining() {
    if (run?.isDone && !run.isBroken) {
      earlier = run.history.map((point) => ({ ...point }));
      earlierLoss = run.latestLoss;
    }
    run = new PlaygroundRun(settings);
    isStale = false;
    if (isLooping) return;
    isLooping = true;
    requestAnimationFrame(frame);
  }

  function changed(retrains: boolean) {
    renderBuilder();
    if (retrains) isStale = true;
    if (retrains && isAutoTrain) startTraining();
    renderResult();
  }

  function menuFor(id: string): { help: string; items: MenuItem[] } {
    const [kind, first, second] = id.split(":");
    if (kind === "set") {
      const key = first as RowKey;
      const row = ROWS[key];
      return {
        help: row.help,
        items: row.options.map((option) => ({
          label: option.label,
          isOn: option.value === rowValue(key),
          apply: () => {
            settings = { ...settings, [key]: option.value };
            changed(row.retrains);
          },
        })),
      };
    }
    if (kind === "layer") {
      const part = second === "attention" ? "hasAttention" : "hasFeedForward";
      const index = Number(first);
      const pick = (isOn: boolean) => ({
        label: isOn ? "On" : "Off",
        isOn: settings.layers[index][part] === isOn,
        apply: () => {
          settings.layers = settings.layers.map((layer, i) => (i === index ? { ...layer, [part]: isOn } : layer));
          changed(true);
        },
      });
      return {
        help: second === "attention" ? "Lets each word look back at earlier words." : "Lets each word think alone about what it collected.",
        items: [pick(true), pick(false)],
      };
    }
    if (kind === "add-text") {
      return {
        help: "Add sentences for the model to learn from.",
        items: TEXT_GROUPS.filter((group) => !settings.textGroups.includes(group.name)).map((group) => ({
          label: group.name,
          isOn: false,
          apply: () => {
            settings.textGroups = TEXT_GROUPS.map((item) => item.name).filter((name) => settings.textGroups.includes(name) || name === group.name);
            changed(true);
          },
        })),
      };
    }
    return {
      help: "Change how the model picks each word.",
      items: OPTIONAL_ROWS.filter((key) => !addedRows.includes(key)).map((key) => ({
        label: ROWS[key].label,
        isOn: false,
        apply: () => {
          addedRows = [...addedRows, key];
          changed(false);
        },
      })),
    };
  }

  function closeMenu(returnFocus = false) {
    if (!openMenu) return;
    const chip = root.querySelector<HTMLElement>(`[data-menu="${openMenu}"]`);
    chip?.setAttribute("aria-expanded", "false");
    root.querySelector("#pg-menu")?.remove();
    openMenu = null;
    if (returnFocus) chip?.focus();
  }

  function showMenu(chip: HTMLElement) {
    const id = chip.dataset.menu as string;
    openMenu = id;
    chip.setAttribute("aria-expanded", "true");
    const { help, items } = menuFor(id);
    const menu = document.createElement("div");
    menu.className = "pg-menu";
    menu.id = "pg-menu";
    menu.setAttribute("role", "menu");
    menu.innerHTML = `<p class="pg-menu-help">${escapeHtml(help)}</p>${items
      .map((item, i) => `<button type="button" class="pg-menu-item" role="menuitemradio" aria-checked="${item.isOn}" data-item="${i}"><span>${escapeHtml(item.label)}</span>${item.isOn ? iconSvg("check") : ""}</button>`)
      .join("")}`;
    menu.addEventListener("click", (event) => {
      const button = (event.target as HTMLElement).closest<HTMLElement>("[data-item]");
      if (!button) return;
      closeMenu();
      items[Number(button.dataset.item)].apply();
      root.querySelector<HTMLElement>(`[data-menu="${id}"]`)?.focus();
    });
    root.append(menu);
    const box = chip.getBoundingClientRect();
    const margin = 8;
    const fitsBelow = box.bottom + margin + menu.offsetHeight < window.innerHeight;
    menu.style.left = `${Math.max(margin, Math.min(box.left, window.innerWidth - menu.offsetWidth - margin))}px`;
    menu.style.top = `${fitsBelow ? box.bottom + margin : Math.max(margin, box.top - margin - menu.offsetHeight)}px`;
    (menu.querySelector<HTMLElement>('[aria-checked="true"]') ?? menu.querySelector<HTMLElement>("[data-item]"))?.focus();
  }

  function runAction(action: string) {
    const [name, value] = action.split(":");
    if (name === "train") startTraining();
    if (name === "write-again") {
      writeSeed++;
      renderResult();
    }
    if (name === "reset") {
      settings = defaultSettings(START_WORDS);
      addedRows = [];
      changed(true);
    }
    if (name === "toggle") {
      isAutoTrain = !isAutoTrain;
      const toggle = root.querySelector("#pg-toggle")!;
      toggle.setAttribute("aria-checked", String(isAutoTrain));
      if (isAutoTrain && isStale) startTraining();
      renderResult();
    }
    if (name === "remove-text") {
      settings.textGroups = settings.textGroups.filter((group) => group !== value);
      changed(true);
    }
    if (name === "add-layer") {
      settings.layers = [...settings.layers, { hasAttention: true, hasFeedForward: true }];
      changed(true);
    }
    if (name === "remove-layer") {
      settings.layers = settings.layers.filter((_, i) => i !== Number(value));
      changed(true);
    }
    if (name === "remove-row") {
      const key = value as RowKey;
      addedRows = addedRows.filter((row) => row !== key);
      settings = { ...settings, [key]: defaultSettings(START_WORDS)[key] };
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
    const chip = target.closest<HTMLElement>("[data-menu]");
    if (target.closest("#pg-menu")) return;
    const wasOpen = openMenu;
    closeMenu();
    if (chip && wasOpen !== chip.dataset.menu) showMenu(chip);
    const action = target.closest<HTMLElement>("[data-do]");
    if (action) runAction(action.dataset.do as string);
  });

  root.addEventListener("scroll", () => closeMenu());
  window.addEventListener("resize", () => closeMenu());

  document.addEventListener("click", (event) => {
    if (openMenu && !root.contains(event.target as Node)) closeMenu();
  });

  root.addEventListener("keydown", (event) => {
    if (!openMenu) return;
    if (event.key === "Escape") closeMenu(true);
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    const items = [...root.querySelectorAll<HTMLElement>("[data-item]")];
    const step = event.key === "ArrowDown" ? 1 : -1;
    items[(items.indexOf(document.activeElement as HTMLElement) + step + items.length) % items.length].focus();
  });

  renderBuilder();
  renderResult();

  return {
    open() {
      if (!run) startTraining();
    },
    close() {
      closeMenu();
    },
  };
}

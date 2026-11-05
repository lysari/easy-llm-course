import { START_CHOICES, joinTokens } from "@/core/model/corpus";
import type { PlaygroundSettings, PlaygroundState, RowKey } from "@/features/playground/interfaces/playground.interface";
import { escapeHtml } from "@/shared/utils/html";
import { iconSvg } from "@/shared/utils/icons";
import { TEXT_GROUPS } from "./playground-run";

type Tint = "green" | "orange" | "blue" | "grey";

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

const MAX_LAYERS = 4;
export const OPTIONAL_ROWS: RowKey[] = ["temperature", "topK", "topP"];

const numbered = (values: number[], unit: string) => values.map((value) => ({ label: `${value} ${unit}`, value }));

export const ROWS: Record<RowKey, Row> = {
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

function optionLabel(settings: PlaygroundSettings, key: RowKey) {
  return ROWS[key].options.find((option) => option.value === settings[key])?.label ?? "";
}

function ruleRow(settings: PlaygroundSettings, key: RowKey, isRemovable = false) {
  const row = ROWS[key];
  return `<div class="rule">
      ${label(row.icon, row.tint, row.label)}
      ${row.verb ? `<span class="rule-word">${row.verb}</span>` : ""}
      ${valueChip(`set:${key}`, optionLabel(settings, key), key === "start")}
      <span class="rule-end">
        <span class="rule-lesson">${row.lesson}</span>
        ${isRemovable ? `<button type="button" class="icon-button" data-do="remove-row:${key}" aria-label="Remove ${escapeHtml(row.label)}">${iconSvg("trash")}</button>` : ""}
      </span>
    </div>`;
}

function textRow(settings: PlaygroundSettings) {
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

function layerGroup(settings: PlaygroundSettings, index: number) {
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

export function builderHtml({ settings, addedRows }: PlaygroundState) {
  const unusedRows = OPTIONAL_ROWS.filter((key) => !addedRows.includes(key));
  const layers = settings.layers.map((_, i) => layerGroup(settings, i));
  const useRows = [ruleRow(settings, "start"), ...addedRows.map((key) => ruleRow(settings, key, true))];
  return `<p class="pg-section">Build</p>
      ${bracket([textRow(settings), ruleRow(settings, "isEvenAnswers"), ruleRow(settings, "vectorSize"), ...layers])}
      <div class="add-row">${addButton(`data-do="add-layer"`, "layers", "Add layer", settings.layers.length >= MAX_LAYERS)}</div>
      <p class="pg-section">Run</p>
      <p class="pg-badge pg-badge--orange">${iconSvg("repeat")} While training</p>
      ${bracket([ruleRow(settings, "rounds"), ruleRow(settings, "learningRate"), ruleRow(settings, "optimizer")])}
      <p class="pg-badge pg-badge--green">${iconSvg("check")} After training</p>
      ${bracket(useRows)}
      <div class="add-row">${unusedRows.length > 0 ? addButton(`data-menu="add-row" aria-haspopup="menu" aria-expanded="false"`, "plus", "Add setting") : ""}</div>`;
}

import type { ReactNode } from "react";
import { START_CHOICES, joinTokens } from "@/core/model/corpus";
import type { PlaygroundSettings, RowKey } from "@/features/playground/interfaces/playground.interface";
import { ConceptIcon } from "@/shared/components/concept-icon";
import { Icon } from "@/shared/components/icon";
import type { ConceptName } from "@/shared/interfaces/concept.interface";
import { TEXT_GROUPS } from "./playground-run";

interface Row {
  icon: ConceptName;
  label: string;
  verb: string;
  lesson: string;
  help: string;
  retrains: boolean;
  options: { label: string; value: PlaygroundSettings[RowKey] }[];
}

interface MenuControl {
  openMenuId: string | null;
  onToggleMenu: (id: string, chip: HTMLElement) => void;
}

interface BuilderProps extends MenuControl {
  settings: PlaygroundSettings;
  addedRows: RowKey[];
  onChange: (settings: PlaygroundSettings, retrains: boolean) => void;
  onRemoveRow: (key: RowKey) => void;
}

const MAX_LAYERS = 4;
export const OPTIONAL_ROWS: RowKey[] = ["temperature", "topK", "topP"];

const numbered = (values: number[], unit: string) => values.map((value) => ({ label: `${value} ${unit}`, value }));

export const ROWS: Record<RowKey, Row> = {
  isEvenAnswers: {
    icon: "pairs", label: "Usual answer", verb: "appears", lesson: "Lesson 14", retrains: true,
    help: "How often the text uses its usual answer. The model copies this pattern.",
    options: [{ label: "3 times more", value: false }, { label: "Just as often", value: true }],
  },
  vectorSize: {
    icon: "vector", label: "Each word", verb: "is", lesson: "Lesson 10", retrains: true,
    help: "How many numbers describe each word. More numbers hold more, but train slower.",
    options: numbered([6, 12, 24, 48], "numbers"),
  },
  rounds: {
    icon: "train-loop", label: "Repeat", verb: "for", lesson: "Lesson 14", retrains: true,
    help: "One round is: guess, score, find the cause, fix.",
    options: numbered([20, 50, 200, 500], "rounds"),
  },
  learningRate: {
    icon: "descent", label: "Step size", verb: "is", lesson: "Lesson 03", retrains: true,
    help: "How far the numbers move in one round. Too small learns slowly. Too big jumps past the answer.",
    options: [0.0004, 0.004, 0.04, 0.4].map((value) => ({ label: String(value), value })),
  },
  optimizer: {
    icon: "optimizer", label: "Fix numbers", verb: "with", lesson: "Lesson 17", retrains: true,
    help: "Adam gives every number its own step size. Plain steps use one size for all.",
    options: [{ label: "Adam", value: "adam" }, { label: "Plain steps", value: "plain" }],
  },
  start: {
    icon: "prompt", label: "Start with", verb: "", lesson: "Lesson 14", retrains: false,
    help: "The words you give the model. It continues from here.",
    options: START_CHOICES.map((value) => ({ label: joinTokens(value.split(" ")), value })),
  },
  temperature: {
    icon: "temperature", label: "Temperature", verb: "is", lesson: "Lesson 23", retrains: false,
    help: "Low picks the top word more often. High takes more risks.",
    options: [0.5, 1, 2].map((value) => ({ label: String(value), value })),
  },
  topK: {
    icon: "top-k", label: "Top-K", verb: "keeps", lesson: "Lesson 23", retrains: false,
    help: "Only pick from this many top words.",
    options: [{ label: "All words", value: 0 }, ...numbered([2, 3, 5], "words")],
  },
  topP: {
    icon: "top-p", label: "Top-P", verb: "keeps", lesson: "Lesson 23", retrains: false,
    help: "Only pick from the top words that add up to this chance.",
    options: [1, 0.9, 0.75, 0.5].map((value) => ({ label: `${Math.round(value * 100)}%`, value })),
  },
};

function optionLabel(settings: PlaygroundSettings, key: RowKey) {
  return ROWS[key].options.find((option) => option.value === settings[key])?.label ?? "";
}

function Label({ icon, text }: { icon: ConceptName; text: string }) {
  return (
    <span className="chip">
      <ConceptIcon name={icon} size="chip" /> {text}
    </span>
  );
}

function ValueChip({ menuId, text, isWide = false, menu }: { menuId: string; text: string; isWide?: boolean; menu: MenuControl }) {
  return (
    <button
      type="button"
      className={`chip chip--button${isWide ? " chip--wide" : ""}`}
      aria-haspopup="menu"
      aria-expanded={menu.openMenuId === menuId}
      onClick={(event) => menu.onToggleMenu(menuId, event.currentTarget)}
    >
      <span className="chip-text">{text}</span>
      <span className="chip-chevron"><Icon name="chevron" /></span>
    </button>
  );
}

function Bracket({ rows }: { rows: ReactNode[] }) {
  return (
    <div className={`rules${rows.length > 1 ? " rules--joined" : ""}`}>
      {rows.length > 1 && <span className="and-pill">And</span>}
      {rows}
    </div>
  );
}

function RuleRow({ rowKey, settings, menu, onRemove }: { rowKey: RowKey; settings: PlaygroundSettings; menu: MenuControl; onRemove?: () => void }) {
  const row = ROWS[rowKey];
  return (
    <div className="rule">
      <Label icon={row.icon} text={row.label} />
      {row.verb !== "" && <span className="rule-word">{row.verb}</span>}
      <ValueChip menuId={`set:${rowKey}`} text={optionLabel(settings, rowKey)} isWide={rowKey === "start"} menu={menu} />
      <span className="rule-end">
        <span className="rule-lesson">{row.lesson}</span>
        {onRemove && (
          <button type="button" className="icon-button" aria-label={`Remove ${row.label}`} onClick={onRemove}>
            <Icon name="trash" />
          </button>
        )}
      </span>
    </div>
  );
}

function TextRow({ settings, menu, onChange }: Pick<BuilderProps, "settings" | "onChange"> & { menu: MenuControl }) {
  const canRemove = settings.textGroups.length > 1;
  const canAdd = settings.textGroups.length < TEXT_GROUPS.length;
  const remove = (name: string) => onChange({ ...settings, textGroups: settings.textGroups.filter((group) => group !== name) }, true);
  return (
    <div className="rule">
      <Label icon="documents" text="Learn from" />
      <span className="tags">
        {settings.textGroups.map((name) => (
          <span key={name} className="tag">
            {name}
            {canRemove && (
              <button type="button" className="tag-remove" aria-label={`Remove ${name}`} onClick={() => remove(name)}>
                <Icon name="close" />
              </button>
            )}
          </span>
        ))}
        {canAdd && (
          <button
            type="button"
            className="icon-button"
            aria-haspopup="menu"
            aria-expanded={menu.openMenuId === "add-text"}
            aria-label="Add text"
            onClick={(event) => menu.onToggleMenu("add-text", event.currentTarget)}
          >
            <Icon name="plus" />
          </button>
        )}
      </span>
      <span className="rule-end"><span className="rule-lesson">Lesson 09</span></span>
    </div>
  );
}

interface PartRowProps {
  menuId: string;
  icon: ConceptName;
  text: string;
  isOn: boolean;
  lesson: string;
  menu: MenuControl;
}

function PartRow({ menuId, icon, text, isOn, lesson, menu }: PartRowProps) {
  return (
    <div className="rule">
      <Label icon={icon} text={text} />
      <span className="rule-word">is</span>
      <ValueChip menuId={menuId} text={isOn ? "On" : "Off"} menu={menu} />
      <span className="rule-end"><span className="rule-lesson">{lesson}</span></span>
    </div>
  );
}

function LayerGroup({ settings, index, menu, onChange }: Pick<BuilderProps, "settings" | "onChange"> & { index: number; menu: MenuControl }) {
  const layer = settings.layers[index];
  const parts = Number(layer.hasAttention) + Number(layer.hasFeedForward);
  const remove = () => onChange({ ...settings, layers: settings.layers.filter((_, i) => i !== index) }, true);
  return (
    <section className="rule-group" aria-label={`Layer ${index + 1}`}>
      <header className="rule-group-head">
        <span><b>{`Layer ${index + 1}`}</b>{` · ${parts} of 2 parts`}</span>
        <button type="button" className="icon-button" aria-label={`Remove layer ${index + 1}`} onClick={remove}>
          <Icon name="trash" />
        </button>
      </header>
      <Bracket
        rows={[
          <PartRow key="attention" menuId={`layer:${index}:attention`} icon="attention" text="Attention" isOn={layer.hasAttention} lesson="Lesson 12" menu={menu} />,
          <PartRow key="feed" menuId={`layer:${index}:feed`} icon="feed-forward" text="Feed-forward" isOn={layer.hasFeedForward} lesson="Lesson 15" menu={menu} />,
        ]}
      />
    </section>
  );
}

export function Builder({ settings, addedRows, openMenuId, onToggleMenu, onChange, onRemoveRow }: BuilderProps) {
  const menu = { openMenuId, onToggleMenu };
  const hasUnusedRows = OPTIONAL_ROWS.some((key) => !addedRows.includes(key));
  const addLayer = () => onChange({ ...settings, layers: [...settings.layers, { hasAttention: true, hasFeedForward: true }] }, true);
  return (
    <>
      <p className="pg-section">Build</p>
      <Bracket
        rows={[
          <TextRow key="text" settings={settings} menu={menu} onChange={onChange} />,
          <RuleRow key="isEvenAnswers" rowKey="isEvenAnswers" settings={settings} menu={menu} />,
          <RuleRow key="vectorSize" rowKey="vectorSize" settings={settings} menu={menu} />,
          ...settings.layers.map((_, index) => <LayerGroup key={`layer-${index}`} settings={settings} index={index} menu={menu} onChange={onChange} />),
        ]}
      />
      <div className="add-row">
        <button type="button" className="add-button" disabled={settings.layers.length >= MAX_LAYERS} onClick={addLayer}>
          <Icon name="plus" /> Add layer
        </button>
      </div>
      <p className="pg-section">Run</p>
      <p className="pg-badge pg-badge--orange"><Icon name="repeat" /> While training</p>
      <Bracket
        rows={[
          <RuleRow key="rounds" rowKey="rounds" settings={settings} menu={menu} />,
          <RuleRow key="learningRate" rowKey="learningRate" settings={settings} menu={menu} />,
          <RuleRow key="optimizer" rowKey="optimizer" settings={settings} menu={menu} />,
        ]}
      />
      <p className="pg-badge pg-badge--green"><Icon name="check" /> After training</p>
      <Bracket
        rows={[
          <RuleRow key="start" rowKey="start" settings={settings} menu={menu} />,
          ...addedRows.map((key) => <RuleRow key={key} rowKey={key} settings={settings} menu={menu} onRemove={() => onRemoveRow(key)} />),
        ]}
      />
      <div className="add-row">
        {hasUnusedRows && (
          <button
            type="button"
            className="add-button"
            aria-haspopup="menu"
            aria-expanded={openMenuId === "add-row"}
            onClick={(event) => onToggleMenu("add-row", event.currentTarget)}
          >
            <Icon name="plus" /> Add setting
          </button>
        )}
      </div>
    </>
  );
}

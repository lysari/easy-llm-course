import { type RefObject, useLayoutEffect } from "react";
import type { PlaygroundSettings, RowKey } from "@/features/playground/interfaces/playground.interface";
import { Icon } from "@/shared/components/icon";
import { OPTIONAL_ROWS, ROWS } from "./builder";
import { TEXT_GROUPS } from "./playground-run";

const MENU_MARGIN = 8;

interface MenuItem {
  label: string;
  isOn: boolean;
  pick: () => void;
}

interface MenuProps {
  ref: RefObject<HTMLDivElement | null>;
  id: string;
  chip: HTMLElement;
  settings: PlaygroundSettings;
  addedRows: RowKey[];
  onChange: (settings: PlaygroundSettings, retrains: boolean) => void;
  onAddRow: (key: RowKey) => void;
  onClose: (returnFocus: boolean) => void;
}

function menuFor({ id, settings, addedRows, onChange, onAddRow }: MenuProps): { help: string; items: MenuItem[] } {
  const [kind, first, second] = id.split(":");
  if (kind === "set") {
    const key = first as RowKey;
    const row = ROWS[key];
    return {
      help: row.help,
      items: row.options.map((option) => ({
        label: option.label,
        isOn: option.value === settings[key],
        pick: () => onChange({ ...settings, [key]: option.value }, row.retrains),
      })),
    };
  }
  if (kind === "layer") {
    const part = second === "attention" ? "hasAttention" : "hasFeedForward";
    const index = Number(first);
    const choice = (isOn: boolean) => ({
      label: isOn ? "On" : "Off",
      isOn: settings.layers[index][part] === isOn,
      pick: () => onChange({ ...settings, layers: settings.layers.map((layer, i) => (i === index ? { ...layer, [part]: isOn } : layer)) }, true),
    });
    return {
      help: second === "attention" ? "Lets each word look back at earlier words." : "Lets each word think alone about what it collected.",
      items: [choice(true), choice(false)],
    };
  }
  if (kind === "add-text") {
    return {
      help: "Add sentences for the model to learn from.",
      items: TEXT_GROUPS.filter((group) => !settings.textGroups.includes(group.name)).map((group) => ({
        label: group.name,
        isOn: false,
        pick: () => onChange({ ...settings, textGroups: TEXT_GROUPS.map((item) => item.name).filter((name) => settings.textGroups.includes(name) || name === group.name) }, true),
      })),
    };
  }
  return {
    help: "Change how the model picks each word.",
    items: OPTIONAL_ROWS.filter((key) => !addedRows.includes(key)).map((key) => ({
      label: ROWS[key].label,
      isOn: false,
      pick: () => onAddRow(key),
    })),
  };
}

export function Menu(props: MenuProps) {
  const { ref, chip, onClose } = props;
  const { help, items } = menuFor(props);

  useLayoutEffect(() => {
    const menu = ref.current;
    if (!menu) return;
    const box = chip.getBoundingClientRect();
    const fitsBelow = box.bottom + MENU_MARGIN + menu.offsetHeight < window.innerHeight;
    menu.style.left = `${Math.max(MENU_MARGIN, Math.min(box.left, window.innerWidth - menu.offsetWidth - MENU_MARGIN))}px`;
    menu.style.top = `${fitsBelow ? box.bottom + MENU_MARGIN : Math.max(MENU_MARGIN, box.top - MENU_MARGIN - menu.offsetHeight)}px`;
    (menu.querySelector<HTMLElement>('[aria-checked="true"]') ?? menu.querySelector<HTMLElement>(".pg-menu-item"))?.focus();
  }, [ref, chip]);

  return (
    <div className="pg-menu" role="menu" ref={ref}>
      <p className="pg-menu-help">{help}</p>
      {items.map((item) => (
        <button
          key={item.label}
          type="button"
          className="pg-menu-item"
          role="menuitemradio"
          aria-checked={item.isOn}
          onClick={() => {
            onClose(true);
            item.pick();
          }}
        >
          <span>{item.label}</span>
          {item.isOn && <Icon name="check" />}
        </button>
      ))}
    </div>
  );
}

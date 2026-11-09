import type { PlaygroundState, RowKey } from "@/features/playground/interfaces/playground.interface";
import { escapeHtml } from "@/shared/utils/html";
import { iconSvg } from "@/shared/utils/icons";
import { OPTIONAL_ROWS, ROWS } from "./builder";
import { TEXT_GROUPS } from "./playground-run";

interface MenuItem {
  label: string;
  isOn: boolean;
  apply: () => void;
}

export function createMenu(root: HTMLElement, state: PlaygroundState, changed: (retrains: boolean) => void) {
  let openMenu: string | null = null;

  function menuFor(id: string): { help: string; items: MenuItem[] } {
    const [kind, first, second] = id.split(":");
    if (kind === "set") {
      const key = first as RowKey;
      const row = ROWS[key];
      return {
        help: row.help,
        items: row.options.map((option) => ({
          label: option.label,
          isOn: option.value === state.settings[key],
          apply: () => {
            state.settings = { ...state.settings, [key]: option.value };
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
        isOn: state.settings.layers[index][part] === isOn,
        apply: () => {
          state.settings.layers = state.settings.layers.map((layer, i) => (i === index ? { ...layer, [part]: isOn } : layer));
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
        items: TEXT_GROUPS.filter((group) => !state.settings.textGroups.includes(group.name)).map((group) => ({
          label: group.name,
          isOn: false,
          apply: () => {
            state.settings.textGroups = TEXT_GROUPS.map((item) => item.name).filter((name) => state.settings.textGroups.includes(name) || name === group.name);
            changed(true);
          },
        })),
      };
    }
    return {
      help: "Change how the model picks each word.",
      items: OPTIONAL_ROWS.filter((key) => !state.addedRows.includes(key)).map((key) => ({
        label: ROWS[key].label,
        isOn: false,
        apply: () => {
          state.addedRows = [...state.addedRows, key];
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

  return {
    close: () => closeMenu(),
    toggle(chip: HTMLElement | null) {
      const wasOpen = openMenu;
      closeMenu();
      if (chip && wasOpen !== chip.dataset.menu) showMenu(chip);
    },
  };
}

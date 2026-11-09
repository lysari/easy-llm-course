import { type RefObject, useCallback, useEffect, useRef, useState } from "react";

/**
 * Opens one `<Drawer>` at a time by id, for cards stored in `triggers` by the same id.
 * Closes on Escape or a click outside the drawer and the cards, moves focus in on open, and back to the card on close.
 */
export function useDrawer(triggers: RefObject<Map<string, HTMLElement>>) {
  const [openId, setOpenId] = useState<string | null>(null);
  const drawerRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!openId) return;
    function closeOnOutsideClick(event: MouseEvent) {
      const target = event.target;
      if (!(target instanceof Node) || drawerRef.current?.contains(target)) return;
      if ([...triggers.current.values()].some((card) => card.contains(target))) return;
      setOpenId(null);
    }
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") setOpenId(null);
    }
    document.addEventListener("click", closeOnOutsideClick, true);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("click", closeOnOutsideClick, true);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [openId, triggers]);

  useEffect(() => {
    if (!openId) return;
    drawerRef.current?.focus();
    const cards = triggers.current;
    return () => {
      if (document.activeElement === document.body) cards.get(openId)?.focus();
    };
  }, [openId, triggers]);

  const close = useCallback(() => setOpenId(null), []);

  function toggle(id: string) {
    setOpenId((current) => (current === id ? null : id));
  }

  return { openId, drawerRef, toggle, close };
}

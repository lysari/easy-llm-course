import { type KeyboardEvent, type MouseEvent, type RefObject, useEffect, useRef, useState } from "react";

interface OpenMenu {
  id: string;
  chip: HTMLElement;
}

export function useMenu(rootRef: RefObject<HTMLElement | null>, isOpen: boolean) {
  const [openMenu, setOpenMenu] = useState<OpenMenu | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isOpen) setOpenMenu(null);
  }, [isOpen]);

  useEffect(() => {
    if (!openMenu) return;
    const close = () => setOpenMenu(null);
    const closeOutsideRoot = (event: Event) => {
      if (!rootRef.current?.contains(event.target as Node)) close();
    };
    document.addEventListener("click", closeOutsideRoot);
    window.addEventListener("resize", close);
    return () => {
      document.removeEventListener("click", closeOutsideRoot);
      window.removeEventListener("resize", close);
    };
  }, [openMenu, rootRef]);

  function close(returnFocus = false) {
    if (returnFocus) openMenu?.chip.focus();
    setOpenMenu(null);
  }

  function toggle(id: string, chip: HTMLElement) {
    setOpenMenu(openMenu?.id === id ? null : { id, chip });
  }

  function closeUnlessInMenu(event: MouseEvent) {
    if (!menuRef.current?.contains(event.target as Node)) setOpenMenu(null);
  }

  function handleKeyDown(event: KeyboardEvent) {
    if (!openMenu) return;
    if (event.key === "Escape") close(true);
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    const items = [...(menuRef.current?.querySelectorAll<HTMLElement>(".pg-menu-item") ?? [])];
    const step = event.key === "ArrowDown" ? 1 : -1;
    items[(items.indexOf(document.activeElement as HTMLElement) + step + items.length) % items.length]?.focus();
  }

  return { openMenu, menuRef, close, toggle, closeUnlessInMenu, handleKeyDown };
}

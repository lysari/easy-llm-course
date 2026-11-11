import { useEffect, useState } from "react";

export type Backdrop = "wash" | "meadow" | "sky" | "citrus" | "photo";

const STORAGE_KEY = "backdrop";

export function parseBackdrop(saved: string | null): Backdrop {
  return saved === "meadow" || saved === "sky" || saved === "citrus" || saved === "photo" ? saved : "wash";
}

function savedBackdrop(): Backdrop {
  try {
    return parseBackdrop(localStorage.getItem(STORAGE_KEY));
  } catch {
    return "wash";
  }
}

export function useBackdrop() {
  const [backdrop, setBackdrop] = useState<Backdrop>(savedBackdrop);

  useEffect(() => {
    document.documentElement.dataset.backdrop = backdrop;
    try {
      localStorage.setItem(STORAGE_KEY, backdrop);
    } catch {
      return;
    }
  }, [backdrop]);

  return { backdrop, setBackdrop };
}

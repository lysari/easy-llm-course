import { useEffect, useState } from "react";

export type Backdrop = "wash" | "photo";

const STORAGE_KEY = "backdrop";

function savedBackdrop(): Backdrop {
  try {
    return localStorage.getItem(STORAGE_KEY) === "photo" ? "photo" : "wash";
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

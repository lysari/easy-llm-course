import { useEffect, useState } from "react";

export type Look = "auto" | "light" | "dark" | "fresh";

const STORAGE_KEY = "look";

function savedLook(): Look {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved === "light" || saved === "dark" || saved === "fresh" ? saved : "auto";
  } catch {
    return "auto";
  }
}

export function useLook() {
  const [look, setLook] = useState<Look>(savedLook);

  useEffect(() => {
    document.documentElement.dataset.look = look;
    try {
      localStorage.setItem(STORAGE_KEY, look);
    } catch {
      return;
    }
  }, [look]);

  return { look, setLook };
}

import { useEffect, useState } from "react";

export type Look = "auto" | "light" | "dark";

const STORAGE_KEY = "look";

export function parseLook(saved: string | null): Look {
  return saved === "light" || saved === "dark" ? saved : "auto";
}

function savedLook(): Look {
  try {
    return parseLook(localStorage.getItem(STORAGE_KEY));
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

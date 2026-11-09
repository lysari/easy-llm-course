import { useSyncExternalStore } from "react";

export const NARROW_WIDTH = 960;

function subscribeToResize(onChange: () => void) {
  window.addEventListener("resize", onChange);
  return () => window.removeEventListener("resize", onChange);
}

export function useWindowWidth() {
  return useSyncExternalStore(subscribeToResize, () => window.innerWidth);
}

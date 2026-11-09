import { useSyncExternalStore } from "react";

export const NARROW_WIDTH = 960;

function subscribeToResize(onChange: () => void) {
  window.addEventListener("resize", onChange);
  return () => window.removeEventListener("resize", onChange);
}

/** Returns the window width and redraws on resize. Compare it with `NARROW_WIDTH` to switch to one column. */
export function useWindowWidth() {
  return useSyncExternalStore(subscribeToResize, () => window.innerWidth);
}

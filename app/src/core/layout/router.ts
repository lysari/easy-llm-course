import { useSyncExternalStore } from "react";

function subscribeToHash(onChange: () => void) {
  window.addEventListener("hashchange", onChange);
  return () => window.removeEventListener("hashchange", onChange);
}

export function useIsPlayground() {
  return useSyncExternalStore(subscribeToHash, () => location.hash === "#playground");
}

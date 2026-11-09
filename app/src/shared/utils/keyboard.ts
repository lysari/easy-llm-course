import type { KeyboardEvent } from "react";

/** Clicks a focusable non-button element on Enter or Space, as a button would. Pass it as the element's onKeyDown. */
export function activateOnKey(event: KeyboardEvent<HTMLElement>) {
  if (event.target !== event.currentTarget || (event.key !== "Enter" && event.key !== " ")) return;
  event.preventDefault();
  event.currentTarget.click();
}

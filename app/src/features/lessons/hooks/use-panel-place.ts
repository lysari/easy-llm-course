import { type CSSProperties, type RefObject, useLayoutEffect, useState } from "react";
import { NESTED_STEPS } from "@/features/lessons/flow";

const PANEL_GAP = 8;

export function usePanelPlace(panelRef: RefObject<HTMLElement | null>, partCards: RefObject<Map<string, HTMLElement>>, stepId: string, width: number) {
  const [place, setPlace] = useState<CSSProperties>({});

  useLayoutEffect(() => {
    const panel = panelRef.current;
    const card = partCards.current.get(stepId);
    if (!panel || !card) return;
    const left = NESTED_STEPS.includes(stepId) ? card.offsetLeft + card.offsetWidth - panel.offsetWidth : card.offsetLeft;
    setPlace({ top: card.offsetTop + card.offsetHeight + PANEL_GAP, left });
  }, [panelRef, partCards, stepId, width]);

  return place;
}

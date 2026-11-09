import { type RefObject, useLayoutEffect, useState } from "react";
import { NESTED_STEPS, stepsOf } from "@/features/lessons/flow";
import type { LinkDrawing } from "@/features/lessons/interfaces/link.interface";
import type { StageId } from "@/features/lessons/interfaces/story.interface";

const CORNER = 10;

interface FlowElements {
  flowRef: RefObject<HTMLElement | null>;
  partsRef: RefObject<HTMLElement | null>;
  stageCards: RefObject<Map<StageId, HTMLElement>>;
  partCards: RefObject<Map<string, HTMLElement>>;
}

function pathBetween(x1: number, y1: number, trunk: number, x2: number, y2: number) {
  if (Math.abs(y2 - y1) < 1) return `M ${x1} ${y1} H ${x2}`;
  const r = Math.min(CORNER, Math.abs(y2 - y1) / 2);
  const direction = y2 > y1 ? 1 : -1;
  return `M ${x1} ${y1} H ${trunk - r} Q ${trunk} ${y1} ${trunk} ${y1 + direction * r} V ${y2 - direction * r} Q ${trunk} ${y2} ${trunk + r} ${y2} H ${x2}`;
}

function measureLinks({ flowRef, partsRef, stageCards, partCards }: FlowElements, stageId: StageId): LinkDrawing | null {
  const stageCard = stageCards.current.get(stageId);
  if (!flowRef.current || !partsRef.current || !stageCard) return null;
  const flow = flowRef.current.getBoundingClientRect();
  const parent = stageCard.getBoundingClientRect();
  const startX = parent.right - flow.left;
  const startY = parent.top + parent.height / 2 - flow.top;
  const trunk = (startX + partsRef.current.getBoundingClientRect().left - flow.left) / 2;
  const drawing: LinkDrawing = { paths: [], dots: [{ id: "start", x: startX, y: startY }] };
  let previous: DOMRect | null = null;
  for (const step of stepsOf(stageId)) {
    const card = partCards.current.get(step.id);
    if (!card) return null;
    const box = card.getBoundingClientRect();
    const y2 = box.top + box.height / 2 - flow.top;
    const endX = box.left - flow.left;
    if (NESTED_STEPS.includes(step.id) && previous) {
      const x = previous.right - flow.left;
      drawing.paths.push({ id: step.id, d: `M ${x} ${y2} H ${endX}` });
      drawing.dots.push({ id: step.id, x, y: y2 });
    } else {
      drawing.paths.push({ id: step.id, d: pathBetween(startX, startY, trunk, endX, y2) });
    }
    previous = box;
  }
  return drawing;
}

export function useFlowLinks(elements: FlowElements, stageId: StageId, isShown: boolean, width: number) {
  const [drawing, setDrawing] = useState<LinkDrawing | null>(null);
  const { flowRef, partsRef, stageCards, partCards } = elements;

  useLayoutEffect(() => {
    setDrawing(isShown ? measureLinks({ flowRef, partsRef, stageCards, partCards }, stageId) : null);
  }, [flowRef, partsRef, stageCards, partCards, stageId, isShown, width]);

  return drawing;
}

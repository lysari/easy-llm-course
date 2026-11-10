import { type RefObject, useLayoutEffect, useState } from "react";
import type { LinkDrawing } from "@/shared/interfaces/link.interface";

const CORNER = 10;
const NO_NESTED: string[] = [];

interface FlowLinkElements {
  flowRef: RefObject<HTMLElement | null>;
  childColumnRef: RefObject<HTMLElement | null>;
  parentCards: RefObject<Map<string, HTMLElement>>;
  childCards: RefObject<Map<string, HTMLElement>>;
}

interface FlowLinkTargets {
  parentId: string;
  childIds: string[];
  nestedIds?: string[];
}

interface FlowLinkOptions extends FlowLinkTargets {
  isShown: boolean;
  width: number;
}

function pathBetween(x1: number, y1: number, trunk: number, x2: number, y2: number) {
  if (Math.abs(y2 - y1) < 1) return `M ${x1} ${y1} H ${x2}`;
  const r = Math.min(CORNER, Math.abs(y2 - y1) / 2);
  const direction = y2 > y1 ? 1 : -1;
  return `M ${x1} ${y1} H ${trunk - r} Q ${trunk} ${y1} ${trunk} ${y1 + direction * r} V ${y2 - direction * r} Q ${trunk} ${y2} ${trunk + r} ${y2} H ${x2}`;
}

/**
 * Measures the lines from a parent card to its child cards; `useFlowLinks` calls it after layout.
 * Each path and dot keeps the id of the card it starts from (`fromId`), so `FlowLinks` can colour the lines of an active card from `startX` to `endX`.
 */
export function measureLinks({ flowRef, childColumnRef, parentCards, childCards }: FlowLinkElements, { parentId, childIds, nestedIds = NO_NESTED }: FlowLinkTargets): LinkDrawing | null {
  const parentCard = parentCards.current.get(parentId);
  if (!flowRef.current || !childColumnRef.current || !parentCard) return null;
  const flow = flowRef.current.getBoundingClientRect();
  const parent = parentCard.getBoundingClientRect();
  const startX = parent.right - flow.left;
  const startY = parent.top + parent.height / 2 - flow.top;
  const trunk = (startX + childColumnRef.current.getBoundingClientRect().left - flow.left) / 2;
  const drawing: LinkDrawing = { paths: [], dots: [{ id: "start", fromId: parentId, x: startX, y: startY }] };
  let previous: { id: string; box: DOMRect } | null = null;
  for (const id of childIds) {
    const card = childCards.current.get(id);
    if (!card) return null;
    const box = card.getBoundingClientRect();
    const y2 = box.top + box.height / 2 - flow.top;
    const endX = box.left - flow.left;
    if (nestedIds.includes(id) && previous) {
      const x = previous.box.right - flow.left;
      drawing.paths.push({ id, fromId: previous.id, d: `M ${x} ${y2} H ${endX}`, startX: x, endX });
      drawing.dots.push({ id, fromId: previous.id, x, y: y2 });
    } else {
      drawing.paths.push({ id, fromId: parentId, d: pathBetween(startX, startY, trunk, endX, y2), startX, endX });
    }
    previous = { id, box };
  }
  return drawing;
}

/**
 * Measures the lines from a parent card to its child cards after layout, for `<FlowLinks drawing={...} />`.
 * Pass a stable (memoized) `childIds`; a child in `nestedIds` links from the child card before it instead of the parent.
 */
export function useFlowLinks(elements: FlowLinkElements, { parentId, childIds, nestedIds = NO_NESTED, isShown, width }: FlowLinkOptions) {
  const [drawing, setDrawing] = useState<LinkDrawing | null>(null);
  const { flowRef, childColumnRef, parentCards, childCards } = elements;

  useLayoutEffect(() => {
    setDrawing(isShown ? measureLinks({ flowRef, childColumnRef, parentCards, childCards }, { parentId, childIds, nestedIds }) : null);
  }, [flowRef, childColumnRef, parentCards, childCards, parentId, childIds, nestedIds, isShown, width]);

  return drawing;
}

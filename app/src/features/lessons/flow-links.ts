import type { StageId } from "@/features/lessons/interfaces/story.interface";
import { byId } from "@/shared/utils/html";
import { NESTED_STEPS, stepsOf } from "./flow";

const NARROW_WIDTH = 960;
const CORNER = 10;
export const STAGGER_MS = 70;

export const isNarrow = () => window.innerWidth <= NARROW_WIDTH;

function pathBetween(x1: number, y1: number, trunk: number, x2: number, y2: number) {
  if (Math.abs(y2 - y1) < 1) return `M ${x1} ${y1} H ${x2}`;
  const r = Math.min(CORNER, Math.abs(y2 - y1) / 2);
  const direction = y2 > y1 ? 1 : -1;
  return `M ${x1} ${y1} H ${trunk - r} Q ${trunk} ${y1} ${trunk} ${y1 + direction * r} V ${y2 - direction * r} Q ${trunk} ${y2} ${trunk + r} ${y2} H ${x2}`;
}

export function drawLinks(stageId: StageId) {
  const links = byId("links");
  if (isNarrow()) {
    links.innerHTML = "";
    return;
  }
  const flow = byId("flow").getBoundingClientRect();
  const parent = byId(`stage-${stageId}`).getBoundingClientRect();
  const startX = parent.right - flow.left;
  const startY = parent.top + parent.height / 2 - flow.top;
  const trunk = (startX + byId("parts").getBoundingClientRect().left - flow.left) / 2;
  const paths: string[] = [];
  const dots: string[] = [`<circle class="link-dot" cx="${startX}" cy="${startY}" r="4"/>`];
  let previous: DOMRect | null = null;
  let delay = 0;
  for (const step of stepsOf(stageId)) {
    const box = byId(`part-${step.id}`).getBoundingClientRect();
    const y2 = box.top + box.height / 2 - flow.top;
    const endX = box.left - flow.left;
    if (NESTED_STEPS.includes(step.id) && previous) {
      const x = previous.right - flow.left;
      paths.push(`<path class="link-path" pathLength="1" style="animation-delay:${delay}ms" d="M ${x} ${y2} H ${endX}"/>`);
      dots.push(`<circle class="link-dot" cx="${x}" cy="${y2}" r="4"/>`);
    } else {
      paths.push(`<path class="link-path" pathLength="1" style="animation-delay:${delay}ms" d="${pathBetween(startX, startY, trunk, endX, y2)}"/>`);
    }
    previous = box;
    delay += STAGGER_MS;
  }
  links.innerHTML = paths.join("") + dots.join("");
}

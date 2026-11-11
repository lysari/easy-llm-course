import type { PageKey, Places, Run, ScoredElement, SiblingSeen } from "@/features/inside-real-models/interfaces/readability-run.interface";
import type { PageBox, ReadabilityTrace, SiblingCheck, SiblingReason } from "@/features/inside-real-models/interfaces/readability-trace.interface";

export const MIN_SIBLING_SCORE = 10;
export const SIBLING_SHARE = 0.2;
export const JOINING_REASONS: SiblingReason[] = ["winner", "high-score", "long-paragraph", "short-sentence"];

const PREVIEW_LENGTH = 90;

const plain = (text: string) => text.replace(/\s+/g, " ").trim();

export function startRun(body: HTMLElement, byline: string | null): Run {
  const order = new Map<Element, number>([[body, -1]]);
  const pageParent = new Map<Element, Element>();
  [...body.getElementsByTagName("*")].forEach((element, i) => {
    order.set(element, i);
    if (element.parentElement) pageParent.set(element, element.parentElement);
  });
  const places = { order, pageParent, liveParent: new Map(), liveKey: new Map(), replacementOf: new Map(), originalOf: new Map() };
  return { phase: "walk", places, byline, seeds: new Map(), removals: new Map(), emptyCount: 0, pendingRemoval: null, paragraphs: new Map(), scores: new Map(), pick: { siblings: [], parent: null, appended: new Set(), content: null, winner: null, joined: new Set(), kept: new Set() } };
}

export function latest(places: Places, node: Element) {
  let current = node;
  for (let next = places.replacementOf.get(current); next; next = places.replacementOf.get(current)) current = next;
  return current;
}

export function placeLiveBoxes(places: Places, body: HTMLElement) {
  let anchor = -1;
  let added = 0;
  places.liveKey.set(body, [anchor, added]);
  for (const element of body.getElementsByTagName("*")) {
    const order = places.order.get(places.originalOf.get(element) ?? element);
    if (order === undefined) added += 1;
    else [anchor, added] = [order, 0];
    places.liveKey.set(element, [anchor, added]);
    if (element.parentElement) places.liveParent.set(element, element.parentElement);
  }
}

export function seed(run: Run, node: Element, text: string) {
  if (run.seeds.has(node)) return;
  const original = run.places.originalOf.get(node) ?? node;
  const classes = (node.getAttribute("class") ?? "").trim().split(/\s+/).filter(Boolean);
  run.seeds.set(node, {
    tag: original.tagName.toLowerCase(),
    name: `${classes.map((name) => `.${name}`).join("")}${node.id ? `#${node.id}` : ""}`,
    preview: plain(text).slice(0, PREVIEW_LENGTH),
    isAdded: !run.places.order.has(original),
  });
}

function siblingReasonOf(seen: SiblingSeen, winner: ScoredElement, threshold: number): { reason: SiblingReason; bonus: number } {
  if (seen.node === winner) return { reason: "winner", bonus: 0 };
  const winnerScore = winner.readability?.contentScore ?? 0;
  const bonus = seen.className === winner.className && winner.className !== "" ? winnerScore * SIBLING_SHARE : 0;
  if (seen.score !== null && seen.score + bonus >= threshold) return { reason: "high-score", bonus };
  if (seen.isParagraph) {
    if (seen.textLength > 80 && seen.linkShare < 0.25) return { reason: "long-paragraph", bonus };
    const isSentence = seen.textLength < 80 && seen.textLength > 0 && seen.linkShare === 0 && /\.( |$)/.test(plain(seen.node.textContent ?? ""));
    return { reason: isSentence ? "short-sentence" : "paragraph-rules", bonus };
  }
  return { reason: seen.score === null ? "no-score" : "low-score", bonus };
}

export function buildTrace(run: Run, attempt: number): ReadabilityTrace | null {
  const { places } = run;
  const { winner } = run.pick;
  if (!winner) return null;
  const keyOf = (node: Element): PageKey | undefined => places.liveKey.get(node) ?? (run.removals.has(node) ? [places.order.get(node) ?? -1, 0] : undefined);
  const nodes = [...run.seeds.keys()].filter((node) => keyOf(node));
  if (!nodes.includes(winner)) return null;
  nodes.sort((a, b) => {
    const [first, second] = [keyOf(a) ?? [0, 0], keyOf(b) ?? [0, 0]];
    return first[0] - second[0] || first[1] - second[1];
  });
  const idOf = new Map(nodes.map((node, i) => [node, i]));
  const up = (node: Element) => {
    const live = places.liveParent.get(node);
    if (live) return live;
    const page = places.pageParent.get(node);
    return page ? latest(places, page) : null;
  };
  const depthOf = (node: Element) => {
    let depth = 0;
    for (let parent = up(node); parent; parent = up(parent)) if (idOf.has(parent)) depth += 1;
    return depth;
  };
  const winnerScore = winner.readability?.contentScore ?? 0;
  const threshold = Math.max(MIN_SIBLING_SCORE, winnerScore * SIBLING_SHARE);
  const siblings = new Map<Element, SiblingCheck>();
  for (const seen of run.pick.siblings) {
    const { reason, bonus } = siblingReasonOf(seen, winner, threshold);
    if (JOINING_REASONS.includes(reason) !== run.pick.appended.has(seen.node)) return null;
    siblings.set(seen.node, { reason, bonus, textLength: seen.textLength });
  }
  let bestId = -1;
  let bestScore = -Infinity;
  for (const [node, record] of run.scores) {
    if (!record.isCandidate || record.score === null || record.score <= bestScore) continue;
    bestId = idOf.get(node) ?? -1;
    bestScore = record.score;
  }
  const boxes = nodes.map((node, id): PageBox => {
    const record = run.scores.get(node);
    const seedOf = run.seeds.get(node) ?? { tag: "", name: "", preview: "", isAdded: false };
    return {
      id,
      depth: depthOf(node),
      ...seedOf,
      removal: run.removals.get(node) ?? null,
      paragraph: run.paragraphs.get(node) ?? null,
      score: record
        ? {
            start: record.start,
            namePoints: record.namePoints,
            nameWords: record.nameWords,
            received: record.received.map((part) => ({ fromId: idOf.get(part.from) ?? -1, level: part.level, points: part.points })),
            beforeLinks: record.beforeLinks ?? record.start,
            linkShare: record.linkShare,
            score: record.score ?? record.start,
          }
        : null,
      sibling: siblings.get(node) ?? null,
      isJoined: run.pick.joined.has(node),
      isKept: run.pick.kept.has(node),
    };
  });
  return { attempt, boxes, emptyCount: run.emptyCount, winnerId: idOf.get(winner) ?? -1, bestId, threshold };
}

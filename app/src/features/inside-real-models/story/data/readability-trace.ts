import type { Readability } from "@mozilla/readability";
import type { Run, ScoredElement } from "@/features/inside-real-models/interfaces/readability-run.interface";
import type { NameWord, ReadabilityTrace } from "@/features/inside-real-models/interfaces/readability-trace.interface";
import { buildTrace, latest, placeLiveBoxes, seed, startRun } from "./readability-run";

export const MIN_PARAGRAPH_LENGTH = 25;

const NAME_POINTS = 25;
const ANCESTOR_LEVELS = 5;

type EachNode = (this: unknown, node: Node, index: number) => void;
type SomeNode = (this: unknown, node: Node) => boolean;

interface ReadabilityInternals {
  REGEXPS: Record<"unlikelyCandidates" | "positive" | "negative" | "commas", RegExp>;
  FLAG_WEIGHT_CLASSES: number;
  _doc: Document;
  _flags: number;
  _articleByline: string | null;
  _attempts: unknown[];
  log(...args: unknown[]): void;
  _removeAndGetNext(node: Element): Element | null;
  _getNextNode(node: Element, ignoreSelfAndKids?: boolean): Element | null;
  _setNodeTag(node: Element, tag: string): Element;
  _forEachNode(nodes: ArrayLike<Node>, fn: EachNode): void;
  _someNode(nodes: ArrayLike<Node>, fn: SomeNode): boolean;
  _initializeNode(node: ScoredElement): void;
  _getLinkDensity(element: Element): number;
  _getInnerText(element: Element, normalizeSpaces?: boolean): string;
  _getClassWeight(element: Element): number;
  _getNodeAncestors(node: Node, maxDepth?: number): ScoredElement[];
  _isElementWithoutContent(node: Element): boolean;
  _prepArticle(content: Element): void;
}

const dividerOf = (level: number) => (level === 0 ? 1 : level === 1 ? 2 : level * 3);

const isElement = (node: unknown): node is ScoredElement => typeof node === "object" && node !== null && "tagName" in node;

export function traceReadability(reader: Readability<Node>) {
  const internals = reader as unknown as ReadabilityInternals;
  const original = {
    log: internals.log.bind(internals),
    removeAndGetNext: internals._removeAndGetNext.bind(internals),
    getNextNode: internals._getNextNode.bind(internals),
    setNodeTag: internals._setNodeTag.bind(internals),
    forEachNode: internals._forEachNode.bind(internals),
    someNode: internals._someNode.bind(internals),
    initializeNode: internals._initializeNode.bind(internals),
    getLinkDensity: internals._getLinkDensity.bind(internals),
    getInnerText: internals._getInnerText.bind(internals),
    getClassWeight: internals._getClassWeight.bind(internals),
    getNodeAncestors: internals._getNodeAncestors.bind(internals),
    isElementWithoutContent: internals._isElementWithoutContent.bind(internals),
    prepArticle: internals._prepArticle.bind(internals),
  };
  let run: Run | null = null;
  let attempt = 0;
  let isBroken = false;

  const check = (isTrue: boolean) => {
    if (!isTrue) isBroken = true;
  };

  function nameWordsOf(node: Element, namePoints: number) {
    const words: NameWord[] = [];
    if ((internals._flags & internals.FLAG_WEIGHT_CLASSES) === 0) return words;
    for (const value of [node.className, node.id]) {
      if (typeof value !== "string" || value === "") continue;
      const bad = internals.REGEXPS.negative.exec(value);
      if (bad) words.push({ word: bad[0].trim(), points: -NAME_POINTS });
      const good = internals.REGEXPS.positive.exec(value);
      if (good) words.push({ word: good[0].trim(), points: NAME_POINTS });
    }
    check(words.reduce((sum, word) => sum + word.points, 0) === namePoints);
    return words;
  }

  function noteRemoval(current: Run, node: Element) {
    const pending = current.pendingRemoval;
    current.pendingRemoval = null;
    if (!current.places.order.has(node)) return;
    let removal = pending;
    if (removal?.reason === "unlikely") removal = { reason: "unlikely", match: internals.REGEXPS.unlikelyCandidates.exec(`${node.className} ${node.id}`)?.[0] ?? "" };
    else if (removal?.reason === "role") removal = { reason: "role", match: node.getAttribute("role") ?? "" };
    else if (!removal && internals._articleByline !== current.byline) removal = { reason: "byline", match: "" };
    else if (!removal && node.getAttribute("aria-modal") === "true" && node.getAttribute("role") === "dialog") removal = { reason: "dialog", match: "" };
    current.byline = internals._articleByline;
    if (removal) {
      current.removals.set(node, removal);
      seed(current, node, node.textContent ?? "");
      return;
    }
    check(original.isElementWithoutContent(node));
    current.emptyCount += 1;
  }

  function scoreParagraph(current: Run, node: Node, scoreIt: () => void) {
    const parent = node.parentNode;
    if (!isElement(node) || !isElement(parent)) return scoreIt();
    const text = original.getInnerText(node);
    const isTooShort = text.length < MIN_PARAGRAPH_LENGTH;
    const commaPoints = isTooShort ? 0 : text.split(internals.REGEXPS.commas).length;
    const lengthPoints = isTooShort ? 0 : Math.min(Math.floor(text.length / 100), 3);
    const total = isTooShort ? 0 : 1 + commaPoints + lengthPoints;
    const ancestors = original.getNodeAncestors(node, ANCESTOR_LEVELS);
    const before = ancestors.map((ancestor) => ancestor.readability?.contentScore);
    scoreIt();
    if (ancestors.length === 0) return;
    current.paragraphs.set(node, { textLength: text.length, commaPoints, lengthPoints, total, isTooShort });
    seed(current, node, text);
    if (isTooShort) return;
    ancestors.forEach((ancestor, level) => {
      const record = current.scores.get(ancestor);
      const after = ancestor.readability?.contentScore;
      if (!record || after === undefined) return;
      const points = total / dividerOf(level);
      check((before[level] ?? record.start) + points === after);
      record.received.push({ from: node, level, points });
    });
  }

  function finishScoring(current: Run) {
    for (const [node, record] of current.scores) {
      if (!record.isCandidate) continue;
      record.beforeLinks = (node as ScoredElement).readability?.contentScore ?? null;
      check(record.received.reduce((sum, part) => sum + part.points, record.start) === record.beforeLinks);
    }
    current.phase = "rank";
  }

  function noteSibling(current: Run, node: ScoredElement) {
    current.phase = "pick";
    current.pick.parent ??= node.parentElement;
    const text = original.getInnerText(node);
    current.pick.siblings.push({ node, className: node.className, isParagraph: node.nodeName === "P", textLength: text.length, linkShare: original.getLinkDensity(node), score: node.readability?.contentScore ?? null });
    seed(current, node, text);
  }

  internals.log = (...args: unknown[]) => {
    const [message, node] = args;
    if (typeof message !== "string") return;
    if (message === "Starting grabArticle loop") {
      attempt += 1;
      run = startRun(internals._doc.body, internals._articleByline);
      return;
    }
    if (!run) return;
    if (message.startsWith("Removing hidden node")) run.pendingRemoval = { reason: "hidden", match: "" };
    else if (message.startsWith("Removing unlikely candidate")) run.pendingRemoval = { reason: "unlikely", match: "" };
    else if (message.startsWith("Removing content with role")) run.pendingRemoval = { reason: "role", match: "" };
    else if (message.startsWith("Removing header:")) run.pendingRemoval = { reason: "title", match: "" };
    if (!isElement(node)) return;
    if (message === "Candidate:") {
      const record = run.scores.get(node);
      if (!record || record.beforeLinks === null || record.linkShare === null) return check(false);
      record.score = node.readability?.contentScore ?? null;
      check(record.beforeLinks * (1 - record.linkShare) === record.score);
    } else if (message === "Looking at sibling node:") noteSibling(run, node);
    else if (message === "Appending node:") run.pick.appended.add(node);
  };

  internals._removeAndGetNext = (node) => {
    if (run?.phase === "walk" || run?.phase === "walked") noteRemoval(run, node);
    return original.removeAndGetNext(node);
  };

  internals._getNextNode = (node, ignoreSelfAndKids) => {
    const next = original.getNextNode(node, ignoreSelfAndKids);
    if (run?.phase === "walk" && !next) run.phase = "walked";
    return next;
  };

  internals._setNodeTag = (node, tag) => {
    const replacement = original.setNodeTag(node, tag);
    if (run && replacement !== node) {
      run.places.replacementOf.set(node, replacement);
      run.places.originalOf.set(replacement, run.places.originalOf.get(node) ?? node);
    }
    return replacement;
  };

  internals._forEachNode = (nodes, fn) => {
    const current = run;
    if (current?.phase !== "walked") return original.forEachNode(nodes, fn);
    current.phase = "score";
    placeLiveBoxes(current.places, internals._doc.body);
    original.forEachNode(nodes, function (this: unknown, node, index) {
      scoreParagraph(current, node, () => fn.call(this, node, index));
    });
    finishScoring(current);
  };

  internals._initializeNode = (node) => {
    original.initializeNode(node);
    const current = run;
    const start = node.readability?.contentScore;
    if (!current || start === undefined) return;
    const namePoints = original.getClassWeight(node);
    current.scores.set(node, { start, namePoints, nameWords: nameWordsOf(node, namePoints), received: [], beforeLinks: null, linkShare: null, score: null, isCandidate: current.phase === "score" });
    seed(current, node, original.getInnerText(node));
  };

  internals._getLinkDensity = (element) => {
    const share = original.getLinkDensity(element);
    const record = run?.phase === "rank" ? run.scores.get(element) : undefined;
    if (record?.isCandidate && record.linkShare === null) record.linkShare = share;
    return share;
  };

  internals._prepArticle = (content) => {
    const current = run;
    if (current?.phase === "pick") {
      current.pick.content = content;
      current.pick.joined = new Set([...current.seeds.keys()].filter((node) => content.contains(latest(current.places, node))));
    }
    original.prepArticle(content);
    if (current?.phase === "pick") current.phase = "prepped";
  };

  internals._someNode = (nodes, fn) => {
    const current = run;
    const [parent, winner] = [nodes[0], nodes[1]];
    if (current?.phase === "prepped" && current.pick.content && parent === current.pick.parent && isElement(winner)) {
      const content = current.pick.content;
      current.pick.winner = winner;
      current.pick.kept = new Set([...current.seeds.keys()].filter((node) => content.contains(latest(current.places, node))));
      current.phase = "done";
    }
    return original.someNode(nodes, fn);
  };

  return (content: Node | null): ReadabilityTrace | null => {
    const done: Run | null = run;
    if (isBroken || !done || done.phase !== "done" || content !== done.pick.content || internals._attempts.length >= attempt) return null;
    return buildTrace(done, attempt);
  };
}

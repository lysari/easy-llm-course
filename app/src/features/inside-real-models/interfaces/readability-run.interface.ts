import type { BoxRemoval, NameWord, ParagraphPoints } from "./readability-trace.interface";

export type ScoredElement = Element & { readability?: { contentScore: number } };

export type RunPhase = "walk" | "walked" | "score" | "rank" | "pick" | "prepped" | "done";

export type PageKey = [number, number];

export interface BoxSeed {
  tag: string;
  name: string;
  preview: string;
  isAdded: boolean;
}

export interface PointsFrom {
  from: Element;
  level: number;
  points: number;
}

export interface ScoreRecord {
  start: number;
  namePoints: number;
  nameWords: NameWord[];
  received: PointsFrom[];
  beforeLinks: number | null;
  linkShare: number | null;
  score: number | null;
  isCandidate: boolean;
}

export interface SiblingSeen {
  node: ScoredElement;
  className: string;
  isParagraph: boolean;
  textLength: number;
  linkShare: number;
  score: number | null;
}

export interface Places {
  order: Map<Element, number>;
  pageParent: Map<Element, Element>;
  liveParent: Map<Element, Element>;
  liveKey: Map<Element, PageKey>;
  replacementOf: Map<Element, Element>;
  originalOf: Map<Element, Element>;
}

export interface RunPick {
  siblings: SiblingSeen[];
  parent: Element | null;
  appended: Set<Element>;
  content: Element | null;
  winner: ScoredElement | null;
  joined: Set<Element>;
  kept: Set<Element>;
}

export interface Run {
  phase: RunPhase;
  places: Places;
  byline: string | null;
  seeds: Map<Element, BoxSeed>;
  removals: Map<Element, BoxRemoval>;
  emptyCount: number;
  pendingRemoval: BoxRemoval | null;
  paragraphs: Map<Element, ParagraphPoints>;
  scores: Map<Element, ScoreRecord>;
  pick: RunPick;
}

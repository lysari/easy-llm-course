export type RemovalReason = "unlikely" | "role" | "hidden" | "dialog" | "byline" | "title";

export type SiblingReason = "winner" | "high-score" | "long-paragraph" | "short-sentence" | "low-score" | "no-score" | "paragraph-rules";

export interface BoxRemoval {
  reason: RemovalReason;
  match: string;
}

export interface ParagraphPoints {
  textLength: number;
  commaPoints: number;
  lengthPoints: number;
  total: number;
  isTooShort: boolean;
}

export interface NameWord {
  word: string;
  points: number;
}

export interface PointsIn {
  fromId: number;
  level: number;
  points: number;
}

export interface BoxScore {
  start: number;
  namePoints: number;
  nameWords: NameWord[];
  received: PointsIn[];
  beforeLinks: number;
  linkShare: number | null;
  score: number;
}

export interface SiblingCheck {
  reason: SiblingReason;
  bonus: number;
  textLength: number;
}

export interface PageBox {
  id: number;
  depth: number;
  tag: string;
  name: string;
  preview: string;
  isAdded: boolean;
  removal: BoxRemoval | null;
  paragraph: ParagraphPoints | null;
  score: BoxScore | null;
  sibling: SiblingCheck | null;
  isJoined: boolean;
  isKept: boolean;
}

export interface ReadabilityTrace {
  attempt: number;
  boxes: PageBox[];
  emptyCount: number;
  winnerId: number;
  bestId: number;
  threshold: number;
}

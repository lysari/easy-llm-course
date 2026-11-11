import { type CSSProperties, useEffect, useMemo, useRef } from "react";
import type { PageBox, ReadabilityTrace, RemovalReason } from "@/features/inside-real-models/interfaces/readability-trace.interface";
import { boxLabel, showPoints } from "./how-it-works-steps";

type Tone = "drop" | "keep" | "win" | "score" | "gone";

interface Badge {
  tone: Tone;
  mark: string;
  text: string;
}

type BoxRow = { kind: "box"; box: PageBox; more: PageBox[] } | { kind: "strip"; boxes: PageBox[] };

const REMOVAL_TEXT: Record<Exclude<RemovalReason, "unlikely" | "role">, string> = {
  title: "repeats the title",
  hidden: "hidden",
  dialog: "pop-up",
  byline: "author line",
};

const isLeafParagraph = (box: PageBox, next: PageBox | undefined) => box.paragraph !== null && !box.score && !box.removal && (!next || next.depth <= box.depth);

const isSameRemoval = (box: PageBox, other: PageBox) => box.depth === other.depth && box.removal?.reason === other.removal?.reason && box.removal?.match.toLowerCase() === other.removal?.match.toLowerCase();

function rowsOf(boxes: PageBox[]) {
  const rows: BoxRow[] = [];
  boxes.forEach((box, i) => {
    const last = rows[rows.length - 1];
    if (isLeafParagraph(box, boxes[i + 1])) {
      if (last?.kind === "strip" && last.boxes[0].depth === box.depth) last.boxes.push(box);
      else rows.push({ kind: "strip", boxes: [box] });
    } else if (box.removal && last?.kind === "box" && last.box.removal && isSameRemoval(last.box, box)) last.more.push(box);
    else rows.push({ kind: "box", box, more: [] });
  });
  return rows;
}

function removalText(box: PageBox) {
  if (!box.removal) return "";
  const { reason, match } = box.removal;
  if (reason === "unlikely") return `removed · “${match.toLowerCase()}”`;
  if (reason === "role") return `removed · role “${match}”`;
  return `removed · ${REMOVAL_TEXT[reason]}`;
}

function siblingBadge(box: PageBox, trace: ReadabilityTrace): Badge | null {
  if (!box.sibling) return null;
  const { reason, bonus, textLength } = box.sibling;
  const score = (box.score?.score ?? 0) + bonus;
  const sameClass = bonus ? " with its same-class bonus" : "";
  if (reason === "winner") return { tone: "win", mark: "★", text: `winner · ${showPoints(box.score?.score ?? 0)}` };
  if (reason === "high-score") return { tone: "keep", mark: "✓", text: `joins · ${showPoints(score)}${sameClass} ≥ ${showPoints(trace.threshold)}` };
  if (reason === "long-paragraph") return { tone: "keep", mark: "✓", text: `joins · a ${textLength}-character paragraph, few links` };
  if (reason === "short-sentence") return { tone: "keep", mark: "✓", text: "joins · a short sentence, no links" };
  if (reason === "low-score") return { tone: "drop", mark: "✕", text: `stays out · ${showPoints(score)}${sameClass} < ${showPoints(trace.threshold)}` };
  if (reason === "no-score") return { tone: "drop", mark: "✕", text: "stays out · no score" };
  return { tone: "drop", mark: "✕", text: "stays out · too short or too many links" };
}

function outerIdsOf(trace: ReadabilityTrace) {
  const ids = new Set<number>();
  let depth = trace.boxes[trace.winnerId].depth;
  for (let i = trace.winnerId - 1; i >= 0 && depth > 0; i -= 1) {
    if (trace.boxes[i].depth >= depth) continue;
    ids.add(i);
    depth = trace.boxes[i].depth;
  }
  return ids;
}

function badgeOf(box: PageBox, step: number, trace: ReadabilityTrace, outerIds: Set<number>): Badge | null {
  if (step === 0) return null;
  if (step === 1) return box.removal ? { tone: "drop", mark: "✕", text: removalText(box) } : null;
  if (box.removal && step < 4) return { tone: "gone", mark: "✕", text: "removed" };
  if (step === 2) return box.score ? { tone: box.id === trace.bestId ? "win" : "score", mark: "", text: `score ${showPoints(box.score.score)}` } : null;
  if (step === 3) return siblingBadge(box, trace) ?? (box.score ? { tone: "gone", mark: "", text: `score ${showPoints(box.score.score)}` } : null);
  if (box.isKept) return { tone: "keep", mark: "✓", text: "kept" };
  if (box.isJoined) return { tone: "gone", mark: "✕", text: "tidied away" };
  return { tone: "gone", mark: "✕", text: outerIds.has(box.id) ? "the rest dropped" : "dropped" };
}

function chipMark(box: PageBox, step: number) {
  if (step === 1 || step === 0) return "";
  if (step === 4) return box.isKept ? "✓ kept" : "✕ dropped";
  if (box.paragraph?.isTooShort) return "too short";
  return `+${showPoints(box.paragraph?.total ?? 0)}`;
}

const isGone = (box: PageBox, step: number) => (step >= 2 && step < 4 && box.removal !== null) || (step === 4 && !box.isKept);

const isQuiet = (box: PageBox, step: number) => step === 3 && !box.removal && !box.sibling && !box.isJoined;

const stateOf = (box: PageBox, step: number) => (isGone(box, step) ? " is-gone" : isQuiet(box, step) ? " is-quiet" : "");

function BadgeView({ badge }: { badge: Badge }) {
  return (
    <span className={`box-badge box-badge--${badge.tone}`}>
      {badge.mark && <span aria-hidden="true">{badge.mark} </span>}
      {badge.text}
    </span>
  );
}

const focusIdOf = (trace: ReadabilityTrace, step: number) => (step < 2 ? null : step === 2 ? trace.bestId : trace.winnerId);

export function PageBoxes({ trace, step }: { trace: ReadabilityTrace; step: number }) {
  const listRef = useRef<HTMLOListElement>(null);
  const outerIds = useMemo(() => outerIdsOf(trace), [trace]);

  useEffect(() => {
    const list = listRef.current;
    if (!list) return;
    const focusId = focusIdOf(trace, step);
    const row = focusId === null ? null : list.querySelector<HTMLElement>(`[data-box="${focusId}"]`);
    const isReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    list.scrollTo({ top: row ? row.offsetTop - list.clientHeight / 3 : 0, behavior: isReduced ? "auto" : "smooth" });
  }, [trace, step]);

  return (
    <ol ref={listRef} className="boxes" tabIndex={0} aria-label="The page's boxes, in page order">
      {rowsOf(trace.boxes).map((row) => {
        if (row.kind === "strip") {
          const first = row.boxes[0];
          return (
            <li key={first.id} className="boxes-strip" style={{ "--how-depth": first.depth } as CSSProperties}>
              <ul aria-label={`${row.boxes.length} paragraph${row.boxes.length === 1 ? "" : "s"}`}>
                {row.boxes.map((box) => (
                  <li key={box.id} className={`box-chip${stateOf(box, step)}${step === 4 && box.isKept ? " is-kept" : ""}`} title={box.preview}>
                    <span className="box-name">{boxLabel(box)}</span>
                    <span className="box-preview">{box.preview || "(no text)"}</span>
                    {chipMark(box, step) && <b>{chipMark(box, step)}</b>}
                  </li>
                ))}
              </ul>
            </li>
          );
        }
        const { box, more } = row;
        const badge = badgeOf(box, step, trace, outerIds);
        const isWinner = step >= 3 && box.id === trace.winnerId;
        const previews = [box, ...more].map((each) => each.preview).filter(Boolean);
        return (
          <li key={box.id} data-box={box.id} className={`box-row${step === 4 && outerIds.has(box.id) ? " is-quiet" : stateOf(box, step)}${step === 1 && box.removal ? " is-struck" : ""}${isWinner ? " is-winner" : ""}`} style={{ "--how-depth": box.depth } as CSSProperties}>
            <span className="box-name">
              {boxLabel(box)}
              {more.length > 0 && <span className="box-more">{` and ${more.length} more like it`}</span>}
            </span>
            <span className="box-preview">{previews.join(" · ") || "(no text)"}</span>
            {badge && <BadgeView badge={badge} />}
          </li>
        );
      })}
    </ol>
  );
}

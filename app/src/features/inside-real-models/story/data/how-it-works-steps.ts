import type { PageBox, ReadabilityTrace, RemovalReason } from "@/features/inside-real-models/interfaces/readability-trace.interface";
import { count } from "@/shared/utils/format";
import { JOINING_REASONS, MIN_SIBLING_SCORE } from "./readability-run";
import { MIN_PARAGRAPH_LENGTH } from "./readability-trace";

export interface WalkStep {
  title: string;
  sentence: string;
}

const joinedCountOf = (trace: ReadabilityTrace) => trace.boxes.filter((box) => box.id !== trace.winnerId && box.sibling && JOINING_REASONS.includes(box.sibling.reason)).length;

const REMOVAL_PHRASES: Record<Exclude<RemovalReason, "unlikely">, [string, string]> = {
  title: ["heading that repeats the page title", "headings that repeat the page title"],
  hidden: ["hidden box", "hidden boxes"],
  role: ["box marked as a menu, sidebar or alert", "boxes marked as a menu, sidebar or alert"],
  dialog: ["pop-up box", "pop-up boxes"],
  byline: ["author line, kept apart as the byline", "author lines"],
};

const TRY_RULES = ["this time without deleting the unlikely boxes", "this time also without the ±25 points for class and id names", "this time also without its last tidy-up"];

export const showPoints = (value: number) => (Number.isInteger(value) ? count(value) : value.toFixed(2));

export const boxLabel = (box: PageBox) => (box.isAdded ? `new ${box.tag}` : `${box.tag}${box.name}`);

const plural = (amount: number, one: string, many: string) => `${count(amount)} ${amount === 1 ? one : many}`;

const listOf = (parts: string[]) => (parts.length < 2 ? (parts[0] ?? "") : `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`);

function removalSentence(trace: ReadabilityTrace) {
  const removed = trace.boxes.filter((box) => box.removal);
  const unlikely = removed.filter((box) => box.removal?.reason === "unlikely");
  const words = [...new Set(unlikely.map((box) => `“${box.removal?.match.toLowerCase()}”`))];
  const parts = unlikely.length ? [`${count(unlikely.length)} with a word from its “unlikely” list in their class or id (here ${listOf(words)})`] : [];
  for (const [reason, [one, many]] of Object.entries(REMOVAL_PHRASES)) {
    const found = removed.filter((box) => box.removal?.reason === reason).length;
    if (found) parts.push(plural(found, one, many));
  }
  const spared = unlikely.length ? " A box whose name also says article, content or main is spared." : "";
  const empty = trace.emptyCount ? ` It also drops ${plural(trace.emptyCount, "empty box", "empty boxes")}.` : "";
  if (!parts.length) return `Before scoring, it looks for boxes that are almost never the article, such as menus, footers and sidebars, and finds none here.${empty}`;
  return `Before scoring anything, it deletes ${plural(removed.length, "box that is", "boxes that are")} almost never the article: ${listOf(parts)}.${spared}${empty}`;
}

function winnerSentence(trace: ReadabilityTrace, winner: PageBox, best: PageBox) {
  const siblings = trace.boxes.filter((box) => box.sibling && box.id !== winner.id);
  const joined = joinedCountOf(trace);
  const moved = best.id === winner.id ? "" : ` The top score was ${boxLabel(best)}'s, ${showPoints(best.score?.score ?? 0)}, but Readability moved up to the bigger box around it, because close runners-up sat inside it, its parent scored higher, or it was an only child.`;
  const beside = siblings.length ? `here ${joined ? plural(joined, "box joins", "boxes join") : "no box joins"} and ${plural(siblings.length - joined, "box stays", "boxes stay")} out` : "here it has no boxes beside it";
  return `${boxLabel(winner)} wins with ${showPoints(winner.score?.score ?? 0)}, so it becomes the article; a box beside it joins only if it scores at least ${showPoints(trace.threshold)} (a fifth of the winner's score, and never under ${MIN_SIBLING_SCORE}) or is a paragraph that reads like text: ${beside}.${moved}`;
}

export function walkStepsOf(trace: ReadabilityTrace): WalkStep[] {
  const winner = trace.boxes[trace.winnerId];
  const best = trace.boxes[trace.bestId] ?? winner;
  const joined = joinedCountOf(trace);
  const tryNote = trace.attempt > 1 ? ` Its first ${plural(trace.attempt - 1, "try", "tries")} kept under 500 characters, so these steps follow try ${trace.attempt}, ${TRY_RULES[trace.attempt - 2]}.` : "";
  return [
    {
      title: "The page as boxes",
      sentence: `Readability deletes the page's scripts and styles, then reads the rest as boxes inside boxes: each box is one HTML tag, often with a class or id name, and here are the ${count(trace.boxes.length)} it acts on, in page order.${tryNote}`,
    },
    { title: "Unlikely boxes go first", sentence: removalSentence(trace) },
    {
      title: "Paragraphs earn points, boxes collect them",
      sentence: `Each paragraph of ${MIN_PARAGRAPH_LENGTH}+ characters earns points and hands them up to the boxes around it, and each box adds them to its own start, then keeps only the share of its text that is not links.`,
    },
    { title: "The top box wins, close neighbours join", sentence: winnerSentence(trace, winner, best) },
    {
      title: "Everything else is dropped",
      sentence: `Only ${boxLabel(winner)}${joined ? ` and the ${plural(joined, "box", "boxes")} that joined it are` : " is"} kept; Readability tidies ${joined ? "them" : "it"} (it removes forms, buttons, share links and small boxes full of links or pictures), and the plain text left over is the one document that goes into the training pile.`,
    },
  ];
}

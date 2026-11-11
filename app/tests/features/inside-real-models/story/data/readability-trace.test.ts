import { afterAll, describe, expect, test } from "bun:test";
import { Readability } from "@mozilla/readability";
import { Window } from "happy-dom";
import type { PageBox } from "@/features/inside-real-models/interfaces/readability-trace.interface";
import { extractPage } from "@/features/inside-real-models/story/data/extract-page";
import { boxLabel } from "@/features/inside-real-models/story/data/how-it-works-steps";
import { MIN_SIBLING_SCORE, SIBLING_SHARE } from "@/features/inside-real-models/story/data/readability-run";
import { SAVED_PAGE } from "@/features/inside-real-models/story/data/saved-page";

const window = new Window({ settings: { disableJavaScriptEvaluation: true, disableJavaScriptFileLoading: true, disableCSSFileLoading: true, disableIframePageLoading: true } });

const docOf = (html: string) => new window.DOMParser().parseFromString(html, "text/html") as unknown as Document;

const extractOf = (html: string) => extractPage(docOf(html), html);

const dividerOf = (level: number) => (level === 0 ? 1 : level === 1 ? 2 : level * 3);

const LONG = "Bees carry pollen from flower to flower, and that is how many plants make the seeds they need, so a field with bees gives more fruit, more nuts and more seeds than a field without them.";

const article = (className: string) => `<html><head><title>Bees</title></head><body><nav class="menu"><a href="/">Home</a> <a href="/about">About us</a></nav><div class="${className}"><p>${LONG}</p><p>${LONG}</p><p>${LONG}</p><p>${LONG}</p></div><div class="sidebar"><p>Buy honey here, cheap.</p></div><footer>Copyright</footer></body></html>`;

function realCandidateScores(html: string) {
  const reader = new Readability(docOf(html));
  const scores: string[] = [];
  (reader as unknown as { log: (...args: unknown[]) => void }).log = (message, node, score) => {
    if (message !== "Candidate:" || typeof node !== "object" || node === null || !("tagName" in node)) return;
    const element = node as Element;
    const classes = (element.getAttribute("class") ?? "").trim().split(/\s+/).filter(Boolean);
    scores.push(`${element.tagName.toLowerCase()}${classes.map((name) => `.${name}`).join("")}${element.id ? `#${element.id}` : ""} ${score}`);
  };
  reader.parse();
  return scores.sort();
}

afterAll(() => window.happyDOM.close());

describe("the saved page's walkthrough", () => {
  const extract = extractOf(SAVED_PAGE.html);
  const trace = extract.trace;
  if (!trace) throw new Error("The saved page has no walkthrough");
  const scored = trace.boxes.filter((box): box is PageBox & { score: NonNullable<PageBox["score"]> } => box.score !== null);
  const winner = trace.boxes[trace.winnerId];

  test("shows exactly the scores Readability itself reports for each box", () => {
    expect(scored.length).toBeGreaterThan(0);
    expect(scored.map((box) => `${boxLabel(box)} with score ${box.score.score}`).sort()).toEqual(realCandidateScores(SAVED_PAGE.html));
  });

  test("each score's parts add up exactly to that score, in Readability's own order", () => {
    for (const box of scored) {
      const { start, namePoints, nameWords, received, beforeLinks, linkShare, score } = box.score;
      expect(nameWords.reduce((sum, word) => sum + word.points, 0), boxLabel(box)).toBe(namePoints);
      expect(Number.isInteger(start - namePoints)).toBe(true);
      expect(received.reduce((sum, part) => sum + part.points, start), boxLabel(box)).toBe(beforeLinks);
      expect(linkShare).not.toBeNull();
      expect(beforeLinks * (1 - (linkShare ?? 0)), boxLabel(box)).toBe(score);
    }
  });

  test("each paragraph's points add up, and each box above gets its real share of them", () => {
    for (const box of trace.boxes) {
      const points = box.paragraph;
      if (!points || points.isTooShort) continue;
      expect(points.total).toBe(1 + points.commaPoints + points.lengthPoints);
      expect(points.lengthPoints).toBe(Math.min(Math.floor(points.textLength / 100), 3));
    }
    for (const box of scored) {
      for (const part of box.score.received) expect(part.points).toBe((trace.boxes[part.fromId].paragraph?.total ?? -1) / dividerOf(part.level));
    }
  });

  test("names the real removals: the menus, the header, the footers and the title heading", () => {
    const removed = trace.boxes.filter((box) => box.removal).map((box) => `${boxLabel(box)} ${box.removal?.reason} ${box.removal?.match.toLowerCase()}`);
    expect(removed).toContain("div.header unlikely header");
    expect(removed).toContain("div.topMenuNavList.section-link unlikely menu");
    expect(removed).toContain("div.footer-legal unlikely footer");
    expect(removed).toContain("h1.location-pagetitle title ");
    expect(trace.emptyCount).toBeGreaterThan(0);
  });

  test("the winner is the top score, and its text is the kept document", () => {
    expect(boxLabel(winner)).toBe("div.two-third-first");
    expect(trace.bestId).toBe(trace.winnerId);
    expect(Math.max(...scored.map((box) => box.score.score))).toBe(winner.score?.score ?? -1);
    expect(winner.isKept).toBe(true);
    expect(extract.text.replace(/\s+/g, " ")).toStartWith(winner.preview);
    for (const box of trace.boxes.filter((each) => each.isKept && each.preview)) expect(extract.text.replace(/\s+/g, " ")).toContain(box.preview.slice(0, 40));
  });

  test("the side box scores under the bar, so it stays out and its text is dropped", () => {
    expect(trace.threshold).toBe(Math.max(MIN_SIBLING_SCORE, (winner.score?.score ?? 0) * SIBLING_SHARE));
    const side = trace.boxes.find((box) => boxLabel(box) === "div.one-third-last");
    expect(side?.sibling?.reason).toBe("low-score");
    expect(side?.score?.score).toBeLessThan(trace.threshold);
    expect(side?.isKept).toBe(false);
    expect(extract.text).not.toContain("How Hot is Lightning?");
    for (const box of trace.boxes.filter((each) => each.removal)) expect(box.isKept).toBe(false);
  });
});

describe("a pasted page's walkthrough", () => {
  test("follows the same steps, and its parts still add up to Readability's scores", () => {
    const html = article("story");
    const trace = extractOf(html).trace;
    expect(trace?.attempt).toBe(1);
    const winner = trace?.boxes[trace.winnerId];
    expect(winner && boxLabel(winner)).toBe("div.story");
    expect(trace?.boxes.find((box) => box.removal?.match === "menu")).toBeDefined();
    const scores = (trace?.boxes ?? []).filter((box) => box.score).map((box) => `${boxLabel(box)} with score ${box.score?.score}`);
    expect(scores.sort()).toEqual(realCandidateScores(html));
  });

  test("when the first try keeps too little, it follows the try that passed", () => {
    const trace = extractOf(article("comment-box")).trace;
    expect(trace?.attempt).toBe(3);
    expect(trace?.boxes.some((box) => box.removal?.reason === "unlikely")).toBe(false);
    expect(trace?.boxes.every((box) => !box.score || box.score.namePoints === 0)).toBe(true);
  });

  test("a page that never reaches 500 characters of text gets no walkthrough", () => {
    expect(extractOf("<html><body><article><h1>Bees</h1><p>Bees carry pollen from flower to flower.</p></article></body></html>").trace).toBeNull();
  });
});

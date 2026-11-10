import { afterAll, describe, expect, test } from "bun:test";
import { Window } from "happy-dom";
import { checkCopy, fingerprint, normalWords, shinglesOf } from "@/features/take-apart/story/data/dedupe";
import { extractPage } from "@/features/take-apart/story/data/extract-page";
import { GPT_2_KARMA, checkQuality, keepsDocument, passesKarma } from "@/features/take-apart/story/data/quality-rules";
import { SAVED_PAGE } from "@/features/take-apart/story/data/saved-page";

const window = new Window({ settings: { disableJavaScriptEvaluation: true, disableJavaScriptFileLoading: true, disableCSSFileLoading: true, disableIframePageLoading: true } });
const page = extractPage(new window.DOMParser().parseFromString(SAVED_PAGE.html, "text/html") as unknown as Document, SAVED_PAGE.html);

afterAll(() => window.happyDOM.close());

describe("the quality check", () => {
  test("keeps the saved page's main text and drops its menus", () => {
    expect(keepsDocument(checkQuality(page.text))).toBe(true);
    expect(keepsDocument(checkQuality(page.droppedText.join("\n")))).toBe(false);
  });

  test("drops empty and very short text without crashing", () => {
    expect(keepsDocument(checkQuality(""))).toBe(false);
    expect(checkQuality("").every((rule) => !rule.measured.includes("NaN"))).toBe(true);
    expect(keepsDocument(checkQuality("Too short."))).toBe(false);
  });

  test("counts words in every script", () => {
    expect(checkQuality("ខ្ញុំ ១២៣ dog")[0].measured).toBe("3 words");
  });

  test("GPT-2's real rule keeps a link at 3 karma or more", () => {
    expect(GPT_2_KARMA).toBe(3);
    expect(passesKarma(3)).toBe(true);
    expect(passesKarma(2)).toBe(false);
  });
});

describe("the deduplication check", () => {
  const pile = page.text.split("\n").filter((line) => normalWords(line).length >= 8);

  test("lower-cases and keeps only words, as the GPT-2 paper's overlap check did", () => {
    expect(normalWords("The DOG, the dog!")).toEqual(["the", "dog", "the", "dog"]);
    expect(fingerprint("The DOG, the dog!")).toBe(fingerprint("the dog the dog"));
  });

  test("finds an exact copy, a near copy and a new document", () => {
    const sample = pile.reduce((longest, line) => (line.length > longest.length ? line : longest), "");
    expect(checkCopy(sample, pile).verdict).toBe("copy");
    expect(checkCopy(sample.replace(/^\S+/u, "Changed"), pile).verdict).toBe("near-copy");
    expect(checkCopy("A completely different paragraph about cats who sleep all day long in the sun.", pile).verdict).toBe("new");
  });

  test("an empty pile keeps anything, and short text has no 8-word runs", () => {
    expect(checkCopy("anything at all", []).verdict).toBe("new");
    expect(shinglesOf("only five words are here").size).toBe(0);
  });
});

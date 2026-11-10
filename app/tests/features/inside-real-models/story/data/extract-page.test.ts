import { afterAll, describe, expect, test } from "bun:test";
import { Window } from "happy-dom";
import packageJson from "../../../../../package.json";
import type { PageExtract } from "@/features/inside-real-models/interfaces/page-extract.interface";
import { MAX_PAGE_LENGTH, READABILITY_VERSION, extractPage, pageProblem } from "@/features/inside-real-models/story/data/extract-page";
import { SAVED_PAGE } from "@/features/inside-real-models/story/data/saved-page";

const window = new Window({ settings: { disableJavaScriptEvaluation: true, disableJavaScriptFileLoading: true, disableCSSFileLoading: true, disableIframePageLoading: true } });

const extractOf = (html: string) => extractPage(new window.DOMParser().parseFromString(html, "text/html") as unknown as Document, html);

const removedOf = (extract: PageExtract) => extract.otherTextLength + extract.codeLength + extract.markupLength;

afterAll(() => window.happyDOM.close());

describe("the saved page", () => {
  test("records where it came from, when and why it may be copied", () => {
    expect(SAVED_PAGE.url).toBe("https://www.weather.gov/safety/lightning-science-overview");
    expect(SAVED_PAGE.fetchedOn).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(SAVED_PAGE.licence).toContain("public domain");
    expect(SAVED_PAGE.licenceUrl).toBe("https://www.weather.gov/disclaimer");
  });

  test("is the whole page as fetched, under 200 KB", () => {
    expect(SAVED_PAGE.html.length).toBe(29_218);
    expect(SAVED_PAGE.html).toStartWith("<!DOCTYPE html");
    expect(SAVED_PAGE.html.trimEnd()).toEndWith("</html>");
    expect(SAVED_PAGE.html).toContain(`<title>${SAVED_PAGE.title}</title>`);
  });
});

describe("copying the text out of the saved page", () => {
  const extract = extractOf(SAVED_PAGE.html);

  test("keeps the article's title and its opening sentence", () => {
    expect(extract.title).toBe("Understanding Lightning Science");
    expect(extract.text).toStartWith("Lightning is fascinating to watch but also extremely dangerous.");
    expect(extract.text).toContain("\nThunder\nThunder is the sound made by a flash of lightning.");
  });

  test("throws away the menus, the footer and the scripts", () => {
    expect(extract.text).not.toContain("FORECAST");
    expect(extract.text).not.toContain("US Dept of Commerce");
    expect(extract.text).not.toContain("googletagmanager");
    expect(extract.droppedText).toContain("FORECAST");
    expect(extract.droppedText).toContain("US Dept of Commerce");
    expect(extract.scriptCount).toBeGreaterThan(0);
    expect(extract.codeLength).toBeGreaterThan(0);
  });

  test("its counts add up to the whole page", () => {
    expect(extract.pageLength).toBe(SAVED_PAGE.html.length);
    expect(extract.keptLength).toBe(extract.text.length);
    expect(removedOf(extract)).toBeGreaterThan(0);
    expect(extract.keptLength + removedOf(extract)).toBe(extract.pageLength);
    for (const part of [extract.otherTextLength, extract.codeLength, extract.markupLength]) expect(part).toBeGreaterThanOrEqual(0);
  });
});

describe("a pasted page", () => {
  test("a small page is cleaned the same way, and its counts add up", () => {
    const html = "<html><head><style>p { color: red; }</style></head><body><nav><a href='/'>Home</a> <a href='/about'>About us</a></nav><article><h1>Bees</h1><p>Bees carry pollen from flower to flower, and that is how many plants make seeds.</p></article><footer>Copyright</footer><script>track()</script></body></html>";
    const extract = extractOf(html);
    expect(extract.text).toContain("Bees\nBees carry pollen from flower to flower, and that is how many plants make seeds.");
    expect(extract.styleCount).toBe(1);
    expect(extract.scriptCount).toBe(1);
    expect(extract.codeLength).toBe("p { color: red; }".length + "track()".length);
    expect(extract.keptLength + removedOf(extract)).toBe(html.length);
  });

  test("a page with no main text keeps nothing, and still adds up", () => {
    const html = "<html><body><script>start()</script></body></html>";
    const extract = extractOf(html);
    expect(extract.text).toBe("");
    expect(extract.keptLength + removedOf(extract)).toBe(html.length);
  });

  test("empty, plain-text and huge input is turned away with a reason", () => {
    expect(pageProblem("   ")).toContain("Paste");
    expect(pageProblem("just some words")).toContain("doesn't look like HTML");
    expect(pageProblem(`<p>${"a".repeat(MAX_PAGE_LENGTH)}</p>`)).toContain("Paste a page under");
    expect(pageProblem("<p>Hi</p>")).toBeNull();
  });
});

test("the demo names the Readability version the app installs", () => {
  expect(packageJson.dependencies["@mozilla/readability"]).toBe(READABILITY_VERSION);
});

import { Readability } from "@mozilla/readability";
import type { PageExtract } from "@/features/inside-real-models/interfaces/page-extract.interface";
import { count } from "@/shared/utils/format";

export const READABILITY_VERSION = "0.6.0";

export const MAX_PAGE_LENGTH = 2_000_000;

const TEXT_NODE = 3;
const CODE_TAGS = ["SCRIPT", "STYLE"];
const BLOCK_TAGS = new Set(["ADDRESS", "ARTICLE", "ASIDE", "BLOCKQUOTE", "BR", "DD", "DIV", "DL", "DT", "FIGCAPTION", "FIGURE", "FOOTER", "FORM", "H1", "H2", "H3", "H4", "H5", "H6", "HEADER", "HR", "LI", "MAIN", "NAV", "OL", "P", "PRE", "SECTION", "TABLE", "TD", "TH", "TR", "UL"]);

const plainLine = (line: string) => line.replace(/\s+/g, " ").trim();

const plainText = (text: string) => text.split("\n").map(plainLine).filter(Boolean).join("\n");

function readText(node: Node, pieces: string[]) {
  if (node.nodeType === TEXT_NODE) pieces.push(node.textContent ?? "");
  for (const child of node.childNodes) {
    if (!CODE_TAGS.includes(child.nodeName.toUpperCase())) readText(child, pieces);
  }
  if (BLOCK_TAGS.has(node.nodeName.toUpperCase())) pieces.push("\n");
}

function blockText(node: Node | null | undefined) {
  const pieces: string[] = [];
  if (node) readText(node, pieces);
  return plainText(pieces.join(""));
}

const codeLengthOf = (elements: HTMLCollectionOf<Element>) => [...elements].reduce((sum, element) => sum + (element.textContent ?? "").length, 0);

export function pageProblem(html: string) {
  if (!html.trim()) return "Paste a page's HTML first.";
  if (html.length > MAX_PAGE_LENGTH) return `That is ${count(html.length)} characters. Paste a page under ${count(MAX_PAGE_LENGTH)}.`;
  if (!/<[a-z!][^>]*>/i.test(html)) return "That doesn't look like HTML: it has no tags such as <p>. Copy the page's source, not the text you see.";
  return null;
}

export function extractPage(doc: Document, html: string): PageExtract {
  const pieces: string[] = [];
  readText(doc.documentElement, pieces);
  const pageText = plainText(pieces.join(""));
  const scripts = doc.getElementsByTagName("script");
  const styles = doc.getElementsByTagName("style");
  const scriptCount = scripts.length;
  const styleCount = styles.length;
  const codeLength = codeLengthOf(scripts) + codeLengthOf(styles);
  const article = new Readability<Node>(doc, { serializer: (node) => node }).parse();
  const text = blockText(article?.content);
  const otherTextLength = Math.max(0, pageText.length - text.length);
  const droppedText = [...new Set(pieces.map(plainLine).filter((piece) => piece && !text.includes(piece)))];
  return {
    title: plainLine(article?.title ?? ""),
    text,
    pageLength: html.length,
    keptLength: text.length,
    otherTextLength,
    codeLength,
    markupLength: html.length - text.length - otherTextLength - codeLength,
    scriptCount,
    styleCount,
    droppedText,
  };
}

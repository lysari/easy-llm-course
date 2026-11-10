import { useState } from "react";
import type { PageExtract } from "@/features/take-apart/interfaces/page-extract.interface";
import { extractPage, pageProblem } from "@/features/take-apart/story/extract-page";
import { SAVED_PAGE } from "@/features/take-apart/story/saved-page";

type PageFrom = "saved" | "pasted";

interface ShownExtract {
  from: PageFrom;
  extract: PageExtract;
}

const extractHtml = (html: string) => extractPage(new DOMParser().parseFromString(html, "text/html"), html);

export function usePageExtract() {
  const [shown, setShown] = useState<ShownExtract | null>(null);
  const [draft, setDraft] = useState("");
  const [problem, setProblem] = useState<string | null>(null);

  function cleanSaved() {
    setProblem(null);
    setShown({ from: "saved", extract: extractHtml(SAVED_PAGE.html) });
  }

  function cleanPasted() {
    const found = pageProblem(draft);
    setProblem(found);
    if (found) setShown((current) => (current?.from === "pasted" ? null : current));
    else setShown({ from: "pasted", extract: extractHtml(draft) });
  }

  return { shown, draft, setDraft, problem, cleanSaved, cleanPasted };
}

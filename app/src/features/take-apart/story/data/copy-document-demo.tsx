import "./data.css";
import { useId } from "react";
import { usePageExtract } from "@/features/take-apart/hooks/use-page-extract";
import { SOURCES } from "@/features/take-apart/trees/sources";
import { count } from "@/shared/utils/format";
import { READABILITY_VERSION } from "./extract-page";
import { PageExtractResult } from "./page-extract-result";
import { SAVED_PAGE } from "./saved-page";

const READABILITY_URL = "https://github.com/mozilla/readability";

export function CopyDocumentDemo() {
  const page = usePageExtract();
  const pasteId = useId();

  return (
    <div className="watch">
      <p className="note">
        A real web page, saved in this app exactly as it came:{" "}
        <a href={SAVED_PAGE.url} target="_blank" rel="noreferrer">
          {SAVED_PAGE.title}
        </a>
        , {SAVED_PAGE.site}, saved {SAVED_PAGE.fetchedOn}.{" "}
        <a className="card-lesson" href={SAVED_PAGE.licenceUrl} target="_blank" rel="noreferrer">
          {SAVED_PAGE.licence}
        </a>
      </p>
      <figure className="watch-pane">
        <figcaption className="card-lesson">{`The page as it arrives: ${count(SAVED_PAGE.html.length)} characters of HTML`}</figcaption>
        <pre className="watch-code" tabIndex={0} role="region" aria-label="The saved page's HTML">
          {SAVED_PAGE.html}
        </pre>
      </figure>
      <button type="button" className="action action--strong watch-run" onClick={page.cleanSaved}>
        Copy the text out of this page
      </button>
      {page.shown?.from === "saved" && <PageExtractResult extract={page.shown.extract} />}
      <div className="watch-paste">
        <label className="section-title" htmlFor={pasteId}>
          Try your own page
        </label>
        <p className="card-lesson">The browser can't fetch another site's page for us, because sites block that. So open any page, choose View source, select all, copy, and paste it here. It never leaves your browser.</p>
        <textarea id={pasteId} className="watch-input" rows={5} value={page.draft} placeholder="<html> …" spellCheck={false} onChange={(event) => page.setDraft(event.target.value)} />
        <div className="watch-paste-foot">
          <span className="card-lesson">{`${count(page.draft.length)} characters`}</span>
          <button type="button" className="action action--strong" onClick={page.cleanPasted}>
            Copy the text out of my page
          </button>
        </div>
        {page.problem && (
          <p className="watch-problem" role="alert">
            {page.problem}
          </p>
        )}
      </div>
      {page.shown?.from === "pasted" && <PageExtractResult extract={page.shown.extract} />}
      <p className="card-lesson">
        Cleaned live in your browser by{" "}
        <a href={READABILITY_URL} target="_blank" rel="noreferrer">
          Mozilla Readability {READABILITY_VERSION}
        </a>
        , the tool behind Firefox Reader View. GPT-2's makers used the Dragnet and Newspaper extractors instead (
        <a href={SOURCES["gpt-2-paper"].url} target="_blank" rel="noreferrer">
          GPT-2 paper
        </a>
        ), and Meta has not published what Llama 3 used, so their text for this page would differ.
      </p>
    </div>
  );
}

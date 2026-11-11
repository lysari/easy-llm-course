import { useEffect, useRef } from "react";
import type { PageExtract } from "@/features/inside-real-models/interfaces/page-extract.interface";
import { count, percent } from "@/shared/utils/format";
import { HowItWorks } from "./how-it-works";

type ShareKind = "kept" | "text" | "code" | "markup";

const shareOf = (length: number, total: number) => (length === 0 ? "0%" : percent(length / total));

const plural = (amount: number, word: string) => `${count(amount)} ${word}${amount === 1 ? "" : "s"}`;

function Share({ kind, label, length, total }: { kind: ShareKind; label: string; length: number; total: number }) {
  return (
    <li className={`extract-share extract-colour--${kind}`}>
      <span className="extract-swatch" />
      <span>{label}</span>
      <b>{count(length)}</b>
      <span>{shareOf(length, total)}</span>
    </li>
  );
}

export function PageExtractResult({ extract }: { extract: PageExtract }) {
  const headlineRef = useRef<HTMLParagraphElement>(null);
  const keptRef = useRef<HTMLPreElement>(null);
  const total = extract.pageLength;

  useEffect(() => {
    headlineRef.current?.focus();
  }, [extract]);

  function showKept() {
    keptRef.current?.scrollIntoView({ block: "center" });
    keptRef.current?.focus({ preventScroll: true });
  }

  return (
    <div className="extract">
      <p className="extract-headline" ref={headlineRef} tabIndex={-1}>
        Kept <b>{shareOf(extract.keptLength, total)}</b> of the page: {count(extract.keptLength)} of {count(total)} characters. Thrown away: {count(total - extract.keptLength)}.
      </p>
      <div className="extract-bar" aria-hidden="true">
        <span className="extract-bar-part extract-colour--kept" style={{ flexGrow: extract.keptLength }} />
        <span className="extract-bar-part extract-colour--text" style={{ flexGrow: extract.otherTextLength }} />
        <span className="extract-bar-part extract-colour--code" style={{ flexGrow: extract.codeLength }} />
        <span className="extract-bar-part extract-colour--markup" style={{ flexGrow: extract.markupLength }} />
      </div>
      <ul className="extract-shares" aria-label="Where the page's characters went">
        <Share kind="kept" label="Kept: the main text" length={extract.keptLength} total={total} />
        <Share kind="text" label="Thrown away: menus, links, side boxes, footer" length={extract.otherTextLength} total={total} />
        <Share kind="code" label={`Thrown away: code, ${plural(extract.scriptCount, "script")} and ${plural(extract.styleCount, "style block")}`} length={extract.codeLength} total={total} />
        <Share kind="markup" label="Thrown away: HTML tags, settings and spacing" length={extract.markupLength} total={total} />
      </ul>
      {extract.trace ? <HowItWorks trace={extract.trace} onShowKept={showKept} /> : <p className="card-lesson">No step-by-step view for this page: Readability did not pick one box as the article here, so there are no real steps to show.</p>}
      <p className="section-title extract-label extract-colour--text">
        <span className="extract-swatch" />
        {`Thrown away: ${plural(extract.droppedText.length, "piece")} of page text`}
      </p>
      <ul className="extract-dropped" tabIndex={0} aria-label="Page text thrown away">
        {extract.droppedText.map((piece, i) => (
          <li key={i}>{piece}</li>
        ))}
      </ul>
      <p className="section-title extract-label extract-colour--kept">
        <span className="extract-swatch" />
        Kept: the document that goes into the pile
      </p>
      {extract.title && (
        <p className="note">
          Title found: <b>{extract.title}</b> (kept apart from the text)
        </p>
      )}
      <pre ref={keptRef} className="extract-text" tabIndex={0} role="region" aria-label="The plain text that is kept">
        {extract.text || "Nothing. Readability found no main text on this page, so nothing would go into the pile."}
      </pre>
    </div>
  );
}

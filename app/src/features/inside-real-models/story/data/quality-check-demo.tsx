import "./data.css";
import { useId, useState } from "react";
import { useSavedPageExtract } from "@/features/inside-real-models/hooks/use-page-extract";
import type { DemoProps } from "@/features/inside-real-models/interfaces/model-part.interface";
import { SOURCES } from "@/features/inside-real-models/trees/sources";
import { FactRows } from "@/features/inside-real-models/fact-rows";
import { HonestNote } from "@/features/inside-real-models/story/watch-parts";
import { GPT_2_KARMA, checkQuality, keepsDocument, passesKarma } from "./quality-rules";
import { SAVED_PAGE } from "./saved-page";

function DocumentCheck({ title, text }: { title: string; text: string }) {
  const rules = checkQuality(text);
  const keeps = keepsDocument(rules);
  return (
    <div className="watch-box">
      <p className="section-title">{title}</p>
      <pre className="extract-text extract-text--short" tabIndex={0}>
        {text || "Nothing yet."}
      </pre>
      <FactRows rows={rules.map((rule): [string, string] => [`${rule.passes ? "✓" : "✗"} ${rule.name}`, rule.measured])} />
      <span className={`verdict ${keeps ? "is-keep" : "is-drop"}`}>{keeps ? "Keep: it passes every rule" : `Drop: fails ${rules.filter((rule) => !rule.passes).length} rule(s)`}</span>
    </div>
  );
}

export function QualityCheckDemo({ modelId }: DemoProps) {
  const page = useSavedPageExtract();
  const [karma, setKarma] = useState(GPT_2_KARMA);
  const [own, setOwn] = useState("");
  const karmaId = useId();
  const ownId = useId();

  return (
    <div className="watch">
      {modelId === "gpt-2-small" ? (
        <div className="watch-box">
          <p className="section-title">GPT-2's real rule, from its paper</p>
          <p className="note">
            Keep a web page only if the Reddit post that linked to it got at least {GPT_2_KARMA} karma. The rule looks at the link, not at the words, so type a karma number. This app can't reach Reddit.
          </p>
          <div className="watch-slider">
            <label htmlFor={karmaId}>Karma of the post</label>
            <input id={karmaId} type="number" className="watch-number" value={karma} min={-100} max={100000} onChange={(event) => setKarma(Number(event.target.value))} />
            <span className={`verdict ${passesKarma(karma) ? "is-keep" : "is-drop"}`}>{passesKarma(karma) ? "Keep the page" : "Drop the page"}</span>
          </div>
          <a className="card-lesson" href={SOURCES["gpt-2-paper"].url} target="_blank" rel="noreferrer">
            {SOURCES["gpt-2-paper"].title}, section 2.1
          </a>
        </div>
      ) : (
        <HonestNote>Meta says Llama 3's data went through heuristic filters, adult-content filters and text-quality classifiers trained on data Llama 2 wrote, but it has not published the rules. So below are this demo's own rules instead.</HonestNote>
      )}
      <p className="note">Now check the text itself. These four rules were written for this demo, not by OpenAI or Meta. Each one is measured live on the text below.</p>
      <div className="watch-grid">
        <DocumentCheck title={`The saved page's main text (${SAVED_PAGE.site})`} text={page.text} />
        <DocumentCheck title="The same page's menus, links and footer" text={page.droppedText.join("\n")} />
      </div>
      <div className="watch-box">
        <label className="section-title" htmlFor={ownId}>
          Try your own text
        </label>
        <textarea id={ownId} className="watch-input" rows={4} value={own} placeholder="Paste or type any text…" onChange={(event) => setOwn(event.target.value)} />
      </div>
      {own.trim() && <DocumentCheck title="Your text" text={own} />}
    </div>
  );
}

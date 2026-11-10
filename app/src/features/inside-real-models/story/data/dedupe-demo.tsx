import "./data.css";
import { useId, useMemo, useState } from "react";
import { useSavedPageExtract } from "@/features/inside-real-models/hooks/use-page-extract";
import type { DemoProps } from "@/features/inside-real-models/interfaces/model-part.interface";
import { FactRows } from "@/features/inside-real-models/fact-rows";
import { BeforeAfter, HonestNote } from "@/features/inside-real-models/story/watch-parts";
import { SOURCES } from "@/features/inside-real-models/trees/sources";
import { count, percent } from "@/shared/utils/format";
import { NEAR_COPY_SHARE, SHINGLE_SIZE, checkCopy, normalWords, shinglesOf } from "./dedupe";

const VERDICT_TEXT = { copy: "Drop: an exact copy", "near-copy": "Drop: a near copy", new: "Keep: it is new" };

export function DedupeDemo({ modelId }: DemoProps) {
  const page = useSavedPageExtract();
  const pile = useMemo(() => page.text.split("\n").filter((line) => normalWords(line).length >= SHINGLE_SIZE), [page.text]);
  const sample = pile.reduce((longest, line) => (line.length > longest.length ? line : longest), "");
  const [candidate, setCandidate] = useState(sample);
  const inputId = useId();
  const check = checkCopy(candidate, pile);
  const matched = check.match === null ? "" : pile[check.match];
  const matchedShingles = shinglesOf(matched);
  const shingles = [...shinglesOf(candidate)];

  return (
    <div className="watch">
      <p className="note">
        The pile already holds {count(pile.length)} documents: each paragraph of the saved page counts as one. The new document starts as a copy of its longest paragraph. Edit the new document below and watch the check decide whether it is a copy.
      </p>
      <div className="watch-box">
        <label className="section-title" htmlFor={inputId}>
          A new document arrives
        </label>
        <textarea id={inputId} className="watch-input" rows={5} value={candidate} onChange={(event) => setCandidate(event.target.value)} />
        <div className="watch-controls">
          <button type="button" className="action" onClick={() => setCandidate(sample)}>
            Put back the page's paragraph
          </button>
          <span className="card-lesson">Try changing one word, then a whole sentence.</span>
        </div>
      </div>
      <BeforeAfter
        beforeLabel="Checked against the pile"
        before={
          <FactRows
            rows={[
              ["Its fingerprint", check.fingerprint],
              ["Closest document's fingerprint", check.match === null ? "none" : checkCopy(matched, []).fingerprint],
              [`Its ${SHINGLE_SIZE}-word runs`, count(check.shingleCount)],
              ["Runs also in the closest document", check.share === 0 ? "0%" : percent(check.share)],
            ]}
          />
        }
        afterLabel="Decision"
        after={
          <>
            <span className={`verdict ${check.verdict === "new" ? "is-keep" : "is-drop"}`}>{VERDICT_TEXT[check.verdict]}</span>
            <p className="card-lesson">{`Pile: ${count(pile.length)} → ${count(pile.length + (check.verdict === "new" ? 1 : 0))} documents`}</p>
          </>
        }
      />
      <div className="watch-box">
        <p className="section-title">{`Its ${SHINGLE_SIZE}-word runs: found in the closest document, or not`}</p>
        <ul className="piece-list">
          {shingles.slice(0, 40).map((shingle) => (
            <li key={shingle} className={`piece-chip${matchedShingles.has(shingle) ? " is-picked" : ""}`}>
              {shingle}
            </li>
          ))}
        </ul>
      </div>
      <HonestNote>
        This check is this demo's own: an exact fingerprint (FNV-1a of the lower-cased words) and a near-copy rule of {percent(NEAR_COPY_SHARE)} shared {SHINGLE_SIZE}-word runs. The {SHINGLE_SIZE}-word runs and the lower-casing come from the{" "}
        <a href={SOURCES["gpt-2-paper"].url} target="_blank" rel="noreferrer">
          GPT-2 paper
        </a>
        , which used them to measure how much of each test set was already in WebText. {modelId === "gpt-2-small" ? "How OpenAI removed copies from WebText itself is not published." : "Meta says it used \"semantic deduplication\" for Llama 3 and has not published how."}
      </HonestNote>
    </div>
  );
}

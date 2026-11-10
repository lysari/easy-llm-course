import "./data.css";
import { useState } from "react";
import { useSavedPageExtract } from "@/features/inside-real-models/hooks/use-page-extract";
import { FactRows } from "@/features/inside-real-models/fact-rows";
import { HonestNote } from "@/features/inside-real-models/story/watch-parts";
import { count } from "@/shared/utils/format";
import { checkCopy } from "./dedupe";
import { checkQuality, keepsDocument } from "./quality-rules";
import { SAVED_PAGE } from "./saved-page";

interface PipelineRun {
  quality: boolean;
  copy: ReturnType<typeof checkCopy>;
}

export function DataPipelineDemo() {
  const page = useSavedPageExtract();
  const [pile, setPile] = useState<string[]>([]);
  const [runs, setRuns] = useState<PipelineRun[]>([]);

  function runOnce() {
    const quality = keepsDocument(checkQuality(page.text));
    const copy = checkCopy(page.text, pile);
    setRuns([...runs, { quality, copy }]);
    if (quality && copy.verdict === "new") setPile([...pile, page.text]);
  }

  function startAgain() {
    setPile([]);
    setRuns([]);
  }

  return (
    <div className="watch">
      <p className="note">
        Send the saved page ({SAVED_PAGE.title}, {SAVED_PAGE.site}) through all three steps. Then send it again, as if a second link pointed to the same page, and watch the last step catch it.
      </p>
      <div className="watch-controls">
        <button type="button" className="action action--strong" onClick={runOnce}>
          {runs.length === 0 ? "Send the page through" : "Send the same page again"}
        </button>
        <button type="button" className="action" onClick={startAgain} disabled={runs.length === 0}>
          Empty the pile
        </button>
        <span className="card-lesson">{`Pile: ${count(pile.length)} document${pile.length === 1 ? "" : "s"}, ${count(pile.reduce((sum, text) => sum + text.length, 0))} characters`}</span>
      </div>
      <div className="watch-grid">
        {runs.map((run, i) => (
          <div className="watch-box" key={i}>
            <p className="section-title">{`Try ${i + 1}`}</p>
            <FactRows
              rows={[
                ["1. Copy the text out", `${count(SAVED_PAGE.html.length)} → ${count(page.keptLength)} characters`],
                ["2. Quality check", run.quality ? "passes every rule" : "fails a rule"],
                ["3. Copy check", run.copy.verdict === "new" ? "new" : `copy of document ${(run.copy.match ?? 0) + 1}`],
              ]}
            />
            <span className={`verdict ${run.quality && run.copy.verdict === "new" ? "is-keep" : "is-drop"}`}>{run.quality && run.copy.verdict === "new" ? "Added to the pile" : "Dropped"}</span>
          </div>
        ))}
      </div>
      <HonestNote>Step 1 runs Mozilla Readability, steps 2 and 3 run this demo's own rules; open each step's card to see them one at a time and how GPT-2's and Llama 3's makers did it.</HonestNote>
    </div>
  );
}

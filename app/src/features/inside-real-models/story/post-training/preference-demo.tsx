import { useId, useState } from "react";
import { END_TOKEN, START_CHOICES, VOCABULARY } from "@/core/model/corpus";
import { copyWeights } from "@/core/model/tiny-gpt";
import { FactRows } from "@/features/inside-real-models/fact-rows";
import type { DemoProps } from "@/features/inside-real-models/interfaces/model-part.interface";
import { BeforeAfter, HonestNote } from "@/features/inside-real-models/story/watch-parts";
import { percent, showWord } from "@/shared/utils/format";
import { answerChance, preferenceGap, pushPreference } from "./post-training-math";

const ANSWERS = VOCABULARY.filter((word) => word !== END_TOKEN && /^\p{L}+$/u.test(word));

const GAP_LABEL = "Gap (bigger = the model prefers your better answer more)";

function WordSelect({ label, value, onChange }: { label: string; value: string; onChange: (word: string) => void }) {
  const id = useId();
  return (
    <div className="watch-slider">
      <label htmlFor={id}>{label}</label>
      <select id={id} className="watch-select" value={value} onChange={(event) => onChange(event.target.value)}>
        {ANSWERS.map((word) => (
          <option key={word} value={word}>
            {word}
          </option>
        ))}
      </select>
    </div>
  );
}

export function PreferenceDemo({ part, modelId, tiny }: DemoProps) {
  const [prompt, setPrompt] = useState(0);
  const [better, setBetter] = useState("bone");
  const [worse, setWorse] = useState("stick");
  const [copy, setCopy] = useState(() => copyWeights(tiny.weights));
  const [nudges, setNudges] = useState(0);
  const promptId = useId();
  const words = START_CHOICES[prompt].split(" ");
  const gapBefore = preferenceGap(words, better, worse, tiny.weights);
  const gapNow = preferenceGap(words, better, worse, copy);

  function nudge() {
    pushPreference(copy, words, better, worse);
    setNudges(nudges + 1);
  }

  function startAgain() {
    setCopy(copyWeights(tiny.weights));
    setNudges(0);
  }

  return (
    <div className="watch">
      <p className="note">You play the person who compares two answers. Pick a question, the answer you like better and the one you like less.</p>
      <div className="watch-controls">
        <div className="watch-slider">
          <label htmlFor={promptId}>Question</label>
          <select id={promptId} className="watch-select" value={prompt} onChange={(event) => setPrompt(Number(event.target.value))}>
            {START_CHOICES.map((choice, i) => (
              <option key={choice} value={i}>
                {choice} …
              </option>
            ))}
          </select>
        </div>
        <WordSelect label="Better answer" value={better} onChange={setBetter} />
        <WordSelect label="Worse answer" value={worse} onChange={setWorse} />
      </div>
      {part.id === "preference" && (
        <div className="watch-controls">
          <button type="button" className="action action--strong" onClick={nudge}>
            Nudge toward the better answer
          </button>
          <button type="button" className="action" onClick={startAgain} disabled={nudges === 0}>
            Start again
          </button>
          <span className="card-lesson">{`${nudges} nudge${nudges === 1 ? "" : "s"}`}</span>
        </div>
      )}
      <BeforeAfter
        beforeLabel="Tiny GPT now"
        before={<FactRows rows={[[`Chance of "${showWord(better)}"`, percent(answerChance(words, better, tiny.weights))], [`Chance of "${showWord(worse)}"`, percent(answerChance(words, worse, tiny.weights))], [GAP_LABEL, gapBefore.toFixed(3)]]} />}
        afterLabel={part.id === "preference" ? `After ${nudges} nudge${nudges === 1 ? "" : "s"}` : "The one number preference tuning grows"}
        after={part.id === "preference" ? <FactRows rows={[[`Chance of "${showWord(better)}"`, percent(answerChance(words, better, copy))], [`Chance of "${showWord(worse)}"`, percent(answerChance(words, worse, copy))], [GAP_LABEL, gapNow.toFixed(3)]]} /> : <p className="term-big">{`ln ${answerChance(words, better, tiny.weights).toFixed(4)} − ln ${answerChance(words, worse, tiny.weights).toFixed(4)} = ${gapBefore.toFixed(3)}`}</p>}
      />
      <p className="card-lesson">Gap = ln p(better) − ln p(worse)</p>
      <HonestNote>{`${part.id === "preference" ? "Live on a copy of Tiny GPT. Each nudge moves every number a small step along the slope that grows the gap, a bare version of preference tuning with no reward model and no reference model." : "Both chances are live from Tiny GPT's current numbers; which answer is better is your choice."} ${modelId === "gpt-2-small" ? "GPT-2 had no preference tuning." : "Meta used PPO and DPO for Llama 3 Instruct; its preference rankings are not published."}`}</HonestNote>
    </div>
  );
}

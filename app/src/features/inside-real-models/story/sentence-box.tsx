import { useId } from "react";
import { joinTokens } from "@/core/model/corpus";
import type { Sentence } from "@/features/inside-real-models/interfaces/sentence.interface";
import { START_TEXTS } from "@/features/inside-real-models/sentence";

const MAX_LENGTH = 120;
const NO_EXAMPLE = "";

interface SentenceBoxProps {
  text: string;
  sentence: Sentence;
  onChange: (text: string) => void;
}

function noteOf({ words, unknown, isSample, isCut }: Sentence) {
  const notes = [];
  if (unknown.length > 0) notes.push(`Our tiny model doesn't know ${unknown.map((word) => `"${word}"`).join(", ")}, so it skips ${unknown.length === 1 ? "it" : "them"}.`);
  if (isSample) notes.push(`With no words it knows, this demo uses "${joinTokens(words)}".`);
  if (isCut) notes.push(`Only the last ${words.length} words fit, so the start is cut off.`);
  return notes.join(" ");
}

export function SentenceBox({ text, sentence, onChange }: SentenceBoxProps) {
  const id = useId();
  return (
    <div className="sentence-box">
      <label className="section-title" htmlFor={`${id}-text`}>
        Your sentence
      </label>
      <div className="sentence-field">
        <input id={`${id}-text`} type="text" name="sentence" value={text} maxLength={MAX_LENGTH} autoComplete="off" spellCheck={false} aria-describedby={`${id}-use ${id}-note`} onChange={(event) => onChange(event.target.value)} />
        <select name="example" aria-label="Example sentences" value={NO_EXAMPLE} onChange={(event) => onChange(event.target.value)}>
          <option value={NO_EXAMPLE} disabled>
            Examples
          </option>
          {START_TEXTS.map((choice) => (
            <option key={choice} value={choice}>
              {choice}
            </option>
          ))}
        </select>
      </div>
      <p className="card-lesson" id={`${id}-use`}>
        This demo runs our tiny model on these words. Change them and watch the result change.
      </p>
      <p className="card-lesson sentence-note" id={`${id}-note`} aria-live="polite">
        {noteOf(sentence)}
      </p>
    </div>
  );
}

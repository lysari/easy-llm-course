import { joinTokens } from "@/core/model/corpus";
import type { Sentence } from "@/features/inside-real-models/interfaces/sentence.interface";
import { START_TEXTS } from "./sentence";

const MAX_LENGTH = 120;

interface SentenceBarProps {
  text: string;
  sentence: Sentence;
  onChange: (text: string) => void;
}

function noteOf({ words, unknown, isSample, isCut }: Sentence) {
  const notes = [];
  if (unknown.length > 0) notes.push(`The tiny model doesn't know ${unknown.map((word) => `"${word}"`).join(", ")}, so it skips ${unknown.length === 1 ? "it" : "them"}.`);
  if (isSample) notes.push(`With no words it knows, the live example uses "${joinTokens(words)}".`);
  if (isCut) notes.push(`Only the last ${words.length} words fit, so the start is cut off.`);
  return notes.join(" ");
}

export function SentenceBar({ text, sentence, onChange }: SentenceBarProps) {
  const note = noteOf(sentence);
  return (
    <section className="sentence-bar" aria-label="Live example">
      <label className="sentence-field">
        Your sentence
        <input type="text" name="sentence" value={text} maxLength={MAX_LENGTH} autoComplete="off" spellCheck={false} aria-describedby="inside-real-models-sentence-note" onChange={(event) => onChange(event.target.value)} />
      </label>
      <div className="start-choices" role="group" aria-label="Quick picks">
        {START_TEXTS.map((choice) => (
          <button key={choice} type="button" className="start-choice" aria-pressed={text === choice} onClick={() => onChange(choice)}>
            {choice}
          </button>
        ))}
      </div>
      <p className="note sentence-note" id="inside-real-models-sentence-note" aria-live="polite">
        {note || `The tiny model reads "${joinTokens(sentence.words)}" and guesses the next word.`}
      </p>
    </section>
  );
}

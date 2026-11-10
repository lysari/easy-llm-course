import { useId, useState } from "react";
import { TRAINING_LINES, joinTokens } from "@/core/model/corpus";
import { FactRows } from "@/features/take-apart/fact-rows";
import type { DemoProps } from "@/features/take-apart/interfaces/model-part.interface";
import { BeforeAfter, HonestNote, WordPicker } from "@/features/take-apart/story/watch-parts";
import { wordChances } from "@/features/take-apart/story/training/training-math";
import { showWord } from "@/shared/utils/format";

const average = (values: number[]) => (values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length);

export function AnswerLossDemo({ part, modelId, tiny }: DemoProps) {
  const [line, setLine] = useState(0);
  const [start, setStart] = useState(TRAINING_LINES[0].text.split(" ").length - 2);
  const lineId = useId();
  const words = TRAINING_LINES[line].text.split(" ");
  const answerStart = Math.min(Math.max(start, 1), words.length - 1);
  const terms = wordChances(words, tiny.weights);
  const answerTerms = terms.slice(answerStart - 1);
  const first = answerTerms[0];

  function pickLine(index: number) {
    setLine(index);
    setStart(TRAINING_LINES[index].text.split(" ").length - 2);
  }

  return (
    <div className="watch">
      <p className="note">Split a sentence into a question and an answer. Supervised fine-tuning scores the model only on the answer words; the question's words are not counted.</p>
      <div className="watch-slider">
        <label htmlFor={lineId}>Sentence</label>
        <select id={lineId} className="watch-select" value={line} onChange={(event) => pickLine(Number(event.target.value))}>
          {TRAINING_LINES.map((each, i) => (
            <option key={i} value={i}>
              {joinTokens(each.text.split(" "))}
            </option>
          ))}
        </select>
      </div>
      <WordPicker label="The answer starts at:" words={words.slice(1)} picked={answerStart - 1} onPick={(i) => setStart(i + 1)} />
      <BeforeAfter
        beforeLabel="Every word scored, as in pre-training"
        before={<FactRows rows={terms.map((term, i): [string, string] => [`${showWord(term.word)}${i + 1 >= answerStart ? " (answer)" : ""}`, term.loss.toFixed(2)])} />}
        afterLabel="Only the answer words counted"
        after={
          <FactRows
            rows={[
              ["Loss over every word", average(terms.map((term) => term.loss)).toFixed(3)],
              ["Loss over the answer only", average(answerTerms.map((term) => term.loss)).toFixed(3)],
              ...(part.id === "sft-term" && first ? [[`One answer word: "${showWord(first.word)}"`, `−ln(${first.chance.toFixed(3)}) = ${first.loss.toFixed(3)}`] as [string, string]] : []),
            ]}
          />
        }
      />
      <HonestNote>{`Each loss is live from our tiny model's current numbers. The split into question and answer is yours, made by hand here; our tiny model was never fine-tuned. ${modelId === "gpt-2-small" ? "GPT-2 had no fine-tuning at all." : "Meta's own SFT examples for Llama 3 are not published."}`}</HonestNote>
    </div>
  );
}

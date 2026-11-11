import { useId, useState } from "react";
import { GUESSING_LOSS, TEST_ANSWER, TEST_WORDS, TRAINING_LINES, VOCABULARY, joinTokens } from "@/core/model/corpus";
import { TARGET_ROUNDS } from "@/core/model/lab";
import { nextOptions } from "@/core/model/generate";
import { countWeights } from "@/core/model/tiny-gpt";
import { FactRows } from "@/features/inside-real-models/fact-rows";
import { useDemoLab } from "@/features/inside-real-models/hooks/use-demo-lab";
import type { DemoProps } from "@/features/inside-real-models/interfaces/model-part.interface";
import { BeforeAfter, HonestNote, WordPicker } from "@/features/inside-real-models/story/watch-parts";
import { ChanceBars } from "@/shared/components/chance-bars";
import { LossPlot } from "@/shared/components/loss-plot";
import { count, percent, showWord } from "@/shared/utils/format";
import { slopeCheck, wordChances } from "./training-math";

const fixed = (value: number, digits = 4) => value.toFixed(digits);

export function TrainingDemo({ part, modelId }: DemoProps) {
  const { lab, adam, change, train, reset } = useDemoLab();
  const [line, setLine] = useState(0);
  const [place, setPlace] = useState(TRAINING_LINES[0].text.split(" ").length - 2);
  const lineId = useId();
  const words = TRAINING_LINES[line].text.split(" ");
  const chances = wordChances(words, lab.live);
  const at = Math.min(Math.max(place, 1), words.length - 1);
  const guess = chances[at - 1];
  const history = lab.history;
  const lastLoss = history[history.length - 1]?.loss;
  const priorLoss = history[history.length - 2]?.loss;

  function pickLine(index: number) {
    setLine(index);
    setPlace(TRAINING_LINES[index].text.split(" ").length - 2);
  }

  return (
    <div className="watch">
      <p className="note">{`A fresh copy of Tiny GPT, our small practice model, separate from the one in the other demos. You train it here, one round at a time. Each round runs all ${count(TRAINING_LINES.length)} hand-written sentences, ${count(TRAINING_LINES.reduce((sum, each) => sum + each.times, 0))} with repeats.`}</p>
      <div className="watch-controls">
        <button type="button" className="action action--strong" onClick={() => train(1)} disabled={lab.isTrained}>
          Run one round
        </button>
        <button type="button" className="action" onClick={() => train(20)} disabled={lab.isTrained}>
          Run 20 rounds
        </button>
        <button type="button" className="action" onClick={reset} disabled={lab.round === 0}>
          Start again
        </button>
        <span className="card-lesson">{`Round ${lab.round} of ${TARGET_ROUNDS}`}</span>
      </div>
      {part.id === "pre-training" && (
        <div className="watch-grid">
          <LossPlot points={history} rounds={TARGET_ROUNDS} guessingLoss={GUESSING_LOSS} />
          <div className="watch-box">
            <p className="section-title">{`Next word after "${joinTokens(TEST_WORDS)}"`}</p>
            <ChanceBars options={nextOptions(TEST_WORDS, lab.live)} answer={TEST_ANSWER} />
          </div>
        </div>
      )}
      {part.id === "training-round" && (
        <BeforeAfter
          beforeLabel={lab.round === 0 ? "No round yet" : `Before round ${lab.round}`}
          before={<FactRows rows={[["Loss", priorLoss === undefined ? (lastLoss === undefined ? "–" : "first round") : fixed(priorLoss)], [`First number of "${VOCABULARY[0]}"`, change ? fixed(change.before) : fixed(lab.live.tokenTable.data[0])]]} />}
          afterLabel={lab.round === 0 ? "Run a round" : `After round ${lab.round}`}
          after={<FactRows rows={[["Loss", lastLoss === undefined ? "–" : fixed(lastLoss)], [`First number of "${VOCABULARY[0]}"`, fixed(lab.live.tokenTable.data[0])], ["Numbers nudged", lab.round > 0 ? `all ${count(countWeights(lab.live))}` : "none yet"]]} />}
        />
      )}
      {(part.id === "guess" || part.id === "loss-term") && (
        <>
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
          <WordPicker label="Guess the word at:" words={words.slice(1)} picked={at - 1} onPick={(i) => setPlace(i + 1)} />
          <BeforeAfter
            beforeLabel="The words before"
            before={<p className="watch-text">{joinTokens(words.slice(0, at))} …</p>}
            afterLabel={part.id === "guess" ? `Chance it gives "${showWord(guess.word)}"` : "How wrong: −ln(chance)"}
            after={<p className="term-big">{part.id === "guess" ? percent(guess.chance) : `−ln(${fixed(guess.chance, 3)}) = ${fixed(guess.loss, 3)}`}</p>}
          />
          {part.id === "loss-term" && <FactRows rows={chances.map((each, i): [string, string] => [`${showWord(each.word)}${i + 1 === at ? " ←" : ""}`, `−ln(${fixed(each.chance, 3)}) = ${fixed(each.loss, 2)}`])} />}
        </>
      )}
      {part.id === "slope" && <SlopeView lab={lab} />}
      {part.id === "adam-update" && (
        <BeforeAfter
          beforeLabel={`First number of "${VOCABULARY[0]}", before the last round`}
          before={<FactRows rows={[["Value", change ? fixed(change.before) : "run a round"], ["Its slope", change ? fixed(change.slope, 5) : "–"], ["Running average of the slope (m)", fixed(adam.momentum, 5)], ["Running average of slope² (v)", fixed(adam.variance, 8)]]} />}
          afterLabel="After Adam's update"
          after={<FactRows rows={[["Value", change ? fixed(change.after) : "–"], ["Step worked out from m and v", change ? fixed(-adam.step, 5) : "–"], ["Step the model really took", change ? fixed(change.after - change.before, 5) : "–"]]} />}
        />
      )}
      <HonestNote>{`This is Tiny GPT training live: ${count(TRAINING_LINES.length)} hand-written sentences, Adam, ${TARGET_ROUNDS} rounds. ${modelId === "gpt-2-small" ? "GPT-2's optimizer, rounds and loss values are not published." : "Llama 3's optimizer, batch size and loss values are not published for the 8B model."} The real model's published numbers are in this part's facts.`}</HonestNote>
    </div>
  );
}

function SlopeView({ lab }: { lab: ReturnType<typeof useDemoLab>["lab"] }) {
  const check = slopeCheck(lab.live);
  return (
    <BeforeAfter
      beforeLabel={`First number of "${VOCABULARY[0]}": ${fixed(check.weight)}`}
      before={<FactRows rows={[["Loss if it moves up a hair", fixed(check.lossUp, 8)], ["Loss if it moves down a hair", fixed(check.lossDown, 8)], ["Slope measured by nudging", fixed(check.measured, 6)]]} />}
      afterLabel="Slope from the chain rule (backpropagation)"
      after={<p className="term-big">{fixed(check.slope, 6)}</p>}
    />
  );
}

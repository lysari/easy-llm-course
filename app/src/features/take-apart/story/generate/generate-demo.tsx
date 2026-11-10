import "./generate.css";
import { Fragment, useState } from "react";
import { END_ID, END_TOKEN, MAX_CONTEXT, VOCABULARY, joinTokens } from "@/core/model/corpus";
import { FactRows } from "@/features/take-apart/fact-rows";
import type { DemoProps } from "@/features/take-apart/interfaces/model-part.interface";
import { BeforeAfter, HonestNote } from "@/features/take-apart/story/watch-parts";
import { ChanceBars } from "@/shared/components/chance-bars";
import type { Sampling } from "@/shared/interfaces/model.interface";
import { percent, showWord } from "@/shared/utils/format";
import { type WriteRound, isGreedy, rollWalk, writeRounds } from "./decode";
import { SamplingControls } from "./sampling-controls";

const PLAIN: Sampling = { temperature: 1, topK: 0, topP: 1 };

const NOTES = {
  "gpt-2-small":
    "The settings work as in GPT-2's released writing code: the scores are divided by the temperature, then the top k are kept, then the top p. One difference: GPT-2's top_p_logits keeps words only while their running total stays at or under p, so it keeps one word fewer than here. GPT-2's released code also never stops at its stop word, <|endoftext|> (ID 50,256): it writes a fixed number of words.",
  "llama-3-8b":
    "The settings work as in Llama 3's generation.py: temperature 0 takes the top word; otherwise the scores are divided by the temperature and top-p keeps the fewest top words whose chances pass p. Meta's code has no top-k; it is here for GPT-2. Llama 3 stops at <|end_of_text|> (ID 128,001) and, for the chat model, at <|eot_id|> (ID 128,009).",
};

function stopText(round: WriteRound, yourStop: string | null) {
  const word = `"${showWord(round.picked.token)}"`;
  if (round.stop === "stop-word") return `${word} is the stop word (ID ${END_ID}): stop`;
  if (round.stop === "your-stop") return `${word} is your stop word: stop, without adding it`;
  if (round.stop === "full") return `added; the text is full at ${MAX_CONTEXT} words: stop`;
  return `not ${yourStop ? `"${yourStop}" or ` : ""}the stop word, ${round.words.length + 1} of ${MAX_CONTEXT} words: go on`;
}

function IdChips({ words, newWord }: { words: string[]; newWord?: string }) {
  return (
    <ol className="piece-list">
      {words.map((word, i) => (
        <li key={i} className="piece-chip">
          {showWord(word)}
          <b>{VOCABULARY.indexOf(word)}</b>
        </li>
      ))}
      {newWord && (
        <li className="piece-chip is-picked">
          {showWord(newWord)}
          <b>{VOCABULARY.indexOf(newWord)}</b>
        </li>
      )}
    </ol>
  );
}

function RoundView({ partId, round, yourStop }: { partId: string; round: WriteRound; yourStop: string | null }) {
  const isAdded = round.stop === null || round.stop === "full";
  if (partId === "pick") {
    const walk = round.roll === null ? [] : rollWalk(round.options, round.roll);
    return (
      <BeforeAfter
        beforeLabel={round.roll === null ? "Temperature 0: no roll" : `The roll: ${round.roll.toFixed(3)}`}
        before={<ChanceBars options={round.options} answer={round.picked.token} limit={6} />}
        afterLabel={round.roll === null ? "The top word is taken" : "Take each chance away until the roll runs out"}
        after={round.roll === null ? <p className="term-big">{`"${showWord(round.picked.token)}" ${percent(round.picked.chance)}`}</p> : <FactRows rows={[...walk.map((each): [string, string] => [`− "${showWord(each.option.token)}" ${each.option.chance.toFixed(3)}`, `left ${each.left.toFixed(3)}`]), [`Picked "${showWord(round.picked.token)}"`, "left ≤ 0"]]} />}
      />
    );
  }
  if (partId === "append") {
    return (
      <BeforeAfter
        beforeLabel={`${round.words.length} word IDs read this round`}
        before={<IdChips words={round.words} />}
        afterLabel={isAdded ? `${round.words.length + 1} IDs: the pick added to the end` : "Nothing added: it was told to stop"}
        after={<IdChips words={round.words} newWord={isAdded ? round.picked.token : undefined} />}
      />
    );
  }
  return (
    <BeforeAfter
      beforeLabel="Chances for the next word, after your settings"
      before={<ChanceBars options={round.options} answer={round.picked.token} limit={6} />}
      afterLabel={round.roll === null ? "Temperature 0 takes the top word" : `Roll ${round.roll.toFixed(3)} picks`}
      after={
        <>
          <p className="term-big">{`"${showWord(round.picked.token)}" (ID ${VOCABULARY.indexOf(round.picked.token)}), ${percent(round.picked.chance)}`}</p>
          <span className={`verdict ${round.stop ? "is-drop" : "is-keep"}`}>{stopText(round, yourStop)}</span>
        </>
      }
    />
  );
}

export function GenerateDemo({ part, modelId, tiny }: DemoProps) {
  const [sampling, setSampling] = useState(PLAIN);
  const [seed, setSeed] = useState(1);
  const [yourStop, setYourStop] = useState<string | null>(null);
  const [shown, setShown] = useState(1);
  const rounds = writeRounds(tiny.words, tiny.weights, sampling, seed, yourStop);
  const visible = Math.min(shown, rounds.length);
  const round = rounds[visible - 1];
  const added = rounds.slice(0, visible).filter((each) => each.stop === null || each.stop === "full").map((each) => each.picked.token);

  return (
    <div className="watch">
      <p className="note">{`The page's own tiny model writes after your sentence, "${joinTokens(tiny.words)}", one word per round, live (training round ${tiny.lab.round}). Change a setting and the whole answer is written again.`}</p>
      <SamplingControls modelId={modelId} sampling={sampling} yourStop={yourStop} onSampling={setSampling} onYourStop={setYourStop} />
      <div className="watch-controls">
        <button type="button" className="action action--strong" onClick={() => setShown(visible + 1)} disabled={visible >= rounds.length}>
          Write the next word
        </button>
        <button type="button" className="action" onClick={() => setShown(rounds.length)} disabled={visible >= rounds.length}>
          Write to the end
        </button>
        <button type="button" className="action" onClick={() => setShown(1)} disabled={visible === 1}>
          Start again
        </button>
        <button type="button" className="action" onClick={() => setSeed(seed + 1)} disabled={isGreedy(sampling)}>
          New random rolls
        </button>
        <span className="card-lesson">{`Round ${visible} of ${rounds.length}${visible === rounds.length ? ", done" : ""}`}</span>
      </div>
      <div className="written">
        <span className="card-lesson">The text so far</span>
        <p className="watch-text">
          {joinTokens(tiny.words)}
          {added.map((word, i) => (
            <Fragment key={i}>
              {word === "," || word === "." ? "" : " "}
              <mark className="written-word">{word}</mark>
            </Fragment>
          ))}
        </p>
      </div>
      {round && <RoundView partId={part.id} round={round} yourStop={yourStop} />}
      {part.id === "stop-check" && (
        <div className="watch-box">
          <p className="section-title">The stop check, every round so far</p>
          <FactRows rows={rounds.slice(0, visible).map((each, i): [string, string] => [`Round ${i + 1}: "${showWord(each.picked.token)}"`, stopText(each, yourStop)])} />
          <p className="card-lesson">{`Our tiny model's stop word is "${END_TOKEN}", ID ${END_ID}, and its text holds ${MAX_CONTEXT} words at most. It learned to pick the stop word after "." because every training sentence ends that way.`}</p>
        </div>
      )}
      <HonestNote>{`Written live by our tiny model, with one seeded random roll per round, so the same settings give the same answer (roll set ${seed}). ${NOTES[modelId]}`}</HonestNote>
    </div>
  );
}

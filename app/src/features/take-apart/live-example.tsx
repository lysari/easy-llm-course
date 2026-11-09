import { GUESSING_LOSS, TEST_ANSWER, TEST_WORDS, TRAINING_LINES, VOCABULARY, joinTokens } from "@/core/model/corpus";
import { TARGET_ROUNDS } from "@/core/model/lab";
import type { TinyLink, TinyView } from "@/features/take-apart/interfaces/model-part.interface";
import { ChanceBars } from "@/shared/components/chance-bars";
import { LossPlot } from "@/shared/components/loss-plot";
import { percent, shortList, showWord } from "@/shared/utils/format";
import { FactRows } from "./fact-rows";
import { pickedWord } from "./tiny-parts";

function WordIds({ words }: { words: string[] }) {
  return (
    <ol className="word-ids" aria-label="Words and their IDs">
      {words.map((word, i) => (
        <li key={`${word}-${i}`}>
          {showWord(word)} <b>{VOCABULARY.indexOf(word)}</b>
        </li>
      ))}
    </ol>
  );
}

export function LiveExample({ link, tiny }: { link: TinyLink; tiny: TinyView }) {
  const { lab, test } = tiny;
  switch (link.part) {
    case "tokens":
    case "lookup":
      return <WordIds words={tiny.words} />;
    case "vocabulary":
      return <WordIds words={VOCABULARY} />;
    case "embedding":
      return <FactRows rows={tiny.words.map((word, i) => [`${showWord(word)} at place ${i + 1}`, shortList(tiny.trace.combined[i])])} />;
    case "model":
    case "chances":
      return <ChanceBars options={tiny.options} answer={pickedWord(tiny)} />;
    case "pick":
    case "write":
      return (
        <FactRows
          rows={[
            ...tiny.written.map(({ word, chance }, i): [string, string] => [`Round ${i + 1}: "${showWord(word)}"`, `had ${percent(chance)}`]),
            [tiny.isStopped ? `Round ${tiny.written.length + 1}: "stop"` : "Full", tiny.isStopped ? "done" : "no room left"],
          ]}
        />
      );
    case "corpus":
      return <FactRows rows={TRAINING_LINES.map(({ text, times }): [string, string] => [joinTokens(text.split(" ")), `× ${times}`])} />;
    case "training":
    case "loss":
    case "slope":
    case "adam":
      return <LossPlot points={lab.history} rounds={TARGET_ROUNDS} guessingLoss={GUESSING_LOSS} />;
    case "evaluation":
      return (
        <FactRows
          rows={[
            ["Test sentence", `${joinTokens(TEST_WORDS)} …`],
            [`Chance of "${TEST_ANSWER}"`, `${percent(test.before)} → ${percent(test.now)}`],
            ["Loss on it", `${(-Math.log(test.before)).toFixed(2)} → ${(-Math.log(test.now)).toFixed(2)}`],
          ]}
        />
      );
    default:
      return null;
  }
}

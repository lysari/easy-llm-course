import { useId, useMemo, useState } from "react";
import { useSavedPageExtract } from "@/features/take-apart/hooks/use-page-extract";
import type { DemoProps } from "@/features/take-apart/interfaces/model-part.interface";
import { SAVED_PAGE } from "@/features/take-apart/story/data/saved-page";
import { HonestNote } from "@/features/take-apart/story/watch-parts";
import { learnMerges, trainingWords } from "./bpe-train";
import { CountPairView, MergeRoundView, MergeRulesView } from "./bpe-views";
import { MAX_TOKENIZE_LENGTH } from "./gpt-2-bpe";

const MAX_OWN_TEXT = MAX_TOKENIZE_LENGTH * 25;

export function BpeLearnDemo({ part, modelId }: DemoProps) {
  const page = useSavedPageExtract();
  const [own, setOwn] = useState("");
  const [rounds, setRounds] = useState(0);
  const ownId = useId();
  const text = own.trim() ? own : page.text;
  const words = useMemo(() => trainingWords(text), [text]);
  const run = useMemo(() => learnMerges(words, rounds), [words, rounds]);

  function changeText(value: string) {
    setOwn(value);
    setRounds(0);
  }

  return (
    <div className="watch">
      <p className="note">{own.trim() ? "Learning merge rules live from your text." : `Learning merge rules live from the saved page's main text (${SAVED_PAGE.title}), with the recipe the GPT-2 paper describes: count every pair of side-by-side pieces, join the most common pair, repeat.`}</p>
      {part.id === "count-pair" && <CountPairView text={text} words={words} />}
      {part.id === "merge-round" && <MergeRoundView text={text} words={words} run={run} rounds={rounds} onRounds={setRounds} />}
      {part.id === "learn-merge" && <MergeRulesView run={run} rounds={rounds} onRounds={setRounds} />}
      <div className="watch-box">
        <label className="section-title" htmlFor={ownId}>
          Learn from your own text instead
        </label>
        <textarea id={ownId} className="watch-input" rows={3} maxLength={MAX_OWN_TEXT} value={own} placeholder="Paste a few paragraphs…" onChange={(event) => changeText(event.target.value)} />
      </div>
      <HonestNote>
        This runs the BPE recipe on one page, with GPT-2's split rule and its 256 starting bytes, so it learns far fewer and different rules than {modelId === "gpt-2-small" ? "GPT-2's 50,000" : "Llama 3's tokenizer, whose rules and training text Meta has not published"}. OpenAI's own training code for its tokenizer is not published either. Ties go to the pair seen first.
      </HonestNote>
    </div>
  );
}

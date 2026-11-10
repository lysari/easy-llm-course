import { useId, useState } from "react";
import { TEST_ANSWER, joinTokens } from "@/core/model/corpus";
import { nextOptions } from "@/core/model/generate";
import { FactRows } from "@/features/inside-real-models/fact-rows";
import type { DemoProps } from "@/features/inside-real-models/interfaces/model-part.interface";
import { MODELS } from "@/features/inside-real-models/models";
import { BeforeAfter, HonestNote } from "@/features/inside-real-models/story/watch-parts";
import { ChanceBars } from "@/shared/components/chance-bars";
import { count, percent, showWord } from "@/shared/utils/format";
import { TINY_QUESTIONS, markQuestion, scoreOf } from "./evaluation-math";

const TEST_QUESTION = Math.max(0, TINY_QUESTIONS.findIndex((question) => question.answer === TEST_ANSWER));

export function TinyTestDemo({ part, modelId, tiny }: DemoProps) {
  const [picked, setPicked] = useState(TEST_QUESTION);
  const pickId = useId();
  const { untrained } = tiny.lab;
  const marked = TINY_QUESTIONS.map((question) => markQuestion(question, tiny.weights));
  const right = marked.filter((question) => question.isRight).length;
  const question = marked[picked];
  const real = "value" in part.facts.headline ? `${part.facts.headline.value}${part.facts.headline.unit.startsWith("%") ? "" : " "}${part.facts.headline.unit}` : part.facts.headline.text;

  return (
    <div className="watch">
      {part.id === "test-question" && (
        <>
          <div className="watch-slider">
            <label htmlFor={pickId}>Question</label>
            <select id={pickId} className="watch-select" value={picked} onChange={(event) => setPicked(Number(event.target.value))}>
              {TINY_QUESTIONS.map((each, i) => (
                <option key={i} value={i}>{`${joinTokens(each.words)} … (${each.answer})`}</option>
              ))}
            </select>
          </div>
          <BeforeAfter
            beforeLabel={`"${joinTokens(question.words)} …"`}
            before={<ChanceBars options={nextOptions(question.words, tiny.weights)} answer={question.answer} />}
            afterLabel="Marked"
            after={<p className="term-big">{`Top answer "${showWord(question.top)}", key "${question.answer}": ${question.isRight ? "1, right" : "0, wrong"}`}</p>}
          />
        </>
      )}
      {part.id !== "test-question" && (
        <BeforeAfter
          beforeLabel="Before training"
          before={<p className="term-big">{`${count(Math.round(scoreOf(untrained) * marked.length))} of ${count(marked.length)} right`}</p>}
          afterLabel={`Now, after ${tiny.lab.round} rounds`}
          after={<p className="term-big">{`${percent(right / marked.length)} right: ${count(right)} of ${count(marked.length)}`}</p>}
        />
      )}
      {part.id === "benchmarks" && <FactRows rows={marked.map((each): [string, string] => [`${joinTokens(each.words)} … ${each.answer}`, `${each.isRight ? "✓" : "✗"} said "${showWord(each.top)}"`])} />}
      {part.id === "evaluation" && <FactRows rows={[[`Real ${MODELS.find((model) => model.id === modelId)?.title ?? ""}, as published`, real], ["Our tiny model, live", `${percent(right / marked.length)} right on ${count(marked.length)} questions`]]} />}
      <HonestNote>{`Our tiny model's ${count(TINY_QUESTIONS.length)} questions are the last word of each of its own hand-written sentences, so it has seen every answer in training: this test leaks. Real tests are kept apart from training text, and the GPT-2 paper measured how much of each test set was in WebText anyway (1–6%). When one question has two right endings in training, only one can be its top answer.`}</HonestNote>
    </div>
  );
}

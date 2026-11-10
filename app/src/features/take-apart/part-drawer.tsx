import type { Ref } from "react";
import { TEST_ANSWER, TEST_WORDS, joinTokens } from "@/core/model/corpus";
import type { ModelId, ModelPart, TinyPartId, TinyView } from "@/features/take-apart/interfaces/model-part.interface";
import type { Sentence } from "@/features/take-apart/interfaces/sentence.interface";
import { CodeBlock } from "@/shared/components/code-block";
import { Drawer } from "@/shared/components/drawer";
import { FormulaBlock } from "@/shared/components/formula-block";
import { count } from "@/shared/utils/format";
import { FactRows } from "./fact-rows";
import { InOut } from "./in-out";
import { LiveExample } from "./live-example";
import { DRAWER_ID, NO_LESSON, numbersOf, partsOf, tagOf, titleOf } from "./part-card";
import { NONE_TO_LEARN, PartNumbers } from "./part-numbers";
import { StoryDemo, StoryInfo } from "./story/story-sections";
import { readTinyPart } from "./tiny-parts";
import { SOURCES } from "./trees/sources";

const REVERSE_TITLES: Record<string, string> = {
  "RL-00": "The Illusion",
  "RL-01": "Text Is Numbers",
  "RL-02": "Meaning Is Position",
  "RL-03": "Attention Is Arithmetic",
  "RL-04": "Prediction, Not Knowing",
  "RL-05": "Weights, Not Wisdom",
  "RL-06": "Ground Zero",
};

const GROUND_ZERO_TEXT = { "multiply-add": "one multiply on two numbers, added to a sum", "stored-number": "one stored number", action: "one small step that uses no stored numbers" };

const TEST_PARTS: TinyPartId[] = ["loss", "evaluation"];
const WHOLE_RUN_PARTS: TinyPartId[] = ["corpus", "training", "slope", "adam"];

function liveCaption(part: TinyPartId, sentence: Sentence) {
  if (TEST_PARTS.includes(part)) return `Live on the test sentence "${joinTokens(TEST_WORDS)}", where the right next word is "${TEST_ANSWER}".`;
  if (WHOLE_RUN_PARTS.includes(part)) return "Live from the tiny model's own training.";
  return `Live on "${joinTokens(sentence.words)}", for its last word.`;
}

interface PartDrawerProps {
  ref: Ref<HTMLElement>;
  root: ModelPart;
  modelId: ModelId;
  modelTitle: string;
  part: ModelPart;
  tiny: TinyView;
  sentence: Sentence;
  nextPart: ModelPart | undefined;
  onOpen: (id: string) => void;
  onClose: () => void;
}

export function PartDrawer({ ref, root, modelId, modelTitle, part, tiny, sentence, nextPart, onOpen, onClose }: PartDrawerProps) {
  const readout = readTinyPart(part.tiny, tiny);
  const notes = part.facts.notes ?? [];
  const sources = [...new Set([...part.facts.sources, ...notes.map((note) => note.source)])];
  const realRows: [string, string][] = [
    ["Shape", part.facts.shape],
    ["Numbers", part.facts.weightCount > 0 ? `${numbersOf(part)}, worked out from the sources` : NONE_TO_LEARN],
    ["Inside", part.groundZero ? `Ground zero: ${GROUND_ZERO_TEXT[part.groundZero]}` : partsOf(part.children ?? [])],
    ...notes.map((note): [string, string] => [note.label, note.value]),
  ];
  const tinyRows: [string, string][] = [
    ["Part", readout.title],
    ["Shape", readout.shape],
    ["Numbers", readout.weightCount > 0 ? count(readout.weightCount) : NONE_TO_LEARN],
    ["Live value", readout.live],
    ["Training", tiny.lab.isTrained ? `done, round ${tiny.lab.round}` : `round ${tiny.lab.round}`],
  ];

  return (
    <Drawer ref={ref} id={DRAWER_ID} icon={part.icon} title={titleOf(part, root)} lesson={`${part.lesson} · ${part.reverseLesson}`} status={tagOf(part)} bodyKey={part.id} onClose={onClose} panel={part.story && <StoryDemo story={part.story} part={part} modelId={modelId} tiny={tiny} />}>
      <section className="section">
        <p className="section-title">What it does</p>
        <p className="part-summary">{part.summary}</p>
        <p className="note">{part.explanation}</p>
        <p className="why">{part.why}</p>
      </section>
      {part.story && <StoryInfo story={part.story} reasons={part.facts.reasons ?? []} nextPart={nextPart} onOpen={onOpen} />}
      <section className="section">
        <p className="section-title">What goes in and out</p>
        <InOut facts={part.facts} isDetailed />
        <PartNumbers modelTitle={modelTitle} part={part} tiny={tiny} isExact />
      </section>
      <section className="section">
        <p className="section-title">{part.tiny.missing ? "Nearest part in our tiny model" : "Live in our tiny model"}</p>
        {part.tiny.missing && <p className="note">{part.tiny.missing}</p>}
        {!part.tiny.missing && "reason" in readout.headline && <p className="note">{readout.headline.reason}</p>}
        {!part.tiny.missing && <LiveExample link={part.tiny} tiny={tiny} />}
        <FactRows rows={tinyRows} />
        <p className="card-lesson">{liveCaption(part.tiny.part, sentence)}</p>
      </section>
      <section className="section">
        <p className="section-title">{`The real part in ${modelTitle}`}</p>
        <FactRows rows={realRows} />
        <p className="card-lesson">Sources</p>
        <ul className="source-list card-lesson">
          {sources.map((id) => (
            <li key={id}>
              <a href={SOURCES[id].url} target="_blank" rel="noreferrer">
                {SOURCES[id].title}
              </a>
            </li>
          ))}
        </ul>
      </section>
      {part.formula && <FormulaBlock formula={part.formula} />}
      {part.code && <CodeBlock code={part.code} />}
      <section className="section">
        <p className="section-title">Read more</p>
        <FactRows
          rows={[
            [part.lesson, part.lesson === NO_LESSON ? "not in the course yet" : "build it, bottom-up"],
            [part.reverseLesson, REVERSE_TITLES[part.reverseLesson]],
          ]}
        />
      </section>
    </Drawer>
  );
}

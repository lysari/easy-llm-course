import type { Ref } from "react";
import type { ModelId, ModelPart, TinyView } from "@/features/take-apart/interfaces/model-part.interface";
import type { PartPlace } from "@/features/take-apart/interfaces/part-place.interface";
import { CodeBlock } from "@/shared/components/code-block";
import { Drawer } from "@/shared/components/drawer";
import { FormulaBlock } from "@/shared/components/formula-block";
import { FactRows } from "./fact-rows";
import { InOut } from "./in-out";
import { DRAWER_ID, NO_LESSON, numbersOf, partsOf, tagOf, titleOf } from "./part-card";
import { PartNumbers } from "./part-numbers";
import { RUNS_LABEL, SHARE_LABEL, runsOf, shareOf } from "./real-place";
import { StoryDemo, StoryInfo } from "./story/story-sections";
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

const NONE_TO_LEARN = "None to learn";

const WORKED_OUT = "worked out from the sources";

interface PartDrawerProps {
  ref: Ref<HTMLElement>;
  root: ModelPart;
  modelId: ModelId;
  modelTitle: string;
  part: ModelPart;
  place: PartPlace;
  tiny: TinyView;
  nextPart: ModelPart | undefined;
  onOpen: (id: string) => void;
  onClose: () => void;
}

export function PartDrawer({ ref, root, modelId, modelTitle, part, place, tiny, nextPart, onOpen, onClose }: PartDrawerProps) {
  const notes = part.facts.notes ?? [];
  const sources = [...new Set([...part.facts.sources, ...notes.map((note) => note.source)])];
  const share = shareOf(part, place, true);
  const runs = runsOf(part, place, true);
  const shareRows: [string, string][] = share ? [[SHARE_LABEL, `${share}, ${WORKED_OUT}`]] : [];
  const runsRows: [string, string][] = runs ? [[RUNS_LABEL, `${runs}, ${WORKED_OUT}`]] : [];
  const realRows: [string, string][] = [
    ["Where", place.path.map((each) => titleOf(each, root)).join(" › ")],
    ["Shape", part.facts.shape],
    ["Numbers", part.facts.weightCount > 0 ? `${numbersOf(part)}, ${WORKED_OUT}` : NONE_TO_LEARN],
    ...shareRows,
    ...runsRows,
    ["Inside", part.groundZero ? `Ground zero: ${GROUND_ZERO_TEXT[part.groundZero]}` : partsOf(part.children ?? [])],
    ...notes.map((note): [string, string] => [note.label, note.value]),
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
        <PartNumbers modelTitle={modelTitle} part={part} place={place} isExact />
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

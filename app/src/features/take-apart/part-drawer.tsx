import type { Ref } from "react";
import { TEST_ANSWER, TEST_WORDS, joinTokens } from "@/core/model/corpus";
import type { ModelPart, TinyPartId, TinyView } from "@/features/take-apart/interfaces/model-part.interface";
import type { Sentence } from "@/features/take-apart/interfaces/sentence.interface";
import { CodeBlock } from "@/shared/components/code-block";
import { Drawer } from "@/shared/components/drawer";
import { FormulaBlock } from "@/shared/components/formula-block";
import { count } from "@/shared/utils/format";
import { FactRows } from "./fact-rows";
import { LiveExample } from "./live-example";
import { DRAWER_ID, numbersOf, statusOf, titleOf } from "./part-card";
import { readTinyPart } from "./tiny-parts";

const REVERSE_TITLES: Record<string, string> = {
  "RL-00": "The Illusion",
  "RL-01": "Text Is Numbers",
  "RL-02": "Meaning Is Position",
  "RL-03": "Attention Is Arithmetic",
  "RL-04": "Prediction, Not Knowing",
  "RL-05": "Weights, Not Wisdom",
  "RL-06": "Ground Zero",
};

const NO_LESSON = "No lesson yet";

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
  part: ModelPart;
  tiny: TinyView;
  sentence: Sentence;
  onClose: () => void;
}

export function PartDrawer({ ref, root, part, tiny, sentence, onClose }: PartDrawerProps) {
  const readout = readTinyPart(part.tiny, tiny);
  const notes = part.facts.notes ?? [];
  const sources = [...new Set(notes.map((note) => note.source))];
  const realRows: [string, string][] = [
    ["Shape", part.facts.shape],
    ["Numbers", numbersOf(part)],
    ["In", part.facts.input],
    ["Out", part.facts.output],
    ["Inside", part.groundZero ? `Ground zero: ${GROUND_ZERO_TEXT[part.groundZero]}` : `${part.children?.length} ${part.children?.length === 1 ? "part" : "parts"}`],
    ...notes.map((note): [string, string] => [note.label, note.value]),
  ];
  const tinyRows: [string, string][] = [
    ["Part", readout.title],
    ["Shape", readout.shape],
    ["Numbers", count(readout.weightCount)],
    ["Live value", readout.live],
    ["Training", tiny.lab.isTrained ? `done, round ${tiny.lab.round}` : `round ${tiny.lab.round}`],
  ];

  return (
    <Drawer ref={ref} id={DRAWER_ID} icon={part.icon} title={titleOf(part, root)} lesson={`${part.lesson} · ${part.reverseLesson}`} status={statusOf(part, tiny)} bodyKey={part.id} onClose={onClose}>
      <p className="note">{part.explanation}</p>
      <section className="section">
        <p className="section-title">The real part</p>
        <FactRows rows={realRows} />
        {sources.length > 0 && <p className="card-lesson">{`Source: ${sources.join("; ")}`}</p>}
      </section>
      <section className="section">
        <p className="section-title">{part.tiny.missing ? "Nearest part in our tiny model" : "The same part in our tiny model"}</p>
        {part.tiny.missing && <p className="note">{part.tiny.missing}</p>}
        <FactRows rows={tinyRows} />
        {!part.tiny.missing && <LiveExample link={part.tiny} tiny={tiny} />}
        <p className="card-lesson">{liveCaption(part.tiny.part, sentence)}</p>
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
      <p className="why">{part.why}</p>
    </Drawer>
  );
}

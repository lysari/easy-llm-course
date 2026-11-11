import type { Ref } from "react";
import type { ModeId, ModelId, ModelPart, TinyView } from "@/features/inside-real-models/interfaces/model-part.interface";
import type { PartPlace } from "@/features/inside-real-models/interfaces/part-place.interface";
import type { RealFacts } from "@/features/inside-real-models/interfaces/real-facts.interface";
import type { Sentence } from "@/features/inside-real-models/interfaces/sentence.interface";
import { CodeBlock } from "@/shared/components/code-block";
import { Drawer } from "@/shared/components/drawer";
import { FormulaBlock } from "@/shared/components/formula-block";
import { count } from "@/shared/utils/format";
import { FactRows } from "./fact-rows";
import { DRAWER_ID, numbersOf, partsOf, tagOf, titleOf } from "./part-card";
import { RUNS_LABEL, amountOf, isTableSize, runsOf, shareLabelOf, shareOf } from "./real-place";
import { ReasonList, StoryDemo, StoryInfo } from "./story/story-sections";
import { SOURCES } from "./trees/sources";

const SMALLEST_STEP_TEXT = { "multiply-add": "one multiply on two numbers, added to a sum", "stored-number": "one stored number", action: "one simple action that uses no stored numbers" };

const NONE_TO_LEARN = "None to learn";

const WORKED_OUT = " *";

function headlineRows({ headline, weightCount }: RealFacts): [string, string][] {
  if ("text" in headline) return [["Main number", headline.text]];
  if (headline.unit === "numbers" && headline.value === weightCount) return [];
  const amount = amountOf(headline, true);
  return [["Main number", `${amount.value}${amount.unit}${headline.isComputed ? WORKED_OUT : ""}`]];
}

interface PartDrawerProps {
  ref: Ref<HTMLElement>;
  mode: ModeId;
  root: ModelPart;
  modelId: ModelId;
  modelTitle: string;
  part: ModelPart;
  place: PartPlace;
  tiny: TinyView;
  text: string;
  sentence: Sentence;
  onTextChange: (text: string) => void;
  nextPart: ModelPart | undefined;
  onOpen: (id: string) => void;
  onClose: () => void;
}

export function PartDrawer({ ref, mode, root, modelId, modelTitle, part, place, tiny, text, sentence, onTextChange, nextPart, onOpen, onClose }: PartDrawerProps) {
  const { facts } = part;
  const notes = facts.notes ?? [];
  const sources = [...new Set([...facts.sources, ...notes.map((note) => note.source)])];
  const reasons = (facts.reasons ?? []).filter((reason) => !notes.some((note) => note.value === reason.value));
  const makerReasons = reasons.filter((reason) => !SOURCES[reason.source].isCode);
  const codeReasons = reasons.filter((reason) => SOURCES[reason.source].isCode);
  const share = shareOf(part, place, true);
  const runs = runsOf(part, place, true);
  const copiesRows: [string, string][] = part.copies > 1 ? [["How many", `${count(part.copies)} of these`]] : [];
  const shareRows: [string, string][] = share ? [[shareLabelOf(place), `${share}${WORKED_OUT}`]] : [];
  const runsRows: [string, string][] = runs ? [[RUNS_LABEL, `${runs}${WORKED_OUT}`]] : [];
  const rows: [string, string][] = [
    ["Goes in", facts.inputDetail ?? facts.input],
    ["Comes out", facts.outputDetail ?? facts.output],
    ...headlineRows(facts),
    [isTableSize(facts.shape) ? "Table size (rows × columns)" : "Shape", facts.shape],
    ["Learned numbers", facts.weightCount > 0 ? `${numbersOf(part)}${WORKED_OUT}` : NONE_TO_LEARN],
    ...copiesRows,
    ...shareRows,
    ...runsRows,
    ["Inside", part.groundZero ? `Nothing smaller inside: ${SMALLEST_STEP_TEXT[part.groundZero]}` : partsOf(part.children ?? [])],
    ...notes.map((note): [string, string] => [note.label, note.value]),
  ];

  return (
    <Drawer ref={ref} id={DRAWER_ID} icon={part.icon} title={titleOf(part, root)} status={tagOf(part)} bodyKey={part.id} onClose={onClose} panel={part.story && <StoryDemo story={part.story} part={part} modelId={modelId} modelTitle={modelTitle} tiny={tiny} text={text} sentence={sentence} onTextChange={onTextChange} />}>
      <section className="section">
        <p className="section-title">What it does</p>
        <p className="part-summary">{part.explanation}</p>
        {part.term && <p className="card-lesson">{part.term}</p>}
      </section>
      <StoryInfo mode={mode} part={part} reasons={makerReasons} />
      <section className="section">
        <p className="section-title">{`The real part in ${modelTitle}`}</p>
        <FactRows rows={rows} />
        {rows.some(([, value]) => value.endsWith(WORKED_OUT)) && <p className="card-lesson">* Worked out from the sources, not stated in them directly.</p>}
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
      {codeReasons.length > 0 && (
        <section className="section">
          <p className="section-title">From the source code</p>
          <ReasonList reasons={codeReasons} />
        </section>
      )}
      {part.formula && <FormulaBlock formula={part.formula} />}
      {part.code && <CodeBlock code={part.code} />}
      {nextPart && (
        <button type="button" className="action story-next" onClick={() => onOpen(nextPart.id)}>
          Next step: {nextPart.title} <span aria-hidden="true">›</span>
        </button>
      )}
    </Drawer>
  );
}

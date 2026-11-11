import "./story.css";
import { Suspense } from "react";
import { type ModelLab, TARGET_ROUNDS } from "@/core/model/lab";
import { LAYER_COUNT, VECTOR_SIZE } from "@/core/model/tiny-gpt";
import type { DemoProps, ModeId, ModelPart, PartStory } from "@/features/inside-real-models/interfaces/model-part.interface";
import type { RealNote } from "@/features/inside-real-models/interfaces/real-facts.interface";
import type { Sentence } from "@/features/inside-real-models/interfaces/sentence.interface";
import { SOURCES } from "@/features/inside-real-models/trees/sources";
import { SentenceBox } from "./sentence-box";
import { DEMOS, isUsedOn } from "./story-demos";

const WHEN_TITLE: Record<ModeId, string> = { use: "When it runs", build: "When you'd do this" };

export function ReasonList({ reasons }: { reasons: RealNote[] }) {
  if (reasons.length === 0) return null;
  return (
    <ul className="reason-list">
      {reasons.map((reason) => (
        <li key={reason.value}>
          <b>{reason.label}:</b> {reason.value}{" "}
          <a className="card-lesson" href={SOURCES[reason.source].url} target="_blank" rel="noreferrer">
            {SOURCES[reason.source].title}
          </a>
        </li>
      ))}
    </ul>
  );
}

export function StoryInfo({ mode, part, reasons }: { mode: ModeId; part: ModelPart; reasons: RealNote[] }) {
  const { story } = part;
  return (
    <>
      <section className="section">
        <p className="section-title">Why it exists</p>
        {story && <p className="note">{story.problem}</p>}
        <p className="note">{part.why}</p>
        <ReasonList reasons={reasons} />
      </section>
      {story && (
        <section className="section">
          <p className="section-title">{WHEN_TITLE[mode]}</p>
          <p className="note">{story.when}</p>
          <p className="note">{story.feeds}</p>
        </section>
      )}
    </>
  );
}

interface StoryDemoProps extends DemoProps {
  story: PartStory;
  modelTitle: string;
  text: string;
  sentence: Sentence;
  onTextChange: (text: string) => void;
}

function TinyGptLine({ lab }: { lab: ModelLab }) {
  if (lab.latestLoss === null) return <p className="card-lesson">Tiny GPT is starting to learn.</p>;
  const loss = lab.latestLoss.toFixed(2);
  return (
    <p className="card-lesson">
      {`Tiny GPT ${lab.isTrained ? "finished learning" : "is learning"}: `}
      <span className="demo-model-numbers">{lab.isTrained ? `${lab.round} rounds · loss ${loss}` : `round ${lab.round} of ${TARGET_ROUNDS} · loss ${loss}`}</span>
      {" (lower loss = better guesses)"}
    </p>
  );
}

export function StoryDemo({ story, modelTitle, text, sentence, onTextChange, ...props }: StoryDemoProps) {
  const { component: Demo, usesSentence, usesTraining } = DEMOS[story.demo];
  const runsTinyGpt = isUsedOn(usesTraining, props.modelId);
  return (
    <section className="section watch-section" aria-label="Watch it happen">
      <h3 className="watch-heading">Watch it happen</h3>
      {runsTinyGpt ? (
        <>
          <p className="card-lesson">{`This demo runs Tiny GPT, our small practice model (${VECTOR_SIZE} numbers per word, ${LAYER_COUNT} layers). ${modelTitle}'s own trained numbers aren't in this app; its real sizes are in this part's facts.`}</p>
          <TinyGptLine lab={props.tiny.lab} />
        </>
      ) : (
        <p className="card-lesson">Try it here while you read. It runs live in your browser.</p>
      )}
      {isUsedOn(usesSentence, props.modelId) && <SentenceBox text={text} sentence={sentence} onChange={onTextChange} />}
      <Suspense fallback={<p className="note">Loading the demo…</p>}>
        <Demo {...props} />
      </Suspense>
    </section>
  );
}

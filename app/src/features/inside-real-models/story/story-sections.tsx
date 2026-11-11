import "./story.css";
import { Suspense } from "react";
import { type ModelLab, TARGET_ROUNDS } from "@/core/model/lab";
import type { DemoProps, ModelPart, PartStory } from "@/features/inside-real-models/interfaces/model-part.interface";
import type { RealNote } from "@/features/inside-real-models/interfaces/real-facts.interface";
import type { Sentence } from "@/features/inside-real-models/interfaces/sentence.interface";
import { SOURCES } from "@/features/inside-real-models/trees/sources";
import { SentenceBox } from "./sentence-box";
import { DEMOS, isUsedOn } from "./story-demos";

interface StoryInfoProps {
  story: PartStory;
  reasons: RealNote[];
  nextPart: ModelPart | undefined;
  onOpen: (id: string) => void;
}

export function StoryInfo({ story, reasons, nextPart, onOpen }: StoryInfoProps) {
  return (
    <>
      <section className="section">
        <p className="section-title">Why it exists</p>
        <p className="note">{story.problem}</p>
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
      </section>
      <section className="section">
        <p className="section-title">When you'd do this</p>
        <p className="note">{story.when}</p>
        <p className="why">{story.feeds}</p>
        {nextPart && (
          <button type="button" className="action story-next" onClick={() => onOpen(nextPart.id)}>
            Next step: {nextPart.title} <span aria-hidden="true">›</span>
          </button>
        )}
      </section>
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

function DemoModelLine({ lab, modelTitle }: { lab: ModelLab; modelTitle: string }) {
  const who = `Our tiny demo model, not ${modelTitle},`;
  if (lab.latestLoss === null) return <p className="card-lesson">{`${who} is starting to learn.`}</p>;
  const loss = lab.latestLoss.toFixed(2);
  return (
    <p className="card-lesson">
      {`${who} ${lab.isTrained ? "finished learning" : "is learning"}: `}
      <span className="demo-model-numbers">{lab.isTrained ? `${lab.round} rounds · loss ${loss}` : `round ${lab.round} of ${TARGET_ROUNDS} · loss ${loss}`}</span>
      {" (lower loss = better guesses)"}
    </p>
  );
}

export function StoryDemo({ story, modelTitle, text, sentence, onTextChange, ...props }: StoryDemoProps) {
  const { component: Demo, usesSentence, usesTraining } = DEMOS[story.demo];
  return (
    <section className="section watch-section" aria-label="Watch it happen">
      <h3 className="watch-heading">Watch it happen</h3>
      <p className="card-lesson">Try it here while you read. Everything below runs live in your browser on real data.</p>
      {isUsedOn(usesSentence, props.modelId) && <SentenceBox text={text} sentence={sentence} onChange={onTextChange} />}
      {isUsedOn(usesTraining, props.modelId) && <DemoModelLine lab={props.tiny.lab} modelTitle={modelTitle} />}
      <Suspense fallback={<p className="note">Loading the demo…</p>}>
        <Demo {...props} />
      </Suspense>
    </section>
  );
}

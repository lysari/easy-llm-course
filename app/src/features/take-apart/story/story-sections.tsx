import "./story.css";
import { Suspense } from "react";
import type { DemoProps, ModelPart, PartStory } from "@/features/take-apart/interfaces/model-part.interface";
import type { RealNote } from "@/features/take-apart/interfaces/real-facts.interface";
import { SOURCES } from "@/features/take-apart/trees/sources";
import { DEMOS } from "./story-demos";

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
            <li key={reason.label}>
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

export function StoryDemo({ story, ...props }: DemoProps & { story: PartStory }) {
  const Demo = DEMOS[story.demo];
  return (
    <section className="section watch-section" aria-label="Watch it happen">
      <h3 className="watch-heading">Watch it happen</h3>
      <p className="card-lesson">Try it here while you read. Everything below runs live in your browser on real data.</p>
      <Suspense fallback={<p className="note">Loading the demo…</p>}>
        <Demo {...props} />
      </Suspense>
    </section>
  );
}

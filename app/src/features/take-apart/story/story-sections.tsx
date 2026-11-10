import "./story.css";
import type { ReactNode } from "react";
import type { DemoId, ModelPart, PartStory } from "@/features/take-apart/interfaces/model-part.interface";
import type { RealNote } from "@/features/take-apart/interfaces/real-facts.interface";
import { SOURCES } from "@/features/take-apart/trees/sources";
import { CopyDocumentDemo } from "./copy-document-demo";

const DEMOS: Record<DemoId, () => ReactNode> = {
  "copy-document": CopyDocumentDemo,
};

interface StorySectionsProps {
  story: PartStory;
  reasons: RealNote[];
  nextPart: ModelPart | undefined;
  onOpen: (id: string) => void;
}

export function StorySections({ story, reasons, nextPart, onOpen }: StorySectionsProps) {
  const Demo = DEMOS[story.demo];
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
        <p className="section-title">Watch it happen</p>
        <Demo />
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

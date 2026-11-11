import type { Ref } from "react";
import type { ModelPart } from "@/features/inside-real-models/interfaces/model-part.interface";
import type { PartPlace } from "@/features/inside-real-models/interfaces/part-place.interface";
import { STAGGER_MS } from "@/shared/components/flow-links";
import { ConceptIcon } from "@/shared/components/concept-icon";
import { StatusPill } from "@/shared/components/status-pill";
import type { Status } from "@/shared/interfaces/card.interface";
import { count } from "@/shared/utils/format";
import { activateOnKey } from "@/shared/utils/keyboard";
import { InOut } from "./in-out";
import { PartNumbers } from "./part-numbers";

export const DRAWER_ID = "inside-real-models-drawer";

interface PartCardProps {
  ref: Ref<HTMLElement>;
  root: ModelPart;
  modelTitle: string;
  part: ModelPart;
  place: PartPlace;
  index: number;
  isSelected: boolean;
  isActive: boolean;
  isOpen: boolean;
  onSelect: () => void;
  onDetails: () => void;
}

export const tagOf = (part: ModelPart): Status | undefined => (part.groundZero ? "ground-zero" : undefined);

const nameOf = (part: ModelPart) => (part.copies > 1 ? `${part.title} × ${count(part.copies)}` : part.title);

const stepOf = (part: ModelPart, root: ModelPart) => (root.children?.indexOf(part) ?? -1) + 1;

export function titleOf(part: ModelPart, root: ModelPart) {
  const step = stepOf(part, root);
  return step > 0 ? `${step} ${nameOf(part)}` : nameOf(part);
}

export const partsOf = (children: ModelPart[]) => `${children.length} ${children.length === 1 ? "part" : "parts"}`;

export const numbersOf = (part: ModelPart) =>
  part.copies > 1 && part.facts.weightCount > 0 ? `${count(part.facts.weightCount)} each · ${count(part.facts.weightCount * part.copies)} in all` : count(part.facts.weightCount);

export function PartCard({ ref, root, modelTitle, part, place, index, isSelected, isActive, isOpen, onSelect, onDetails }: PartCardProps) {
  const step = stepOf(part, root);
  const tag = tagOf(part);
  return (
    <article
      ref={ref}
      className={`card part-card is-entering${isActive ? " is-active" : isSelected ? " is-selected" : ""}`}
      id={`inside-real-models-${part.id}`}
      tabIndex={0}
      aria-label={titleOf(part, root)}
      aria-current={isSelected}
      aria-expanded={isOpen}
      aria-controls={isOpen ? DRAWER_ID : undefined}
      aria-description={part.children ? "Opens its parts" : "Opens details"}
      style={{ animationDelay: `${index * STAGGER_MS}ms` }}
      onClick={onSelect}
      onKeyDown={activateOnKey}
    >
      <div className="part-card-head">
        {step > 0 ? <span className="step-badge">{step}</span> : <ConceptIcon name={part.icon} size="card" />}
        <div className="part-card-name">
          <h3 className="card-title">{nameOf(part)}</h3>
          {tag && <StatusPill status={tag} />}
        </div>
      </div>
      <div className="card-body">
        <p className="part-summary">{part.summary}</p>
        <InOut facts={part.facts} isDetailed={false} />
        <PartNumbers modelTitle={modelTitle} part={part} place={place} isExact={false} />
        <div className="part-card-foot">
          {part.children && <span>{partsOf(part.children)} <span aria-hidden="true">›</span></span>}
          <button
            type="button"
            className="action"
            onClick={(event) => {
              event.stopPropagation();
              onDetails();
            }}
          >
            Details <span aria-hidden="true">›</span>
          </button>
        </div>
      </div>
    </article>
  );
}

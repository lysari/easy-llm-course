import type { Ref } from "react";
import type { ModelPart, TinyView } from "@/features/take-apart/interfaces/model-part.interface";
import { FlowCard } from "@/shared/components/flow-card";
import { STAGGER_MS } from "@/shared/components/flow-links";
import type { Status } from "@/shared/interfaces/card.interface";
import { count } from "@/shared/utils/format";
import { readTinyPart } from "./tiny-parts";

export const DRAWER_ID = "take-apart-drawer";

const DETAILS = [{ label: "Details", command: "details" }];

interface PartCardProps {
  ref: Ref<HTMLElement>;
  root: ModelPart;
  part: ModelPart;
  tiny: TinyView;
  index: number;
  isSelected: boolean;
  isOpen: boolean;
  onSelect: () => void;
  onDetails: () => void;
}

export const statusOf = (part: ModelPart, tiny: TinyView): Status => (part.tiny.missing ? "lesson" : tiny.lab.isTrained ? "done" : "running");

export function titleOf(part: ModelPart, root: ModelPart) {
  const title = part.copies > 1 ? `${part.title} × ${count(part.copies)}` : part.title;
  const step = root.children?.indexOf(part) ?? -1;
  return step >= 0 ? `${step + 1} ${title}` : title;
}

export const numbersOf = (part: ModelPart) =>
  part.copies > 1 && part.facts.weightCount > 0 ? `${count(part.facts.weightCount)} each · ${count(part.facts.weightCount * part.copies)} in all` : count(part.facts.weightCount);

export function PartCard({ ref, root, part, tiny, index, isSelected, isOpen, onSelect, onDetails }: PartCardProps) {
  const readout = readTinyPart(part.tiny, tiny);
  const tinyNumbers = readout.weightCount === 0 ? "no stored numbers" : `${count(readout.weightCount)} numbers`;
  return (
    <FlowCard
      ref={ref}
      className={`is-entering${isSelected ? " is-active" : ""}${isOpen ? " is-open" : ""}`}
      id={`take-apart-${part.id}`}
      aria-current={isSelected}
      aria-expanded={isOpen}
      aria-controls={isOpen ? DRAWER_ID : undefined}
      aria-description={part.children ? "Opens its parts" : "Opens details"}
      style={{ animationDelay: `${index * STAGGER_MS}ms` }}
      onClick={onSelect}
      icon={part.icon}
      title={titleOf(part, root)}
      lesson={`${part.lesson} · ${part.reverseLesson}`}
      status={statusOf(part, tiny)}
      task={`${part.facts.input} → ${part.facts.output}`}
      foot={part.tiny.missing ? `Tiny model: nearest is ${readout.title}` : <>Tiny model live <b>{readout.live}</b></>}
      actions={DETAILS}
      onCommand={onDetails}
    >
      <dl className="part-facts">
        <div>
          <dt>{root.children?.includes(part) ? "Real model" : "Shape"}</dt>
          <dd>{part.facts.shape}</dd>
        </div>
        <div>
          <dt>Numbers</dt>
          <dd>{numbersOf(part)}</dd>
        </div>
        <div>
          <dt>Tiny model</dt>
          <dd>{part.tiny.missing ? "Does not have this part" : `${readout.shape} · ${tinyNumbers}`}</dd>
        </div>
        <div>
          <dt>Inside</dt>
          <dd>{part.children ? `${part.children.length} ${part.children.length === 1 ? "part" : "parts"} ›` : "Ground zero"}</dd>
        </div>
      </dl>
    </FlowCard>
  );
}

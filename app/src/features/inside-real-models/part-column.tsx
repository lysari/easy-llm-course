import { type RefObject, useMemo, useRef } from "react";
import type { ModeId, ModelPart, RealModel } from "@/features/inside-real-models/interfaces/model-part.interface";
import { FlowLinks } from "@/shared/components/flow-links";
import { useFlowLinks } from "@/shared/hooks/use-flow-links";
import { PartCard, titleOf } from "./part-card";
import { placeOf } from "./real-place";

interface PartColumnProps {
  model: RealModel;
  mode: ModeId;
  root: ModelPart;
  parent: ModelPart;
  parts: ModelPart[];
  selectedId: string | undefined;
  activeId: string;
  pathIds: string[];
  openId: string | null;
  cards: RefObject<Map<string, HTMLElement>>;
  hasLinks: boolean;
  width: number;
  onSelect: (part: ModelPart) => void;
  onDetails: (id: string) => void;
}

const STEP_WORD: Record<ModeId, string> = { use: "step", build: "stage" };

export const stepsTitleOf = (model: RealModel, mode: ModeId) => `${model.title}, ${STEP_WORD[mode]} by ${STEP_WORD[mode]}`;

export function PartColumn({ model, mode, root, parent, parts, selectedId, activeId, pathIds, openId, cards, hasLinks, width, onSelect, onDetails }: PartColumnProps) {
  const columnRef = useRef<HTMLElement>(null);
  const partIds = useMemo(() => parts.map((part) => part.id), [parts]);
  const isSteps = parent === root;
  const links = useFlowLinks(
    { flowRef: columnRef, childColumnRef: columnRef, parentCards: cards, childCards: cards },
    { parentId: parent.id, childIds: partIds, isShown: hasLinks && !isSteps, width },
  );
  const title = isSteps ? stepsTitleOf(model, mode) : `Inside ${titleOf(parent, root)}`;

  return (
    <section className="column part-column inside-real-models-column" ref={columnRef} aria-label={title}>
      {!isSteps && <FlowLinks id={`inside-real-models-links-${parent.id}`} drawing={links} pathIds={pathIds} glowIds={[activeId]} />}
      <p className="section-title">{title}</p>
      {isSteps && <p className="card-lesson">{`The ${parts.length} numbered cards are the whole story. Open deeper cards only if you're curious.`}</p>}
      {parts.map((part, i) => (
        <PartCard
          key={part.id}
          ref={(element) => {
            if (element) cards.current.set(part.id, element);
            else cards.current.delete(part.id);
          }}
          root={root}
          part={part}
          place={placeOf(model, root, part)}
          index={i}
          isSelected={part.id === selectedId}
          isActive={part.id === activeId}
          isOpen={part.id === openId}
          onSelect={() => onSelect(part)}
          onDetails={() => onDetails(part.id)}
        />
      ))}
    </section>
  );
}

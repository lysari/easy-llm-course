import { type RefObject, useMemo, useRef } from "react";
import type { ModelPart, TinyView } from "@/features/take-apart/interfaces/model-part.interface";
import { FlowLinks } from "@/shared/components/flow-links";
import { useFlowLinks } from "@/shared/hooks/use-flow-links";
import { PartCard, titleOf } from "./part-card";

interface PartColumnProps {
  root: ModelPart;
  modelTitle: string;
  parent: ModelPart;
  parts: ModelPart[];
  selectedId: string | undefined;
  openId: string | null;
  cards: RefObject<Map<string, HTMLElement>>;
  tiny: TinyView;
  hasLinks: boolean;
  width: number;
  onSelect: (part: ModelPart) => void;
  onDetails: (id: string) => void;
}

export function PartColumn({ root, modelTitle, parent, parts, selectedId, openId, cards, tiny, hasLinks, width, onSelect, onDetails }: PartColumnProps) {
  const columnRef = useRef<HTMLElement>(null);
  const partIds = useMemo(() => parts.map((part) => part.id), [parts]);
  const isSteps = parent === root;
  const links = useFlowLinks(
    { flowRef: columnRef, childColumnRef: columnRef, parentCards: cards, childCards: cards },
    { parentId: parent.id, childIds: partIds, isShown: hasLinks && !isSteps, width },
  );
  const title = isSteps ? `${root.title} in ${root.facts.shape}` : `Inside ${titleOf(parent, root)}`;

  return (
    <section className="column part-column take-apart-column" ref={columnRef} aria-label={title}>
      {!isSteps && <FlowLinks id={`take-apart-links-${parent.id}`} drawing={links} />}
      <p className="section-title">{title}</p>
      {parts.map((part, i) => (
        <PartCard
          key={part.id}
          ref={(element) => {
            if (element) cards.current.set(part.id, element);
            else cards.current.delete(part.id);
          }}
          root={root}
          modelTitle={modelTitle}
          part={part}
          tiny={tiny}
          index={i}
          isSelected={part.id === selectedId}
          isOpen={part.id === openId}
          onSelect={() => onSelect(part)}
          onDetails={() => onDetails(part.id)}
        />
      ))}
    </section>
  );
}

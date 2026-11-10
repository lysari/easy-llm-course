import type { ModelPart } from "@/features/inside-real-models/interfaces/model-part.interface";
import type { PartPlace } from "@/features/inside-real-models/interfaces/part-place.interface";
import type { Headline } from "@/features/inside-real-models/interfaces/real-facts.interface";
import { keyFactOf, showAmount } from "./real-place";

const COMPUTED = "Worked out from the sources";

function RealHeadline({ headline, isExact }: { headline: Headline; isExact: boolean }) {
  if ("text" in headline) return <dd className="is-muted">{headline.text}</dd>;
  const isPercent = headline.unit.startsWith("%");
  return (
    <dd>
      <b>{`${showAmount(headline.value, isExact || Boolean(headline.isId))}${headline.isOver ? "+" : ""}${isPercent ? "%" : ""}`}</b>
      {isPercent ? headline.unit.slice(1) : ` ${headline.unit}`}
      {isExact && <span className="part-numbers-basis">{headline.isComputed ? COMPUTED : "As published"}</span>}
    </dd>
  );
}

interface PartNumbersProps {
  modelTitle: string;
  part: ModelPart;
  place: PartPlace;
  isExact: boolean;
}

export function PartNumbers({ modelTitle, part, place, isExact }: PartNumbersProps) {
  const key = keyFactOf(part, place, isExact);
  return (
    <dl className="part-numbers">
      <div>
        <dt>{`Real ${modelTitle}`}</dt>
        <RealHeadline headline={part.facts.headline} isExact={isExact} />
      </div>
      <div className="part-numbers-key">
        <dt>{key.label}</dt>
        <dd>
          {key.value}
          {isExact && key.isComputed && <span className="part-numbers-basis">{COMPUTED}</span>}
        </dd>
      </div>
    </dl>
  );
}

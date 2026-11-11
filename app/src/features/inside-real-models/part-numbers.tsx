import type { ModelPart } from "@/features/inside-real-models/interfaces/model-part.interface";
import type { PartPlace } from "@/features/inside-real-models/interfaces/part-place.interface";
import type { Headline } from "@/features/inside-real-models/interfaces/real-facts.interface";
import { amountOf, keyFactOf } from "./real-place";

function RealHeadline({ headline }: { headline: Headline }) {
  if ("text" in headline) return <p>{headline.text}</p>;
  const amount = amountOf(headline, false);
  return (
    <p>
      <b>{amount.value}</b>
      {amount.unit}
    </p>
  );
}

export function PartNumbers({ part, place }: { part: ModelPart; place: PartPlace }) {
  const key = keyFactOf(part, place, false);
  return (
    <div className="part-numbers">
      <RealHeadline headline={part.facts.headline} />
      <p className="part-numbers-key">
        <span>{key.label}</span>
        {key.value}
      </p>
    </div>
  );
}

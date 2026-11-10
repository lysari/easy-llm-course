import type { ModelPart, TinyView } from "@/features/take-apart/interfaces/model-part.interface";
import type { Amount, Headline } from "@/features/take-apart/interfaces/real-facts.interface";
import { count } from "@/shared/utils/format";
import { readTinyPart } from "./tiny-parts";

const COMPACT = new Intl.NumberFormat("en-US", { notation: "compact", maximumSignificantDigits: 3 });
const COMPACT_FROM = 100_000;

export const NONE_TO_LEARN = "None to learn";

export const showAmount = (value: number, isExact: boolean) => (!isExact && value >= COMPACT_FROM ? COMPACT.format(value) : count(value));

function AmountText({ amount, isExact }: { amount: Amount; isExact: boolean }) {
  const isPercent = amount.unit.startsWith("%");
  return (
    <>
      <b>{`${showAmount(amount.value, isExact || Boolean(amount.isId))}${amount.isOver ? "+" : ""}${isPercent ? "%" : ""}`}</b>
      {isPercent ? amount.unit.slice(1) : ` ${amount.unit}`}
    </>
  );
}

function RealHeadline({ headline, isExact }: { headline: Headline; isExact: boolean }) {
  if ("text" in headline) return <dd className="is-muted">{headline.text}</dd>;
  return (
    <dd>
      <AmountText amount={headline} isExact={isExact} />
      {isExact && <span className="part-numbers-basis">{headline.isComputed ? "Worked out from the sources" : "As published"}</span>}
    </dd>
  );
}

interface PartNumbersProps {
  modelTitle: string;
  part: ModelPart;
  tiny: TinyView;
  isExact: boolean;
}

export function PartNumbers({ modelTitle, part, tiny, isExact }: PartNumbersProps) {
  const headline = part.tiny.missing ? null : readTinyPart(part.tiny, tiny).headline;
  const amount = headline && !("reason" in headline) ? headline : null;
  return (
    <dl className="part-numbers">
      <div>
        <dt>{`Real ${modelTitle}`}</dt>
        <RealHeadline headline={part.facts.headline} isExact={isExact} />
      </div>
      <div>
        <dt>
          Tiny model
          {amount && !tiny.lab.isTrained && <span className="live-tag">live</span>}
        </dt>
        {amount ? (
          <dd>
            <AmountText amount={amount} isExact={isExact} />
          </dd>
        ) : <dd className="is-muted">{headline ? NONE_TO_LEARN : "Not in the tiny model"}</dd>}
      </div>
    </dl>
  );
}

import type { Amount, ModelPart, TinyView } from "@/features/take-apart/interfaces/model-part.interface";
import { count } from "@/shared/utils/format";
import { readTinyPart } from "./tiny-parts";

const COMPACT = new Intl.NumberFormat("en-US", { notation: "compact", maximumSignificantDigits: 3 });
const COMPACT_FROM = 100_000;

export const showAmount = (value: number, isExact: boolean) => (!isExact && value >= COMPACT_FROM ? COMPACT.format(value) : count(value));

function AmountText({ amount, isExact }: { amount: Amount; isExact: boolean }) {
  const isPercent = amount.unit.startsWith("%");
  return (
    <>
      <b>{`${showAmount(amount.value, isExact)}${amount.isOver ? "+" : ""}${isPercent ? "%" : ""}`}</b>
      {isPercent ? amount.unit.slice(1) : ` ${amount.unit}`}
    </>
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
        <dd>
          <AmountText amount={part.facts.headline} isExact={isExact} />
        </dd>
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
        ) : <dd className="is-muted">{headline ? "None to learn" : "Not in the tiny model"}</dd>}
      </div>
    </dl>
  );
}

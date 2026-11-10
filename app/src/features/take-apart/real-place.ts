import type { ModelPart, RealModel } from "@/features/take-apart/interfaces/model-part.interface";
import type { PartPlace } from "@/features/take-apart/interfaces/part-place.interface";
import type { KeyFact } from "@/features/take-apart/interfaces/real-facts.interface";
import { count } from "@/shared/utils/format";
import { pathTo } from "./models";

const COMPACT = new Intl.NumberFormat("en-US", { notation: "compact", maximumSignificantDigits: 3 });
const COMPACT_FROM = 100_000;
const SHARE = new Intl.NumberFormat("en-US", { style: "percent", maximumSignificantDigits: 3 });

export const SHARE_LABEL = "Share of all numbers";
export const RUNS_LABEL = "Runs per word";

export const showAmount = (value: number, isExact: boolean) => (!isExact && value >= COMPACT_FROM ? COMPACT.format(value) : count(value));

export function placeOf(model: RealModel, root: ModelPart, part: ModelPart): PartPlace {
  const path = pathTo(root, part.id);
  if (!path) throw new Error(`${part.id} is not in ${root.id}`);
  return {
    path,
    copies: path.slice(1).reduce((all, each) => all * each.copies, 1),
    isInModel: pathTo(model.root, part.id) !== null,
    isUpTo: path.some((each) => each.facts.isPerEarlierWord),
    total: model.root.facts.weightCount,
  };
}

export function shareOf(part: ModelPart, place: PartPlace, isExact: boolean) {
  const all = part.facts.weightCount * place.copies;
  if (all === 0 || all >= place.total) return null;
  const share = SHARE.format(all / place.total);
  if (isExact) return `${count(all)} of ${count(place.total)} (${share})${place.copies > 1 ? `, all ${count(place.copies)} together` : ""}`;
  return place.copies > 1 ? `${share} for all ${showAmount(place.copies, false)}` : share;
}

export function runsOf(part: ModelPart, place: PartPlace, isExact: boolean) {
  if (!place.isInModel || (part.facts.weightCount > 0 && part.groundZero !== "multiply-add")) return null;
  const times = place.copies === 1 ? "once" : `${showAmount(place.copies, isExact)} times`;
  return place.isUpTo ? `up to ${times}` : times;
}

export function keyFactOf(part: ModelPart, place: PartPlace, isExact: boolean): KeyFact {
  const [note] = part.facts.notes ?? [];
  if (note) return { ...note, isComputed: false };
  const source = part.facts.sources[0];
  const runs = runsOf(part, place, isExact);
  if (runs) return { label: RUNS_LABEL, value: runs, source, isComputed: true };
  const share = shareOf(part, place, isExact);
  if (share) return { label: SHARE_LABEL, value: share, source, isComputed: true };
  return { label: "Shape", value: part.facts.shape, source, isComputed: false };
}

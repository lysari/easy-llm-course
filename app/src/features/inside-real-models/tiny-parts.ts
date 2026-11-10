import { VOCABULARY } from "@/core/model/corpus";
import { nextOptions, writeText } from "@/core/model/generate";
import type { ModelLab } from "@/core/model/lab";
import { seededRandom } from "@/core/model/random";
import { traceForward } from "@/core/model/tiny-gpt";
import type { TinyView } from "@/features/inside-real-models/interfaces/model-part.interface";

const WRITE_SEED = 7;

export function viewTiny(lab: ModelLab, words: string[]): TinyView {
  const weights = lab.live;
  const ids = words.map((word) => VOCABULARY.indexOf(word));
  return {
    lab,
    weights,
    words,
    ids,
    trace: traceForward(ids, weights),
    options: nextOptions(words, weights),
    written: writeText(words, weights, seededRandom(WRITE_SEED)).slice(words.length),
  };
}

import { useState } from "react";
import { TEST_ANSWER, TEST_WORDS, VOCABULARY, joinTokens } from "@/core/model/corpus";
import { nextOptions } from "@/core/model/generate";
import { countWeights } from "@/core/model/tiny-gpt";
import { FactRows } from "@/features/inside-real-models/fact-rows";
import type { DemoProps } from "@/features/inside-real-models/interfaces/model-part.interface";
import { MODELS } from "@/features/inside-real-models/models";
import { BeforeAfter, HonestNote } from "@/features/inside-real-models/story/watch-parts";
import { ChanceBars } from "@/shared/components/chance-bars";
import { count } from "@/shared/utils/format";
import { BIT_CHOICES, TINY_BITS, levelsFor, quantizeWeights, roundNumber, scaleFor } from "./evaluation-math";

const BITS_PER_BYTE = 8;

const sizeOf = (numbers: number, bits: number) => {
  const bytes = (numbers * bits) / BITS_PER_BYTE;
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(2)} GB` : bytes >= 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${count(Math.round(bytes / 1e3))} KB`;
};

export function QuantizeDemo({ part, modelId, tiny }: DemoProps) {
  const [bits, setBits] = useState(BIT_CHOICES[0]);
  const numbers = countWeights(tiny.weights);
  const table = tiny.weights.tokenTable.data;
  const scale = scaleFor(table, bits);
  const one = roundNumber(table[0], scale, bits);
  const real = MODELS.find((model) => model.id === modelId) ?? MODELS[0];
  const realCount = real.root.facts.weightCount;
  const shrunk = part.id === "quantization" ? quantizeWeights(tiny.weights, bits) : null;

  return (
    <div className="watch">
      <div className="watch-controls" role="group" aria-label="Bits per number">
        <span className="card-lesson">Bits per number:</span>
        {BIT_CHOICES.map((choice) => (
          <button key={choice} type="button" className={`word-chip${choice === bits ? " is-picked" : ""}`} aria-pressed={choice === bits} onClick={() => setBits(choice)}>
            {choice}
          </button>
        ))}
        <span className="card-lesson">{`${bits} bits hold whole numbers from −${levelsFor(bits)} to ${levelsFor(bits)}`}</span>
      </div>
      {part.id === "round-number" ? (
        <BeforeAfter
          beforeLabel={`First number of "${VOCABULARY[0]}" in our tiny word table, full size`}
          before={<p className="term-big">{table[0].toFixed(6)}</p>}
          afterLabel={`Stored in ${bits} bits`}
          after={<FactRows rows={[["Scale (biggest number ÷ " + levelsFor(bits) + ")", scale.toFixed(6)], ["Number ÷ scale, rounded", `${count(one.level)}`], ["Read back: whole number × scale", one.back.toFixed(6)], ["Lost by rounding", one.error.toFixed(6)]]} />}
        />
      ) : (
        shrunk && (
          <BeforeAfter
            beforeLabel={`Full size: ${count(numbers)} numbers × ${TINY_BITS} bits = ${sizeOf(numbers, TINY_BITS)}`}
            before={<ChanceBars options={nextOptions(TEST_WORDS, tiny.weights)} answer={TEST_ANSWER} />}
            afterLabel={`${bits} bits each: ${sizeOf(numbers, bits)}, biggest change ${shrunk.biggestError.toFixed(4)}`}
            after={<ChanceBars options={nextOptions(TEST_WORDS, shrunk.weights)} answer={TEST_ANSWER} />}
          />
        )
      )}
      {part.id === "quantization" && <p className="card-lesson">{`Next-word chances after "${joinTokens(TEST_WORDS)}", from the full-size and the rounded numbers.`}</p>}
      <FactRows rows={[[`Real ${real.title}: ${count(realCount)} numbers at 16 bits`, sizeOf(realCount, 16)], [`… at ${bits} bits`, sizeOf(realCount, bits)]]} />
      <HonestNote>{`Rounded live: every table of our tiny model, each with its own scale (its biggest number ÷ ${levelsFor(bits)}). The real sizes are worked out from the published number count, ${modelId === "gpt-2-small" ? "not from a file OpenAI published at that size; no quantization of GPT-2 is published in its paper or model card." : "and Llama 3 8B was released at 16 bits (bfloat16); Meta published no quantized version in its model card or blog."}`}</HonestNote>
    </div>
  );
}

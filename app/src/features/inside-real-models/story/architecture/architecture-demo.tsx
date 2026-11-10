import "./architecture.css";
import { useState } from "react";
import { joinTokens } from "@/core/model/corpus";
import { nextOptions } from "@/core/model/generate";
import { LAYER_COUNT, countWeights, createWeights } from "@/core/model/tiny-gpt";
import { FactRows } from "@/features/inside-real-models/fact-rows";
import type { DemoProps, ModelPart } from "@/features/inside-real-models/interfaces/model-part.interface";
import { HonestNote, NumberRow } from "@/features/inside-real-models/story/watch-parts";
import { ChanceBars } from "@/shared/components/chance-bars";
import { count, showWord } from "@/shared/utils/format";

const FIRST_SEED = 42;

const partRows = (parts: ModelPart[]) => parts.map((each): [string, string] => [each.copies > 1 ? `${each.title} × ${count(each.copies)}` : each.title, `${count(each.facts.weightCount * each.copies)} numbers`]);

function RealShape({ model, modelTitle }: { model: ModelPart; modelTitle: string }) {
  return (
    <div className="watch-box">
      <p className="section-title">{`The real ${modelTitle}, worked out from its published settings`}</p>
      <FactRows rows={[...partRows(model.children ?? []), ["In all", `${count(model.facts.weightCount)} numbers`]]} />
    </div>
  );
}

export function ArchitectureDemo({ part, tiny }: DemoProps) {
  const [seed, setSeed] = useState(FIRST_SEED);
  const model = part.id === "architecture" ? part.children?.[0] : part;
  const random = seed === FIRST_SEED ? tiny.lab.untrained : createWeights(seed);
  const sentence = joinTokens(tiny.words);
  const last = tiny.ids.length - 1;

  if (part.id !== "architecture") {
    return (
      <div className="watch">
        <p className="note">{`The whole tiny model, live on "${sentence}": IDs in, one score per word out. Nothing else happens in between but table lookups, multiplies and adds.`}</p>
        <FactRows rows={tiny.words.map((word, i): [string, string] => [showWord(word), `ID ${tiny.ids[i]}`])} />
        <NumberRow label={`The last word's numbers going in (word row + place row)`} values={tiny.trace.combined[last]} />
        {tiny.trace.layers.map((layer, i) => (
          <NumberRow key={i} label={`After layer ${i + 1} of ${LAYER_COUNT}`} values={layer.afterFeedForward[last]} />
        ))}
        <p className="section-title">Scores turned into chances for the next word</p>
        <ChanceBars options={tiny.options} answer={tiny.options[0].token} />
        {model && <RealShape model={model} modelTitle={part.title} />}
        <HonestNote>{`Live in our tiny model (${count(countWeights(tiny.weights))} numbers, round ${tiny.lab.round} of training). The real model runs the same kind of pass with its own published sizes.`}</HonestNote>
      </div>
    );
  }

  return (
    <div className="watch">
      <p className="note">{`Same code, two sets of numbers. Both columns run our tiny model's code on "${sentence}". Only the stored numbers differ.`}</p>
      <div className="watch-grid">
        <div className="watch-box">
          <p className="section-title">{seed === FIRST_SEED ? "Its starting random numbers" : `New random numbers (seed ${seed})`}</p>
          <ChanceBars options={nextOptions(tiny.words, random)} answer="" />
          <button type="button" className="action" onClick={() => setSeed(seed + 1)}>
            Draw new random numbers
          </button>
        </div>
        <div className="watch-box">
          <p className="section-title">{`The same code after ${tiny.lab.round} training rounds`}</p>
          <ChanceBars options={tiny.options} answer="" />
        </div>
      </div>
      <p className="card-lesson">With random numbers the guesses are close to even and change with every draw. Training is what makes one word stand out.</p>
      {model && <RealShape model={model} modelTitle={model.title} />}
      <HonestNote>The guesses are live from our tiny model. The real model's part sizes are worked out exactly from its published settings; its own random starting numbers are not published.</HonestNote>
    </div>
  );
}

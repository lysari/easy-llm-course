import "./output.css";
import { useId, useState } from "react";
import { VOCABULARY, joinTokens } from "@/core/model/corpus";
import { LAYER_COUNT } from "@/core/model/tiny-gpt";
import { FactRows } from "@/features/inside-real-models/fact-rows";
import type { DemoProps } from "@/features/inside-real-models/interfaces/model-part.interface";
import { MODELS } from "@/features/inside-real-models/models";
import { layerNormParts } from "@/features/inside-real-models/story/architecture/exact-formulas";
import { BeforeAfter, HonestNote, NumberRow } from "@/features/inside-real-models/story/watch-parts";
import { ChanceBars } from "@/shared/components/chance-bars";
import { count, percent, showWord } from "@/shared/utils/format";
import { type ChanceRow, showPower, softmaxOf } from "./next-word";

function ScoreTable({ rows, total, picked }: { rows: ChanceRow[]; total: number; picked: number }) {
  return (
    <div className="score-scroll">
      <table className="score-table">
        <thead>
          <tr>
            <th scope="col">Word</th>
            <th scope="col">Score</th>
            <th scope="col">e^score</th>
            <th scope="col">Chance</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id} className={row.id === picked ? "is-picked" : undefined}>
              <th scope="row">{showWord(row.word)}</th>
              <td>{row.score.toFixed(3)}</td>
              <td>{showPower(row.power)}</td>
              <td>{`${(row.chance * 100).toFixed(2)}%`}</td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <th scope="row">Total</th>
            <td />
            <td>{showPower(total)}</td>
            <td>100%</td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

export function NextWordDemo({ part, modelId, tiny }: DemoProps) {
  const last = tiny.ids.length - 1;
  const finalIn = tiny.trace.layers[LAYER_COUNT - 1].afterFeedForward[last];
  const { rows, total } = softmaxOf(tiny.trace.logits[last]);
  const [picked, setPicked] = useState(rows[0].id);
  const wordId = useId();
  const real = MODELS.find((model) => model.id === modelId) ?? MODELS[0];
  const { headline } = part.facts;
  const one = rows.find((row) => row.id === picked) ?? rows[0];
  const sentence = joinTokens(tiny.words);

  return (
    <div className="watch">
      <p className="note">{`Our tiny model, live on "${sentence}": what it gives for the word after "${showWord(tiny.words[last])}".`}</p>
      {part.id === "output" && (
        <>
          <BeforeAfter
            beforeLabel={`The last word's numbers after layer ${LAYER_COUNT}`}
            before={<NumberRow label="into the final norm" values={finalIn} />}
            afterLabel="Steadied by the final norm"
            after={<NumberRow label="average 0, spread 1" values={layerNormParts(finalIn).out} />}
          />
          <BeforeAfter
            beforeLabel={`Scored against all ${count(VOCABULARY.length)} words in its list`}
            before={<FactRows rows={rows.slice(0, 6).map((row): [string, string] => [showWord(row.word), `score ${row.score.toFixed(3)}`])} />}
            afterLabel="Turned into chances by softmax"
            after={<ChanceBars options={rows.map((row) => ({ token: row.word, chance: row.chance }))} answer={rows[0].word} limit={6} />}
          />
        </>
      )}
      {part.id === "softmax" && <ScoreTable rows={rows} total={total} picked={rows[0].id} />}
      {part.id === "chance" && (
        <>
          <div className="watch-slider">
            <label htmlFor={wordId}>Word</label>
            <select id={wordId} className="watch-select" value={one.id} onChange={(event) => setPicked(Number(event.target.value))}>
              {rows.map((row) => (
                <option key={row.id} value={row.id}>
                  {showWord(row.word)}
                </option>
              ))}
            </select>
          </div>
          <BeforeAfter
            beforeLabel={`"${showWord(one.word)}": its score, raised as a power of e`}
            before={<FactRows rows={[["Score", one.score.toFixed(4)], ["e^score", showPower(one.power)], [`Total of e^score for all ${count(VOCABULARY.length)} words`, showPower(total)]]} />}
            afterLabel="Divided by the total"
            after={<p className="term-big">{`${showPower(one.power)} ÷ ${showPower(total)} = ${one.chance.toFixed(4)} (${percent(one.chance)})`}</p>}
          />
          <ScoreTable rows={rows} total={total} picked={one.id} />
        </>
      )}
      <HonestNote>
        {`Worked out live from our tiny model's real scores, ${count(VOCABULARY.length)} of them, one per word it knows; ${real.title} makes ${"value" in headline ? count(headline.value) : "one per word"} the same way. ${modelId === "gpt-2-small" ? "GPT-2's own trained numbers are not shipped in this app, so its scores are not shown." : "Llama 3's trained numbers are only given out under Meta's licence, so its scores are not shown. Its final norm is RMSNorm; our tiny model's is LayerNorm."}`}
      </HonestNote>
    </div>
  );
}

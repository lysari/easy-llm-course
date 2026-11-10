import { useId, useState } from "react";
import { END_TOKEN, MAX_CONTEXT, VOCABULARY } from "@/core/model/corpus";
import { LAYER_COUNT, VECTOR_SIZE } from "@/core/model/tiny-gpt";
import { FactRows } from "@/features/take-apart/fact-rows";
import type { DemoProps } from "@/features/take-apart/interfaces/model-part.interface";
import { MODELS } from "@/features/take-apart/models";
import { BeforeAfter, HonestNote, NumberRow, WordPicker } from "@/features/take-apart/story/watch-parts";
import { count, showWord } from "@/shared/utils/format";
import { cacheFor, cacheNumbers, realCacheOf, reuseGap, wordPasses } from "./kv-cache";

const BYTE_CHOICES = [2, 4];

const sizeOf = (bytes: number) => (bytes >= 1e9 ? `${(bytes / 1e9).toFixed(2)} GB` : bytes >= 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${count(Math.round(bytes / 1e3))} KB`);

export function KvCacheDemo({ modelId, tiny }: DemoProps) {
  const last = tiny.ids.length - 1;
  const [word, setWord] = useState(last);
  const [bytes, setBytes] = useState(BYTE_CHOICES[0]);
  const [answer, setAnswer] = useState(MAX_CONTEXT - tiny.ids.length);
  const answerId = useId();
  const picked = Math.min(word, last);
  const cache = cacheFor(tiny.ids, tiny.weights);
  const nextWord = tiny.written[0]?.word ?? END_TOKEN;
  const gap = reuseGap(tiny.ids, VOCABULARY.indexOf(nextWord), tiny.weights);
  const most = MAX_CONTEXT - tiny.ids.length;
  const passes = wordPasses(tiny.ids.length, Math.min(answer, most));
  const model = MODELS.find((each) => each.id === modelId) ?? MODELS[0];
  const real = realCacheOf(model);
  const perWord = cacheNumbers(real.layers, real.kvWidth, 1);
  const full = cacheNumbers(real.layers, real.kvWidth, real.context);
  const hasGroups = real.kvWidth < real.width;
  const perHead = cacheNumbers(real.layers, real.width, real.context);
  const ungrouped: [string, string] = ["If every query head kept its own key and value", `${count(perHead)} numbers, ${sizeOf(perHead * bytes)}: ${real.width / real.kvWidth} times more`];

  return (
    <div className="watch">
      <WordPicker label="What is saved for this word of your sentence:" words={tiny.words} picked={picked} onPick={setWord} />
      <div className="watch-grid">
        {cache.map((layer, l) => (
          <div key={l} className="watch-box">
            <p className="section-title">{`Layer ${l + 1}`}</p>
            <NumberRow label="its key" values={layer[picked].keys} />
            <NumberRow label="its value" values={layer[picked].values} />
          </div>
        ))}
      </div>
      <p className="card-lesson">{`Our tiny model: 2 × ${VECTOR_SIZE} numbers per layer × ${LAYER_COUNT} layers = ${count(cacheNumbers(LAYER_COUNT, VECTOR_SIZE, 1))} numbers saved for each word.`}</p>
      <BeforeAfter
        beforeLabel={`Keys and values of all ${tiny.ids.length} words, worked out once`}
        before={<p className="note">Each word's key and value only depend on that word and the words before it.</p>}
        afterLabel={`Worked out again with "${showWord(nextWord)}" added at the end`}
        after={
          <>
            <p className="note">Largest change in any earlier word's key or value</p>
            <p className="term-big">{gap}</p>
            <span className={`verdict ${gap === 0 ? "is-keep" : "is-drop"}`}>{gap === 0 ? "Exactly the same, so they can be saved and reused" : "Changed"}</span>
          </>
        }
      />
      <div className="watch-box">
        <div className="watch-slider">
          <label htmlFor={answerId}>{`Words to write after your ${tiny.ids.length}: ${Math.min(answer, most)}`}</label>
          <input id={answerId} type="range" min={0} max={most} value={Math.min(answer, most)} onChange={(event) => setAnswer(Number(event.target.value))} />
        </div>
        <FactRows
          rows={[
            ["Words run through the layers, no cache", `${count(passes.without)}: the whole text, every round`],
            ["With the cache", `${count(passes.with)}: your words once, then only each new word`],
          ]}
        />
      </div>
      <div className="watch-box">
        <p className="section-title">{`The real ${model.title}, worked out from its settings`}</p>
        <div className="watch-controls" role="group" aria-label="Bytes per number">
          <span className="card-lesson">Number format (you pick):</span>
          {BYTE_CHOICES.map((choice) => (
            <button key={choice} type="button" className={`word-chip${choice === bytes ? " is-picked" : ""}`} aria-pressed={choice === bytes} onClick={() => setBytes(choice)}>
              {`${choice * 8}-bit`}
            </button>
          ))}
        </div>
        <FactRows
          rows={[
            ["Per word, per layer", `2 × ${count(real.kvWidth)} = ${count(2 * real.kvWidth)} numbers`],
            [`Per word, all ${real.layers} layers`, `${count(perWord)} numbers, ${sizeOf(perWord * bytes)}`],
            [`A full text of ${count(real.context)} words`, `${count(full)} numbers, ${sizeOf(full * bytes)}`],
            ...(hasGroups ? [ungrouped] : []),
          ]}
        />
      </div>
      <HonestNote>{`Our tiny model keeps no cache: it runs the whole sentence again each round. The keys and values above are the ones it works out anyway, live on your sentence. The real sizes are counts worked out from the published settings; the bytes depend on the number format you run the model in. ${modelId === "llama-3-8b" ? "Llama 3 8B's numbers are published at 16 bits, and its code keeps the cache in the model's own format (\"self.cache_k = self.cache_k.to(xq)\"). Its 8 key-value groups, shared by 4 query heads each, are why it saves a quarter of what one per head would." : "GPT-2 gives every one of its 12 heads its own key and value, so it saves 768 + 768 numbers per word in each layer."}`}</HonestNote>
    </div>
  );
}

import { useId, useState } from "react";
import { joinTokens } from "@/core/model/corpus";
import { useGpt2Vocab } from "@/features/take-apart/hooks/use-gpt-2-vocab";
import type { DemoProps } from "@/features/take-apart/interfaces/model-part.interface";
import { BeforeAfter, HonestNote } from "@/features/take-apart/story/watch-parts";
import { SOURCES } from "@/features/take-apart/trees/sources";
import { count } from "@/shared/utils/format";
import { BASE_PIECES, type EncodedWord, type Gpt2Vocab, MAX_TOKENIZE_LENGTH, type Token, encodeText } from "./gpt-2-bpe";
import { GptTwoNote, PieceChips } from "./tokenizer-parts";

const START_TEXT = "Lightning is a giant spark of electricity.";
const USE_PARTS = new Set(["tokenize", "split", "apply-merge", "lookup"]);
const LOOKUP_PARTS = new Set(["assign-id", "lookup"]);

function IdLookup({ token, vocab }: { token: Token; vocab: Gpt2Vocab }) {
  const near = [token.id - 2, token.id - 1, token.id, token.id + 1, token.id + 2].filter((id) => id >= 0 && id < vocab.pieces.length);
  const from = token.id < BASE_PIECES.length ? `one of the ${BASE_PIECES.length} single bytes the list starts with` : `merge rule ${count(token.id - BASE_PIECES.length + 1)} in vocab.bpe, plus the ${BASE_PIECES.length} bytes before it`;
  return (
    <div className="watch-box">
      <p className="section-title">{`"${token.piece}" gets ID ${count(token.id)}: its place in the list`}</p>
      <p className="note">{`It is at that place because it is ${from}.`}</p>
      <ol className="piece-list" aria-label="Its neighbours in the list">
        {near.map((id) => (
          <li key={id} className={`piece-chip${id === token.id ? " is-picked" : ""}`}>
            {vocab.pieces[id]}
            <b>{count(id)}</b>
          </li>
        ))}
      </ol>
      <p className="card-lesson">Neighbours in the list are not related in meaning. An ID is only a place number.</p>
    </div>
  );
}

function OneMerge({ word, vocab, index, onIndex }: { word: EncodedWord; vocab: Gpt2Vocab; index: number; onIndex: (index: number) => void }) {
  const rounds = word.steps.length - 1;
  if (rounds === 0) return <p className="note">{`"${word.text.replace(/ /g, "Ġ")}" is already one piece of the list after its bytes, so no rule is needed.`}</p>;
  const i = Math.min(Math.max(index, 1), rounds);
  const { rule, rank, pieces } = word.steps[i];
  const toChips = (list: string[]) => list.map((piece) => ({ piece, id: vocab.ids.get(piece) ?? -1 }));
  return (
    <div className="watch-box">
      <div className="watch-controls">
        <button type="button" className="action" onClick={() => onIndex(i - 1)} disabled={i === 1}>
          Back
        </button>
        <button type="button" className="action action--strong" onClick={() => onIndex(i + 1)} disabled={i === rounds}>
          Next rule
        </button>
        <span className="card-lesson">{`Join ${i} of ${rounds}`}</span>
      </div>
      <BeforeAfter
        beforeLabel={rule ? `Rule ${count((rank ?? 0) + 1)} of ${count(vocab.merges.length)}: "${rule[0]}" + "${rule[1]}"` : "Bytes"}
        before={<PieceChips tokens={toChips(word.steps[i - 1].pieces)} />}
        afterLabel="Those two side by side are joined"
        after={<PieceChips tokens={toChips(pieces)} />}
      />
      <p className="card-lesson">Of all the pairs side by side, the one whose rule was learned first is joined first. It stops when no pair has a rule.</p>
    </div>
  );
}

export function Gpt2TokenizerDemo({ part, modelId, tiny }: DemoProps) {
  const { vocab, problem } = useGpt2Vocab();
  const [text, setText] = useState(() => (USE_PARTS.has(part.id) ? joinTokens(tiny.words) : START_TEXT));
  const [picked, setPicked] = useState(0);
  const [mergeIndex, setMergeIndex] = useState(1);
  const inputId = useId();
  if (problem) return <p className="watch-problem" role="alert">{problem}</p>;
  if (!vocab) return <p className="note">Loading GPT-2's 50,000 merge rules…</p>;
  const words = encodeText(text, vocab);
  const tokens = words.flatMap((word) => word.tokens);
  const word = words[Math.min(picked, words.length - 1)];

  return (
    <div className="watch">
      {modelId === "llama-3-8b" && (
        <HonestNote>
          This is GPT-2's tokenizer, not Llama 3's. Llama 3's tokenizer file is only given out after you accept Meta's licence (
          <a href={SOURCES["llama-3-readme"].url} target="_blank" rel="noreferrer">
            Llama 3 README
          </a>
          ), so this app does not ship it. It also works by joining byte pairs in a learned order (it is built on tiktoken), but its list has 128,256 words, so its pieces and IDs differ.
        </HonestNote>
      )}
      <label className="section-title" htmlFor={inputId}>
        Type any text
      </label>
      <input id={inputId} className="watch-input" value={text} maxLength={MAX_TOKENIZE_LENGTH} onChange={(event) => setText(event.target.value)} />
      <BeforeAfter
        beforeLabel={`Your text: ${count(new TextEncoder().encode(text).length)} bytes`}
        before={<p className="watch-text">{text || "Nothing yet."}</p>}
        afterLabel={`${count(tokens.length)} pieces, each with its real GPT-2 ID`}
        after={<PieceChips tokens={tokens} />}
      />
      {LOOKUP_PARTS.has(part.id) && word && <IdLookup token={word.tokens[word.tokens.length - 1]} vocab={vocab} />}
      {word && (
        <div className="watch-box">
          <p className="section-title">Pick a word to watch its merges</p>
          <div className="word-picker">
            {words.map((each, i) => (
              <button key={i} type="button" className={`word-chip${each === word ? " is-picked" : ""}`} aria-pressed={each === word} onClick={() => { setPicked(i); setMergeIndex(1); }}>
                {each.text.replace(/ /g, "Ġ")}
              </button>
            ))}
          </div>
          {part.id === "apply-merge" && <OneMerge word={word} vocab={vocab} index={mergeIndex} onIndex={setMergeIndex} />}
          <ol className="merge-steps">
            {word.steps.map((each, i) => (
              <li key={i}>
                <span className="card-lesson">{each.rule ? `rule ${count((each.rank ?? 0) + 1)}: ${each.rule[0]} + ${each.rule[1]}` : "bytes"}</span>
                <PieceChips tokens={each.pieces.map((piece) => ({ piece, id: vocab.ids.get(piece) ?? -1 }))} />
              </li>
            ))}
          </ol>
        </div>
      )}
      {part.id === "tokenize" && (
        <div className="watch-box">
          <p className="section-title">{`Our tiny model's own split of "${joinTokens(tiny.words)}"`}</p>
          <PieceChips tokens={tiny.words.map((each, i) => ({ piece: each, id: tiny.ids[i] }))} />
          <p className="card-lesson">It has no merge rules: it splits on spaces and looks each word up in its own list. These are the IDs the next steps use.</p>
        </div>
      )}
      <GptTwoNote />
    </div>
  );
}

import { useId, useState } from "react";
import { useGpt2Vocab } from "@/features/take-apart/hooks/use-gpt-2-vocab";
import type { DemoProps } from "@/features/take-apart/interfaces/model-part.interface";
import { BeforeAfter, HonestNote } from "@/features/take-apart/story/watch-parts";
import { SOURCES } from "@/features/take-apart/trees/sources";
import { count } from "@/shared/utils/format";
import { BASE_PIECES, type Gpt2Vocab, MAX_TOKENIZE_LENGTH, type Token, encodeText } from "./gpt-2-bpe";
import { GptTwoNote, PieceChips } from "./tokenizer-parts";

const START_TEXT = "Lightning is a giant spark of electricity.";

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

export function Gpt2TokenizerDemo({ part, modelId }: DemoProps) {
  const { vocab, problem } = useGpt2Vocab();
  const [text, setText] = useState(START_TEXT);
  const [picked, setPicked] = useState(0);
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
      {part.id === "assign-id" && word && <IdLookup token={word.tokens[word.tokens.length - 1]} vocab={vocab} />}
      {word && (
        <div className="watch-box">
          <p className="section-title">Pick a word to watch its merges</p>
          <div className="word-picker">
            {words.map((each, i) => (
              <button key={i} type="button" className={`word-chip${each === word ? " is-picked" : ""}`} aria-pressed={each === word} onClick={() => setPicked(i)}>
                {each.text.replace(/ /g, "Ġ")}
              </button>
            ))}
          </div>
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
      <GptTwoNote />
    </div>
  );
}

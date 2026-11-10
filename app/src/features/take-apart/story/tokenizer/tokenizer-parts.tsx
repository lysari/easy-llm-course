import "./tokenizer.css";
import { HonestNote } from "@/features/take-apart/story/watch-parts";
import { SOURCES } from "@/features/take-apart/trees/sources";
import { count } from "@/shared/utils/format";
import type { Token } from "./gpt-2-bpe";

const GPT_2_LICENCE_URL = "https://github.com/openai/gpt-2/blob/master/LICENSE";

export function PieceChips({ tokens }: { tokens: Token[] }) {
  return (
    <ol className="piece-list">
      {tokens.map((token, i) => (
        <li key={i} className="piece-chip">
          {token.piece}
          {token.id >= 0 && <b>{count(token.id)}</b>}
        </li>
      ))}
    </ol>
  );
}

export function GptTwoNote() {
  return (
    <HonestNote>
      Run live with GPT-2's real merge rules,{" "}
      <a href={SOURCES["gpt-2-merges"].url} target="_blank" rel="noreferrer">
        vocab.bpe
      </a>
      , shipped in this app under OpenAI's{" "}
      <a href={GPT_2_LICENCE_URL} target="_blank" rel="noreferrer">
        Modified MIT License
      </a>
      , and its split rule from{" "}
      <a href={SOURCES["gpt-2-encoder"].url} target="_blank" rel="noreferrer">
        encoder.py
      </a>
      . Ġ marks a space, and a letter outside plain English shows as its bytes, so é shows as Ã©. IDs: the 256 single bytes come first, then one ID per merge rule in order, so rule 1 makes ID 256; checked against OpenAI's encoder.json.
    </HonestNote>
  );
}

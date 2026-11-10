import { useId, useState } from "react";
import { joinTokens } from "@/core/model/corpus";
import type { DemoProps } from "@/features/take-apart/interfaces/model-part.interface";
import { BeforeAfter, HonestNote } from "@/features/take-apart/story/watch-parts";
import { SOURCES } from "@/features/take-apart/trees/sources";
import { count, showWord } from "@/shared/utils/format";
import { type ChatMessage, LLAMA_3_BASE_WORDS, chatPieces } from "./post-training-math";

function BaseModelView({ tiny }: Pick<DemoProps, "tiny">) {
  const written = tiny.written.map((each) => showWord(each.word));
  return (
    <div className="watch">
      <HonestNote>
        GPT-2 had no post-training. Its paper reports every result from the pre-trained model alone: "No training or fine-tuning was performed for any of these results" (
        <a href={SOURCES["gpt-2-paper"].url} target="_blank" rel="noreferrer">
          GPT-2 paper, Table 3
        </a>
        ). So there is nothing of GPT-2's to run here.
      </HonestNote>
      <p className="note">What a model without post-training does instead: it only continues text. Watch our tiny model, which is also only pre-trained, carry on from your sentence (change it in the bar at the top of the page). It never answers; it writes what usually comes next.</p>
      <BeforeAfter
        beforeLabel="Your sentence"
        before={<p className="watch-text">{joinTokens(tiny.words)}</p>}
        afterLabel="What the tiny model writes next, live"
        after={<p className="watch-text">{written.length > 0 ? `… ${written.join(" ")}` : "It stops right away."}</p>}
      />
    </div>
  );
}

export function ChatFormatDemo({ modelId, tiny }: DemoProps) {
  const [system, setSystem] = useState("");
  const [user, setUser] = useState(joinTokens(tiny.words));
  const systemId = useId();
  const userId = useId();
  if (modelId === "gpt-2-small") return <BaseModelView tiny={tiny} />;
  const messages: ChatMessage[] = [...(system.trim() ? [{ role: "system" as const, content: system }] : []), { role: "user", content: user }];
  const pieces = chatPieces(messages);

  return (
    <div className="watch">
      <p className="note">Post-training teaches Llama 3 Instruct to read a conversation in one fixed layout and to write its answer after the last header. Type a message and watch the exact layout its code builds.</p>
      <div className="watch-grid">
        <div className="watch-box">
          <label className="section-title" htmlFor={systemId}>
            System message (optional)
          </label>
          <input id={systemId} className="watch-input" value={system} placeholder="You are a helpful assistant." onChange={(event) => setSystem(event.target.value)} />
          <label className="section-title" htmlFor={userId}>
            Your message
          </label>
          <textarea id={userId} className="watch-input" rows={3} value={user} onChange={(event) => setUser(event.target.value)} />
        </div>
        <div className="watch-box">
          <p className="section-title">{`What Llama 3 Instruct reads: ${count(pieces.length)} parts`}</p>
          <ol className="piece-list">
            {pieces.map((piece, i) => (
              <li key={i} className={`piece-chip${piece.id === null ? "" : " is-picked"}`}>
                {piece.text === "\n\n" ? "↵↵" : piece.text || "(empty)"}
                {piece.id !== null && <b>{count(piece.id)}</b>}
              </li>
            ))}
          </ol>
          <p className="card-lesson">The model's answer is written after the last part, and it ends its turn with &lt;|eot_id|&gt;.</p>
        </div>
      </div>
      <HonestNote>
        Built live with the same steps as ChatFormat in Meta's{" "}
        <a href={SOURCES["llama-3-tokenizer"].url} target="_blank" rel="noreferrer">
          tokenizer.py
        </a>
        . The marker IDs are worked out from it: {count(LLAMA_3_BASE_WORDS)} regular words come first, then the special words in the file's order. The IDs of your own words need Llama 3's tokenizer file, which is only given out after accepting Meta's licence, so they are not shown. Meta's own training conversations are not published.
      </HonestNote>
    </div>
  );
}

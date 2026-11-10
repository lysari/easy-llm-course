import { useId, useState } from "react";
import { END_TOKEN, START_CHOICES, VOCABULARY, joinTokens } from "@/core/model/corpus";
import { FactRows } from "@/features/take-apart/fact-rows";
import type { DemoProps } from "@/features/take-apart/interfaces/model-part.interface";
import { BeforeAfter, HonestNote } from "@/features/take-apart/story/watch-parts";
import { count, showWord } from "@/shared/utils/format";
import { SAFETY_TRIES, breaksRule, redTeam, writeOnce } from "./evaluation-math";

const PROMPTS = START_CHOICES.map((choice) => choice.split(" "));
const WORDS = VOCABULARY.filter((word) => word !== END_TOKEN && /^\p{L}+$/u.test(word));

export function SafetyDemo({ part, modelId, tiny }: DemoProps) {
  const [banned, setBanned] = useState("stick");
  const [prompt, setPrompt] = useState(0);
  const [seed, setSeed] = useState(1);
  const bannedId = useId();
  const promptId = useId();
  const written = writeOnce(PROMPTS[prompt], tiny.weights, seed);
  const fails = breaksRule(written, banned);
  const results = part.id === "safety" ? redTeam(PROMPTS, banned, tiny.weights) : [];
  const failCount = results.reduce((sum, result) => sum + result.fails, 0);

  return (
    <div className="watch">
      <p className="note">A stand-in safety rule for a model that only knows pets: it must never write the word you pick. Then try prompts on it and count how often it breaks the rule.</p>
      <div className="watch-controls">
        <div className="watch-slider">
          <label htmlFor={bannedId}>Never write</label>
          <select id={bannedId} className="watch-select" value={banned} onChange={(event) => setBanned(event.target.value)}>
            {WORDS.map((word) => (
              <option key={word} value={word}>
                {word}
              </option>
            ))}
          </select>
        </div>
        {part.id === "safety-check" && (
          <div className="watch-slider">
            <label htmlFor={promptId}>Prompt</label>
            <select id={promptId} className="watch-select" value={prompt} onChange={(event) => setPrompt(Number(event.target.value))}>
              {START_CHOICES.map((choice, i) => (
                <option key={choice} value={i}>
                  {choice} …
                </option>
              ))}
            </select>
            <button type="button" className="action" onClick={() => setSeed(seed + 1)}>
              Try another roll
            </button>
          </div>
        )}
      </div>
      {part.id === "safety-check" ? (
        <BeforeAfter
          beforeLabel={`Prompt, roll ${seed}`}
          before={<p className="watch-text">{joinTokens(PROMPTS[prompt])} …</p>}
          afterLabel="The answer, checked"
          after={
            <>
              <p className="watch-text">{written.length > 0 ? written.map(showWord).join(" ") : "(it stopped right away)"}</p>
              <span className={`verdict ${fails ? "is-drop" : "is-keep"}`}>{fails ? `Fail: it wrote "${banned}"` : "Pass"}</span>
            </>
          }
        />
      ) : (
        <BeforeAfter
          beforeLabel={`${count(PROMPTS.length)} prompts × ${SAFETY_TRIES} rolls each`}
          before={<FactRows rows={PROMPTS.map((each): [string, string] => [joinTokens(each), `${SAFETY_TRIES} tries`])} />}
          afterLabel={`${count(failCount)} of ${count(PROMPTS.length * SAFETY_TRIES)} answers broke the rule`}
          after={<FactRows rows={results.map((result): [string, string] => [joinTokens(result.prompt), `${result.fails} failed`])} />}
        />
      )}
      <HonestNote>{`The rule is hand-made for this demo, and the answers are written live by our tiny model with seeded random rolls. ${modelId === "gpt-2-small" ? "No safety tests for GPT-2 are published in its paper or model card; the card advises studying biases first." : "Meta red-teamed Llama 3 Instruct with experts and automated attack prompts, which are not published, and offers Llama Guard 2, a separate model that filters prompts and answers, which cannot run in this page."}`}</HonestNote>
    </div>
  );
}

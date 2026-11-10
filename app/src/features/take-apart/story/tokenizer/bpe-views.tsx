import { useMemo, useState } from "react";
import { FactRows } from "@/features/take-apart/fact-rows";
import { useGpt2Vocab } from "@/features/take-apart/hooks/use-gpt-2-vocab";
import { BeforeAfter } from "@/features/take-apart/story/watch-parts";
import { count } from "@/shared/utils/format";
import { type MergeRound, type TrainingWord, countFirstPairs, countPairs, pairPlaces, pieceCount } from "./bpe-train";
import { BASE_PIECES, mergePair, splitWords, toBytePieces } from "./gpt-2-bpe";

const SAMPLE_WORDS = 14;
const SHOWN_COUNTS = 8;
const SHOWN_RULES = 12;

interface LearnRun {
  learned: MergeRound[];
  words: TrainingWord[];
}

const sampleRows = (text: string, rules: MergeRound[]) => splitWords(text).slice(0, SAMPLE_WORDS).map((word) => rules.reduce((pieces, { rule }) => mergePair(pieces, rule), toBytePieces(word)));

function PieceRows({ rows, mark }: { rows: string[][]; mark?: string }) {
  return (
    <ol className="piece-list">
      {rows.flatMap((pieces, w) => pieces.map((piece, i) => (
        <li key={`${w}-${i}`} className={`piece-chip${piece === mark ? " is-picked" : ""}`}>
          {piece}
        </li>
      )))}
    </ol>
  );
}

export function CountPairView({ text, words }: { text: string; words: TrainingWord[] }) {
  const rows = useMemo(() => sampleRows(text, []), [text]);
  const places = useMemo(() => pairPlaces(rows), [rows]);
  const [counted, setCounted] = useState(0);
  const done = Math.min(counted, places.length);
  const counts = countFirstPairs(rows, places.slice(0, done)).slice(0, SHOWN_COUNTS);
  const current = places[done - 1];
  const pair = current ? rows[current.word].slice(current.at, current.at + 2) : null;
  const whole = countPairs(words)[0];

  return (
    <>
      <div className="watch-controls">
        <button type="button" className="action action--strong" onClick={() => setCounted(done + 1)} disabled={done >= places.length}>
          Count the next pair
        </button>
        <button type="button" className="action" onClick={() => setCounted(places.length)}>
          Count them all
        </button>
        <button type="button" className="action" onClick={() => setCounted(0)}>
          Start again
        </button>
        <span className="card-lesson">{`${count(done)} of ${count(places.length)} pairs counted in the first ${SAMPLE_WORDS} words`}</span>
      </div>
      <BeforeAfter
        beforeLabel={pair ? `Pair ${count(done)}: ${pair.join(" + ")}` : "The text as single bytes"}
        before={
          <ol className="piece-list">
            {rows.flatMap((pieces, w) => pieces.map((piece, i) => (
              <li key={`${w}-${i}`} className={`piece-chip${current && current.word === w && (i === current.at || i === current.at + 1) ? " is-picked" : ""}`}>
                {piece}
              </li>
            )))}
          </ol>
        }
        afterLabel="Counts so far, most common first"
        after={counts.length > 0 ? <FactRows rows={counts.map((each): [string, string] => [each.pair.join(" + "), `${count(each.count)}`])} /> : <p className="note">Nothing counted yet.</p>}
      />
      {whole && <p className="card-lesson">{`Across the whole text, the most common pair is ${whole.pair.join(" + ")}, seen ${count(whole.count)} times. That pair becomes the first merge rule.`}</p>}
    </>
  );
}

interface RoundProps {
  text: string;
  words: TrainingWord[];
  run: LearnRun;
  rounds: number;
  onRounds: (rounds: number) => void;
}

export function MergeRoundView({ text, words, run, rounds, onRounds }: RoundProps) {
  const last = run.learned[run.learned.length - 1];
  const before = sampleRows(text, run.learned.slice(0, -1));
  const after = sampleRows(text, run.learned);
  const piecesBefore = run.learned.length > 1 ? run.learned[run.learned.length - 2].pieceCount : pieceCount(words);
  return (
    <>
      <div className="watch-controls">
        <button type="button" className="action action--strong" onClick={() => onRounds(rounds + 1)}>
          Run one merge round
        </button>
        <button type="button" className="action" onClick={() => onRounds(rounds + 10)}>
          Run 10 rounds
        </button>
        <button type="button" className="action" onClick={() => onRounds(0)} disabled={rounds === 0}>
          Start again
        </button>
        <span className="card-lesson">{`Round ${count(run.learned.length)}`}</span>
      </div>
      {last ? (
        <BeforeAfter
          beforeLabel={`Before round ${run.learned.length}: ${count(piecesBefore)} pieces in the text`}
          before={<PieceRows rows={before} />}
          afterLabel={`Joined ${last.rule.join(" + ")} (seen ${count(last.count)} times): ${count(last.pieceCount)} pieces`}
          after={<PieceRows rows={after} mark={last.rule.join("")} />}
        />
      ) : (
        <p className="note">{`The text starts as ${count(pieceCount(words))} single bytes. Run a round to join the most common pair.`}</p>
      )}
    </>
  );
}

export function MergeRulesView({ run, rounds, onRounds }: Omit<RoundProps, "text" | "words">) {
  const { vocab } = useGpt2Vocab();
  const shown = run.learned.slice(-SHOWN_RULES);
  const offset = run.learned.length - shown.length;
  return (
    <>
      <div className="watch-controls">
        <button type="button" className="action action--strong" onClick={() => onRounds(rounds + 1)}>
          Learn one more rule
        </button>
        <button type="button" className="action" onClick={() => onRounds(0)} disabled={rounds === 0}>
          Start again
        </button>
      </div>
      <div className="watch-grid">
        <div className="watch-box">
          <p className="section-title">{`Rules learned from this text: ${count(run.learned.length)}`}</p>
          {shown.length > 0 ? (
            <FactRows rows={shown.map((each, i): [string, string] => [`Rule ${count(offset + i + 1)}: ${each.rule.join(" + ")} → ${each.rule.join("")}`, `new ID ${count(BASE_PIECES.length + offset + i)}`])} />
          ) : (
            <p className="note">None yet. Each rule you learn joins one pair and adds one piece to the list.</p>
          )}
        </div>
        <div className="watch-box">
          <p className="section-title">GPT-2's real first rules, from vocab.bpe</p>
          {vocab ? <FactRows rows={vocab.merges.slice(0, SHOWN_RULES).map((rule, i): [string, string] => [`Rule ${i + 1}: ${rule.join(" + ")} → ${rule.join("")}`, `ID ${count(BASE_PIECES.length + i)}`])} /> : <p className="note">Loading…</p>}
        </div>
      </div>
    </>
  );
}

import type { PageBox, ReadabilityTrace } from "@/features/inside-real-models/interfaces/readability-trace.interface";
import { count } from "@/shared/utils/format";
import { boxLabel, showPoints } from "./how-it-works-steps";

const TOP_COUNT = 5;
const SHARES = ["all", "half", "a sixth", "a ninth", "a twelfth"];
const PLACES = ["its parent", "its grandparent", "3 levels up", "4 levels up", "5 levels up"];

const signed = (value: number) => (value < 0 ? `−${showPoints(-value)}` : `+${showPoints(value)}`);

function startParts(box: PageBox) {
  if (!box.score) return "";
  const tagPoints = box.score.start - box.score.namePoints;
  const parts = box.score.nameWords.map((word) => `“${word.word}” ${signed(word.points)}`);
  return [tagPoints ? `${box.tag} ${signed(tagPoints)}` : `${box.tag} 0`, ...parts].join(", ");
}

function ParagraphExample({ trace, paragraph }: { trace: ReadabilityTrace; paragraph: PageBox }) {
  const points = paragraph.paragraph;
  if (!points) return null;
  const handedUp = trace.boxes.flatMap((box) => (box.score?.received ?? []).filter((part) => part.fromId === paragraph.id).map((part) => ({ box, part }))).sort((a, b) => a.part.level - b.part.level);
  const commas = points.commaPoints - 1;
  return (
    <div className="watch-box">
      <p className="section-title">One paragraph's points</p>
      <p className="note">
        <span className="box-name">{boxLabel(paragraph)}</span> “{paragraph.preview}…” <span className="card-lesson">({count(points.textLength)} characters)</span>
      </p>
      <ul className="points-sum">
        <li>
          <b>+1</b> <span>to start</span>
        </li>
        <li>
          <b>+{points.commaPoints}</b> <span>{`for pieces between commas (${count(commas)} comma${commas === 1 ? "" : "s"}, plus 1)`}</span>
        </li>
        <li>
          <b>+{points.lengthPoints}</b> <span>for length: 1 per 100 characters, at most 3</span>
        </li>
        <li className="points-total">
          <b>= {points.total}</b> <span>points</span>
        </li>
      </ul>
      <p className="section-title">It hands them up to the boxes around it</p>
      <ol className="points-up">
        {handedUp.map(({ box, part }) => (
          <li key={box.id}>
            <b>{signed(part.points)}</b> <span className="box-name">{boxLabel(box)}</span> <span>{`(${PLACES[part.level]}: ${SHARES[part.level]})`}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function ScoreTable({ trace }: { trace: ReadabilityTrace }) {
  const top = trace.boxes.filter((box) => box.score).sort((a, b) => (b.score?.score ?? 0) - (a.score?.score ?? 0) || a.id - b.id).slice(0, TOP_COUNT);
  return (
    <div className="score-wrap" tabIndex={0} role="region" aria-label="How the top scores add up">
      <table className="score-table">
        <caption className="section-title">{`How the ${top.length} highest scores add up`}</caption>
        <thead>
          <tr>
            <th scope="col">Box</th>
            <th scope="col">Start</th>
            <th scope="col">+ Paragraphs</th>
            <th scope="col">= Sum</th>
            <th scope="col">× Not links</th>
            <th scope="col">= Score</th>
          </tr>
        </thead>
        <tbody>
          {top.map((box) => {
            const score = box.score;
            if (!score) return null;
            const fromParagraphs = score.beforeLinks - score.start;
            const senders = new Set(score.received.map((part) => part.fromId)).size;
            return (
              <tr key={box.id} className={box.id === trace.winnerId ? "is-winner" : undefined}>
                <th scope="row" className="box-name">
                  {box.id === trace.winnerId && <span aria-hidden="true">★ </span>}
                  {boxLabel(box)}
                </th>
                <td>
                  <b>{showPoints(score.start)}</b>
                  <span>{startParts(box)}</span>
                </td>
                <td>
                  <b>{signed(fromParagraphs)}</b>
                  <span>{`from ${count(senders)}`}</span>
                </td>
                <td>
                  <b>{showPoints(score.beforeLinks)}</b>
                </td>
                <td>
                  {score.linkShare === null ? (
                    <span>not applied</span>
                  ) : (
                    <>
                      <b>{`× ${(1 - score.linkShare).toFixed(3)}`}</b>
                      <span>{`${(score.linkShare * 100).toFixed(1)}% links`}</span>
                    </>
                  )}
                </td>
                <td>
                  <b>{showPoints(score.score)}</b>
                  {box.id === trace.winnerId && <span>winner</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function ScoreDetails({ trace }: { trace: ReadabilityTrace }) {
  const winner = trace.boxes[trace.winnerId];
  const firstSender = winner.score?.received.find((part) => part.level === 0)?.fromId;
  const paragraph = trace.boxes[firstSender ?? -1] ?? trace.boxes.find((box) => box.paragraph && !box.paragraph.isTooShort);
  return (
    <>
      {paragraph && <ParagraphExample trace={trace} paragraph={paragraph} />}
      <ScoreTable trace={trace} />
    </>
  );
}

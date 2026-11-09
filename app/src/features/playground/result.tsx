import { joinTokens } from "@/core/model/corpus";
import { applySampling, nextOptions, writeText } from "@/core/model/generate";
import { seededRandom } from "@/core/model/random";
import type { PlaygroundState } from "@/features/playground/interfaces/playground.interface";
import { ChanceBars } from "@/shared/components/chance-bars";
import { LossPlot } from "@/shared/components/loss-plot";
import { count, showWord } from "@/shared/utils/format";
import { GUESSING_LOSS, type PlaygroundRun } from "./playground-run";

const PLOT_SIZE = { width: 400, height: 220 };
const PLOT_CEILING = GUESSING_LOSS * 3;

interface ResultProps {
  state: PlaygroundState;
  onTrain: () => void;
  onWriteAgain: () => void;
  onReset: () => void;
}

function StatusPill({ state: { run, isStale, isAutoTrain } }: { state: PlaygroundState }) {
  if (run?.isBroken) return <span className="pill pill--needs">Numbers blew up</span>;
  if (isStale && !isAutoTrain) return <span className="pill pill--needs">Settings changed</span>;
  if (!run) return <span className="pill pill--queued">• Not trained</span>;
  if (run.isDone) return <span className="pill pill--done">✓ Trained</span>;
  return <span className="pill pill--running"><i className="spinner" /> Training</span>;
}

function resultNote(activeRun: PlaygroundRun) {
  const loss = activeRun.latestLoss ?? GUESSING_LOSS;
  if (activeRun.isBroken) return "The numbers grew too big to count. Try a smaller step size.";
  if (loss >= GUESSING_LOSS) return "Worse than random guessing, so it has learned nothing useful yet.";
  return `Loss went from ${activeRun.history[0].loss.toFixed(2)} to ${loss.toFixed(2)}. Lower means it matches the text better.`;
}

export function Result({ state, onTrain, onWriteAgain, onReset }: ResultProps) {
  const { settings, run, isStale, earlier, earlierLoss, writeSeed } = state;
  const rounds = Math.max(settings.rounds, earlier.length);
  const history = (run?.history ?? []).filter((point) => Number.isFinite(point.loss)).map((point) => ({ ...point, loss: Math.min(point.loss, PLOT_CEILING) }));
  const canPredict = run !== null && run.history.length > 0 && !run.isBroken;
  const words = settings.start.split(" ");
  const ranked = canPredict ? applySampling(nextOptions(words, run.weights), settings) : [];
  const written = canPredict && run.isDone ? joinTokens(writeText(words, run.weights, seededRandom(writeSeed), settings)) : null;
  const lastWord = words[words.length - 1];
  return (
    <aside className="pg-result" aria-label="Result">
      <header className="pg-result-head">
        <h2>Result</h2>
        <StatusPill state={state} />
      </header>
      <dl className="pg-stats">
        <div><dt>Loss</dt><dd>{run?.latestLoss != null && Number.isFinite(run.latestLoss) ? run.latestLoss.toFixed(2) : "–"}</dd></div>
        <div><dt>Last run</dt><dd>{earlierLoss === null ? "–" : earlierLoss.toFixed(2)}</dd></div>
        <div><dt>Round</dt><dd>{`${run ? run.history.length : 0}/${settings.rounds}`}</dd></div>
        <div><dt>Numbers</dt><dd>{run ? count(run.numberCount) : "–"}</dd></div>
      </dl>
      <LossPlot points={history} rounds={rounds} guessingLoss={GUESSING_LOSS} earlier={earlier} {...PLOT_SIZE} />
      {earlier.length > 0 && (
        <p className="pg-legend">
          <span className="pg-key pg-key--now">This run</span>
          <span className="pg-key pg-key--earlier">Last run</span>
        </p>
      )}
      {run && <p className="note">{resultNote(run)}</p>}
      <section className="section">
        <p className="section-title">{`Word after “${showWord(lastWord)}”`}</p>
        {ranked.length > 0 ? <ChanceBars options={ranked} answer={ranked[0].token} /> : <p className="note">Train the model first.</p>}
      </section>
      <section className="section">
        <p className="section-title">It writes</p>
        <p className="quote">{written || "Waiting for training to finish."}</p>
      </section>
      <footer className="pg-actions">
        <button type="button" className="action action--strong" disabled={!isStale} onClick={onTrain}>Train</button>
        <button type="button" className="action" disabled={!written} onClick={onWriteAgain}>Write again</button>
        <button type="button" className="action" onClick={onReset}>Reset all</button>
      </footer>
    </aside>
  );
}

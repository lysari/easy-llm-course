import { joinTokens } from "@/core/model/corpus";
import { applySampling, nextOptions, writeText } from "@/core/model/generate";
import { seededRandom } from "@/core/model/random";
import type { PlaygroundState } from "@/features/playground/interfaces/playground.interface";
import { chanceBars, lossPlot } from "@/shared/utils/charts";
import { count, showWord } from "@/shared/utils/format";
import { escapeHtml } from "@/shared/utils/html";
import { GUESSING_LOSS, type PlaygroundRun } from "./playground-run";

const PLOT_SIZE = { width: 400, height: 220 };
const PLOT_CEILING = GUESSING_LOSS * 3;

function statusPill({ run, isStale, isAutoTrain }: PlaygroundState) {
  if (run?.isBroken) return `<span class="pill pill--needs">Numbers blew up</span>`;
  if (isStale && !isAutoTrain) return `<span class="pill pill--needs">Settings changed</span>`;
  if (!run) return `<span class="pill pill--queued">• Not trained</span>`;
  return run.isDone ? `<span class="pill pill--done">✓ Trained</span>` : `<span class="pill pill--running"><i class="spinner"></i> Training</span>`;
}

function resultNote(activeRun: PlaygroundRun) {
  const loss = activeRun.latestLoss ?? GUESSING_LOSS;
  if (activeRun.isBroken) return "The numbers grew too big to count. Try a smaller step size.";
  if (loss >= GUESSING_LOSS) return "Worse than random guessing, so it has learned nothing useful yet.";
  return `Loss went from ${activeRun.history[0].loss.toFixed(2)} to ${loss.toFixed(2)}. Lower means it matches the text better.`;
}

export function resultHtml(state: PlaygroundState) {
  const { settings, run, isStale, earlier, earlierLoss, writeSeed } = state;
  const rounds = Math.max(settings.rounds, earlier.length);
  const history = (run?.history ?? []).filter((point) => Number.isFinite(point.loss)).map((point) => ({ ...point, loss: Math.min(point.loss, PLOT_CEILING) }));
  const canPredict = run && run.history.length > 0 && !run.isBroken;
  const words = settings.start.split(" ");
  const ranked = canPredict ? applySampling(nextOptions(words, run!.weights), settings) : [];
  const written = canPredict && run!.isDone ? joinTokens(writeText(words, run!.weights, seededRandom(writeSeed), settings)) : null;
  const lastWord = words[words.length - 1];
  return `<header class="pg-result-head"><h2>Result</h2>${statusPill(state)}</header>
      <dl class="pg-stats">
        <div><dt>Loss</dt><dd>${run?.latestLoss != null && Number.isFinite(run.latestLoss) ? run.latestLoss.toFixed(2) : "–"}</dd></div>
        <div><dt>Last run</dt><dd>${earlierLoss === null ? "–" : earlierLoss.toFixed(2)}</dd></div>
        <div><dt>Round</dt><dd>${run ? run.history.length : 0}/${settings.rounds}</dd></div>
        <div><dt>Numbers</dt><dd>${run ? count(run.numberCount) : "–"}</dd></div>
      </dl>
      ${lossPlot(history, rounds, GUESSING_LOSS, { earlier, ...PLOT_SIZE })}
      ${earlier.length > 0 ? `<p class="pg-legend"><span class="pg-key pg-key--now">This run</span><span class="pg-key pg-key--earlier">Last run</span></p>` : ""}
      ${run ? `<p class="note">${resultNote(run)}</p>` : ""}
      <section class="section"><p class="section-title">Word after “${escapeHtml(showWord(lastWord))}”</p>${ranked.length > 0 ? chanceBars(ranked, ranked[0].token) : `<p class="note">Train the model first.</p>`}</section>
      <section class="section"><p class="section-title">It writes</p><p class="quote">${written ? escapeHtml(written) : "Waiting for training to finish."}</p></section>
      <footer class="pg-actions">
        <button type="button" class="action action--strong" data-do="train"${isStale ? "" : " disabled"}>Train</button>
        <button type="button" class="action" data-do="write-again"${written ? "" : " disabled"}>Write again</button>
        <button type="button" class="action" data-do="reset">Reset all</button>
      </footer>`;
}

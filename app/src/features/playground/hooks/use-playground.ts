import { useEffect, useState } from "react";
import { START_CHOICES } from "@/core/model/corpus";
import type { PlaygroundSettings, PlaygroundState, RowKey } from "@/features/playground/interfaces/playground.interface";
import { PlaygroundRun, defaultSettings } from "@/features/playground/playground-run";
import { useTrainingLoop } from "./use-training-loop";

const START_WORDS = START_CHOICES[0];

const startingState = (): PlaygroundState => ({
  settings: defaultSettings(START_WORDS),
  addedRows: [],
  isAutoTrain: true,
  run: null,
  isStale: true,
  earlier: [],
  earlierLoss: null,
  writeSeed: 1,
});

function withNewRun(state: PlaygroundState): PlaygroundState {
  const { run } = state;
  const isKept = run !== null && run.isDone && !run.isBroken;
  return {
    ...state,
    run: new PlaygroundRun(state.settings),
    isStale: false,
    earlier: isKept ? run.history.map((point) => ({ ...point })) : state.earlier,
    earlierLoss: isKept ? run.latestLoss : state.earlierLoss,
  };
}

export function usePlayground(isOpen: boolean) {
  const [state, setState] = useState(startingState);
  useTrainingLoop(state.run);

  useEffect(() => {
    if (isOpen && !state.run) setState(withNewRun(state));
  }, [isOpen]);

  function change(next: Partial<PlaygroundState>, retrains: boolean) {
    const updated = { ...state, ...next };
    if (!retrains) setState(updated);
    else setState(updated.isAutoTrain ? withNewRun(updated) : { ...updated, isStale: true });
  }

  return {
    state,
    changeSettings: (settings: PlaygroundSettings, retrains: boolean) => change({ settings }, retrains),
    addRow: (key: RowKey) => change({ addedRows: [...state.addedRows, key] }, false),
    removeRow: (key: RowKey) =>
      change({ addedRows: state.addedRows.filter((row) => row !== key), settings: { ...state.settings, [key]: defaultSettings(START_WORDS)[key] } }, false),
    reset: () => change({ settings: defaultSettings(START_WORDS), addedRows: [] }, true),
    train: () => setState(withNewRun(state)),
    writeAgain: () => setState({ ...state, writeSeed: state.writeSeed + 1 }),
    toggleAutoTrain() {
      const updated = { ...state, isAutoTrain: !state.isAutoTrain };
      setState(updated.isAutoTrain && updated.isStale ? withNewRun(updated) : updated);
    },
  };
}

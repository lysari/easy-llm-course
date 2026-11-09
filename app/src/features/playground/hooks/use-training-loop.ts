import { useEffect, useReducer } from "react";
import type { PlaygroundRun } from "@/features/playground/playground-run";

const FRAME_BUDGET_MS = 12;

export function useTrainingLoop(run: PlaygroundRun | null) {
  const [, redraw] = useReducer((frames: number) => frames + 1, 0);

  useEffect(() => {
    if (!run) return;
    let frameId = requestAnimationFrame(function frame() {
      run.trainFor(FRAME_BUDGET_MS);
      redraw();
      if (!run.isDone) frameId = requestAnimationFrame(frame);
    });
    return () => cancelAnimationFrame(frameId);
  }, [run]);
}

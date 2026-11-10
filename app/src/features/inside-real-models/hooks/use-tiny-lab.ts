import { useEffect, useReducer, useState } from "react";
import { ModelLab } from "@/core/model/lab";

const ROUNDS_PER_SECOND = 20;

export function useTinyLab(isOpen: boolean) {
  const [lab] = useState(() => new ModelLab());
  const [, redraw] = useReducer((frames: number) => frames + 1, 0);

  useEffect(() => {
    if (!isOpen || lab.isTrained) return;
    let lastFrame = performance.now();
    let owed = 0;
    let frameId = requestAnimationFrame(function frame(now) {
      owed += ((now - lastFrame) / 1000) * ROUNDS_PER_SECOND;
      lastFrame = now;
      const rounds = Math.floor(owed);
      owed -= rounds;
      if (rounds > 0) {
        lab.trainRounds(rounds);
        redraw();
      }
      if (!lab.isTrained) frameId = requestAnimationFrame(frame);
    });
    return () => cancelAnimationFrame(frameId);
  }, [isOpen, lab]);

  return lab;
}

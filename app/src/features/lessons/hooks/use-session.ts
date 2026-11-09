import { useEffect, useReducer, useState } from "react";
import { Session } from "@/features/lessons/session";

export function useSession() {
  const [session] = useState(() => new Session());
  const [, redraw] = useReducer((frames: number) => frames + 1, 0);
  const [loopId, startLoop] = useReducer((id: number) => id + 1, 0);

  useEffect(() => {
    if (!session.hasRunningJobs) return;
    let lastFrame = performance.now();
    let frameId = requestAnimationFrame(function frame(now) {
      session.tick((now - lastFrame) / 1000);
      lastFrame = now;
      redraw();
      if (session.hasRunningJobs) frameId = requestAnimationFrame(frame);
    });
    return () => cancelAnimationFrame(frameId);
  }, [session, loopId]);

  function runCommand(command: string) {
    const wasRunning = session.hasRunningJobs;
    session.runCommand(command);
    if (!wasRunning && session.hasRunningJobs) startLoop();
    else redraw();
  }

  return { session, runCommand };
}

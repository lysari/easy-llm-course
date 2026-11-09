import { useEffect, useRef, useState } from "react";
import type { StageId } from "@/features/lessons/interfaces/story.interface";
import { TRAINING_PATH, stepsOf } from "@/features/lessons/flow";
import { useDrawer } from "@/shared/hooks/use-drawer";
import { useSession } from "./use-session";

const LEAVE_MS = 150;
const STEP_ORDER_MS = 450;

const prefersReducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

export function useLessons(isOpen: boolean, stageId: StageId) {
  const { session, runCommand } = useSession();
  const [partsStageId, setPartsStageId] = useState(stageId);
  const partCards = useRef(new Map<string, HTMLElement>());
  const drawer = useDrawer(partCards);
  const closeDrawer = drawer.close;

  if (partsStageId !== stageId && prefersReducedMotion()) setPartsStageId(stageId);
  const isLeaving = partsStageId !== stageId;

  useEffect(() => {
    if (!isLeaving) return;
    const timer = window.setTimeout(() => setPartsStageId(stageId), LEAVE_MS);
    return () => window.clearTimeout(timer);
  }, [isLeaving, stageId]);

  useEffect(() => {
    closeDrawer();
  }, [isOpen, stageId, closeDrawer]);

  const openStep = drawer.openId && !isLeaving ? stepsOf(partsStageId).find((step) => step.id === drawer.openId) : undefined;
  const activeStepId = session.isRunning("train") ? TRAINING_PATH[Math.floor(performance.now() / STEP_ORDER_MS) % TRAINING_PATH.length] : null;

  return {
    session,
    partsStageId,
    isLeaving,
    openStep,
    activeStepId,
    partCards,
    drawerRef: drawer.drawerRef,
    runCommand,
    togglePart: drawer.toggle,
    closeDrawer,
  };
}

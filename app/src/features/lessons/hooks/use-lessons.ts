import { useEffect, useRef, useState } from "react";
import type { StageId } from "@/features/lessons/interfaces/story.interface";
import { TRAINING_PATH, stepsOf } from "@/features/lessons/flow";
import { useSession } from "./use-session";

const LEAVE_MS = 150;
const STEP_ORDER_MS = 450;

const prefersReducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

export function useLessons(isOpen: boolean, stageId: StageId) {
  const { session, runCommand } = useSession();
  const [partsStageId, setPartsStageId] = useState(stageId);
  const [openStepId, setOpenStepId] = useState<string | null>(null);
  const partCards = useRef(new Map<string, HTMLElement>());
  const drawerRef = useRef<HTMLElement>(null);

  if (partsStageId !== stageId && prefersReducedMotion()) setPartsStageId(stageId);
  const isLeaving = partsStageId !== stageId;

  useEffect(() => {
    if (!isLeaving) return;
    const timer = window.setTimeout(() => setPartsStageId(stageId), LEAVE_MS);
    return () => window.clearTimeout(timer);
  }, [isLeaving, stageId]);

  useEffect(() => {
    setOpenStepId(null);
  }, [isOpen, stageId]);

  useEffect(() => {
    if (!openStepId) return;
    function closeOnOutsideClick(event: MouseEvent) {
      const target = event.target;
      if (!(target instanceof Node) || drawerRef.current?.contains(target)) return;
      if ([...partCards.current.values()].some((card) => card.contains(target))) return;
      setOpenStepId(null);
    }
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") setOpenStepId(null);
    }
    document.addEventListener("click", closeOnOutsideClick, true);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("click", closeOnOutsideClick, true);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [openStepId]);

  useEffect(() => {
    if (!openStepId) return;
    drawerRef.current?.focus();
    const cards = partCards.current;
    return () => {
      if (document.activeElement === document.body) cards.get(openStepId)?.focus();
    };
  }, [openStepId]);

  const openStep = openStepId && !isLeaving ? stepsOf(partsStageId).find((step) => step.id === openStepId) : undefined;
  const activeStepId = session.isRunning("train") ? TRAINING_PATH[Math.floor(performance.now() / STEP_ORDER_MS) % TRAINING_PATH.length] : null;

  return {
    session,
    partsStageId,
    isLeaving,
    openStep,
    activeStepId,
    partCards,
    drawerRef,
    runCommand,
    togglePart(stepId: string) {
      setOpenStepId((current) => (current === stepId ? null : stepId));
    },
    closeDrawer() {
      setOpenStepId(null);
    },
  };
}

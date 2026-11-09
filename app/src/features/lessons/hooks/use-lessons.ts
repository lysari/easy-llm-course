import { useEffect, useRef, useState } from "react";
import type { StageId } from "@/features/lessons/interfaces/story.interface";
import { TRAINING_PATH, stepsOf } from "@/features/lessons/flow";
import { useSession } from "./use-session";

const SELECT_DELAY_MS = 120;
const CLOSE_DELAY_MS = 260;
const LEAVE_MS = 150;
const STEP_ORDER_MS = 450;

interface OpenPanel {
  stepId: string;
  isPinned: boolean;
}

const canHover = () => !matchMedia("(hover: none)").matches;
const prefersReducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

export function useLessons(isOpen: boolean, stageId: StageId, onSelectStage: (id: StageId) => void) {
  const { session, runCommand } = useSession();
  const [partsStageId, setPartsStageId] = useState(stageId);
  const [panel, setPanel] = useState<OpenPanel | null>(null);
  const timers = useRef({ select: 0, open: 0, close: 0 });

  if (partsStageId !== stageId && prefersReducedMotion()) setPartsStageId(stageId);
  const isLeaving = partsStageId !== stageId;

  function clearTimers() {
    window.clearTimeout(timers.current.select);
    window.clearTimeout(timers.current.open);
    window.clearTimeout(timers.current.close);
  }

  function openPanel(stepId: string, pin: boolean) {
    window.clearTimeout(timers.current.close);
    setPanel((current) => ({ stepId, isPinned: pin || (current?.isPinned === true && current.stepId === stepId) }));
  }

  useEffect(() => {
    if (!isLeaving) return;
    const timer = window.setTimeout(() => setPartsStageId(stageId), LEAVE_MS);
    return () => window.clearTimeout(timer);
  }, [isLeaving, stageId]);

  useEffect(() => {
    setPanel(null);
    return clearTimers;
  }, [isOpen, stageId]);

  useEffect(() => {
    if (!isOpen) return;
    function closeOnOutsideClick() {
      window.clearTimeout(timers.current.select);
      window.clearTimeout(timers.current.open);
      if (panel?.isPinned) {
        setPanel(null);
      } else if (panel) {
        window.clearTimeout(timers.current.close);
        timers.current.close = window.setTimeout(() => setPanel(null), CLOSE_DELAY_MS);
      }
    }
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape" && panel) setPanel(null);
    }
    document.addEventListener("click", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("click", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [isOpen, panel]);

  const openStep = panel && !isLeaving ? stepsOf(partsStageId).find((step) => step.id === panel.stepId) : undefined;
  const activeStepId = session.isRunning("train") ? TRAINING_PATH[Math.floor(performance.now() / STEP_ORDER_MS) % TRAINING_PATH.length] : null;

  return {
    session,
    partsStageId,
    isLeaving,
    openStep,
    activeStepId,
    clickStage(id: StageId) {
      window.clearTimeout(timers.current.select);
      onSelectStage(id);
    },
    stageCommand(id: StageId, command: string) {
      runCommand(command);
      window.clearTimeout(timers.current.select);
      if (canHover()) timers.current.select = window.setTimeout(() => onSelectStage(id), SELECT_DELAY_MS);
    },
    clickPart(stepId: string) {
      clearTimers();
      if (panel?.isPinned && panel.stepId === stepId) setPanel(null);
      else openPanel(stepId, true);
    },
    partCommand(stepId: string, command: string) {
      runCommand(command);
      if (!canHover()) return;
      clearTimers();
      if (panel?.isPinned && panel.stepId !== stepId) return;
      timers.current.open = window.setTimeout(() => openPanel(stepId, false), SELECT_DELAY_MS);
    },
    panelCommand: runCommand,
    clickPanel: clearTimers,
  };
}

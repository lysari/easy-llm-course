import "./lessons.css";
import { Fragment, useRef } from "react";
import type { FlowStep, StageId } from "@/features/lessons/interfaces/story.interface";
import { NESTED_STEPS, STAGES, STEPS, stepsOf } from "./flow";
import { FlowLinks } from "./flow-links";
import { useFlowLinks } from "./hooks/use-flow-links";
import { useLessons } from "./hooks/use-lessons";
import { NARROW_WIDTH, useWindowWidth } from "./hooks/use-window-width";
import { PartCard, StageCard } from "./lesson-cards";
import { PartDrawer } from "./part-drawer";

interface LessonsPageProps {
  isOpen: boolean;
  stageId: StageId;
  onSelectStage: (id: StageId) => void;
}

export function LessonsPage({ isOpen, stageId, onSelectStage }: LessonsPageProps) {
  const lessons = useLessons(isOpen, stageId);
  const { session, partsStageId, isLeaving, openStep, partCards } = lessons;
  const width = useWindowWidth();
  const isNarrow = width <= NARROW_WIDTH;
  const flowRef = useRef<HTMLDivElement>(null);
  const partsRef = useRef<HTMLDivElement>(null);
  const stageCards = useRef(new Map<StageId, HTMLElement>());
  const links = useFlowLinks({ flowRef, partsRef, stageCards, partCards }, partsStageId, isOpen && !isLeaving && !isNarrow, width);

  const statuses = STEPS.map((step) => step.status(session));
  const steps = stepsOf(partsStageId);
  const partCard = (step: FlowStep) => (
    <PartCard
      key={step.id}
      ref={(element) => {
        if (element) partCards.current.set(step.id, element);
        else partCards.current.delete(step.id);
      }}
      step={step}
      session={session}
      index={steps.indexOf(step)}
      isNested={NESTED_STEPS.includes(step.id)}
      isActive={step.id === lessons.activeStepId}
      isOpen={step === openStep}
      onToggle={() => lessons.togglePart(step.id)}
      onCommand={lessons.runCommand}
    />
  );

  const partColumn = (
    <div className={`column part-column${isLeaving ? " is-leaving" : ""}`} id="parts" ref={partsRef}>
      {steps.map((step, i) => {
        if (NESTED_STEPS.includes(step.id)) return null;
        const next = steps[i + 1];
        const subStep = next && NESTED_STEPS.includes(next.id) ? next : undefined;
        return (
          <div className="part-row" key={step.id}>
            {partCard(step)}
            {subStep && partCard(subStep)}
          </div>
        );
      })}
    </div>
  );

  return (
    <main className="page" id="lessons" hidden={!isOpen}>
      <div className="fleet">
        <header className="top">
          <h1 className="page-title">Lessons</h1>
          <p className="top-stats" id="top-stats">
            <span>{`${STAGES.length} stages / ${STEPS.length} parts`}</span>
            <span>{`${statuses.filter((status) => status === "running").length} running`}</span>
            <span>{`${statuses.filter((status) => status === "needs").length} need you`}</span>
            <strong>{`Loss ${(session.lab.latestLoss ?? session.startLoss).toFixed(2)}`}</strong>
          </p>
        </header>
        <div className="flow" id="flow" ref={flowRef}>
          <FlowLinks drawing={links} />
          <div className="column" id="stages">
            {STAGES.map((stage) => (
              <Fragment key={stage.id}>
                <StageCard
                  ref={(element) => {
                    if (element) stageCards.current.set(stage.id, element);
                    else stageCards.current.delete(stage.id);
                  }}
                  stage={stage}
                  session={session}
                  isSelected={stage.id === stageId}
                  onSelect={() => onSelectStage(stage.id)}
                  onCommand={lessons.runCommand}
                />
                {isNarrow && stage.id === partsStageId && partColumn}
              </Fragment>
            ))}
          </div>
          {!isNarrow && partColumn}
        </div>
      </div>
      {openStep && <PartDrawer ref={lessons.drawerRef} step={openStep} session={session} onCommand={lessons.runCommand} onClose={lessons.closeDrawer} />}
    </main>
  );
}

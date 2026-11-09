import "./layout.css";
import { useState } from "react";
import type { StageId } from "@/features/lessons/interfaces/story.interface";
import { LessonsPage } from "@/features/lessons/lessons-page";
import { PlaygroundPage } from "@/features/playground/playground-page";
import { useIsPlayground } from "./router";
import { Sidebar } from "./sidebar";

export function Layout() {
  const isPlayground = useIsPlayground();
  const [stageId, setStageId] = useState<StageId>("learn");
  return (
    <>
      <Sidebar isPlayground={isPlayground} stageId={stageId} onSelectStage={setStageId} />
      <LessonsPage isOpen={!isPlayground} stageId={stageId} onSelectStage={setStageId} />
      <PlaygroundPage isOpen={isPlayground} />
    </>
  );
}

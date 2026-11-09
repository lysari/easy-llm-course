import "./layout.css";
import { useState } from "react";
import { Navigate, Route, Routes, useLocation, useMatch } from "react-router";
import type { StageId } from "@/features/lessons/interfaces/story.interface";
import { LessonsPage } from "@/features/lessons/lessons-page";
import { PlaygroundPage } from "@/features/playground/playground-page";
import { Sidebar } from "./sidebar";

const LESSONS_PATH = "/";
const PLAYGROUND_PATH = "/playground";
const OLD_HASH_PATHS: Partial<Record<string, string>> = { "#lessons": LESSONS_PATH, "#playground": PLAYGROUND_PATH };

export function Layout() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route path={LESSONS_PATH} element={null} />
        <Route path={PLAYGROUND_PATH} element={null} />
      </Route>
      <Route path="*" element={<Navigate to={LESSONS_PATH} replace />} />
    </Routes>
  );
}

function Shell() {
  const oldHashPath = OLD_HASH_PATHS[useLocation().hash];
  const isPlaygroundPath = useMatch(PLAYGROUND_PATH) !== null;
  const isPlayground = oldHashPath ? oldHashPath === PLAYGROUND_PATH : isPlaygroundPath;
  const [stageId, setStageId] = useState<StageId>("learn");
  return (
    <>
      {oldHashPath && <Navigate to={oldHashPath} replace />}
      <Sidebar isPlayground={isPlayground} stageId={stageId} onSelectStage={setStageId} />
      <LessonsPage isOpen={!isPlayground} stageId={stageId} onSelectStage={setStageId} />
      <PlaygroundPage isOpen={isPlayground} />
    </>
  );
}

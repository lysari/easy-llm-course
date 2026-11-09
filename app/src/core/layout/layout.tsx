import "./layout.css";
import { useState } from "react";
import { Navigate, Route, Routes, useLocation, useMatch } from "react-router";
import type { StageId } from "@/features/lessons/interfaces/story.interface";
import { LessonsPage } from "@/features/lessons/lessons-page";
import { PlaygroundPage } from "@/features/playground/playground-page";
import { TAKE_APART_PATH } from "@/features/take-apart/models";
import { TakeApartPage } from "@/features/take-apart/take-apart-page";
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
        <Route path={`${TAKE_APART_PATH}/*`} element={null} />
      </Route>
      <Route path="*" element={<Navigate to={LESSONS_PATH} replace />} />
    </Routes>
  );
}

function Shell() {
  const oldHashPath = OLD_HASH_PATHS[useLocation().hash];
  const isPlaygroundPath = useMatch(PLAYGROUND_PATH) !== null;
  const isTakeApartPath = useMatch(`${TAKE_APART_PATH}/*`) !== null;
  const isPlayground = oldHashPath ? oldHashPath === PLAYGROUND_PATH : isPlaygroundPath;
  const isTakeApart = !oldHashPath && isTakeApartPath;
  const isLessons = !isPlayground && !isTakeApart;
  const [stageId, setStageId] = useState<StageId>("learn");
  return (
    <>
      {oldHashPath && <Navigate to={oldHashPath} replace />}
      <Sidebar isLessons={isLessons} stageId={stageId} onSelectStage={setStageId} />
      <LessonsPage isOpen={isLessons} stageId={stageId} onSelectStage={setStageId} />
      <PlaygroundPage isOpen={isPlayground} />
      <TakeApartPage isOpen={isTakeApart} />
    </>
  );
}

import "./layout.css";
import { Suspense, lazy, useState } from "react";
import { Navigate, Route, Routes, useLocation, useMatch } from "react-router";
import { INSIDE_REAL_MODELS_PATH, OLD_INSIDE_REAL_MODELS_PATH, movedPath } from "@/features/inside-real-models/inside-real-models-path";
import type { StageId } from "@/features/lessons/interfaces/story.interface";
import { LessonsPage } from "@/features/lessons/lessons-page";
import { PlaygroundPage } from "@/features/playground/playground-page";
import { Sidebar } from "./sidebar";

const LESSONS_PATH = "/";
const PLAYGROUND_PATH = "/playground";
const InsideRealModelsPage = lazy(() => import("@/features/inside-real-models/inside-real-models-page").then((module) => ({ default: module.InsideRealModelsPage })));
const OLD_HASH_PATHS: Partial<Record<string, string>> = { "#lessons": LESSONS_PATH, "#playground": PLAYGROUND_PATH };

export function Layout() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route path={LESSONS_PATH} element={null} />
        <Route path={PLAYGROUND_PATH} element={null} />
        <Route path={`${INSIDE_REAL_MODELS_PATH}/*`} element={null} />
      </Route>
      <Route path={`${OLD_INSIDE_REAL_MODELS_PATH}/*`} element={<OldPathRedirect />} />
      <Route path="*" element={<Navigate to={LESSONS_PATH} replace />} />
    </Routes>
  );
}

function OldPathRedirect() {
  const { pathname, search, hash } = useLocation();
  return <Navigate to={{ pathname: movedPath(pathname), search, hash }} replace />;
}

function Shell() {
  const oldHashPath = OLD_HASH_PATHS[useLocation().hash];
  const isPlaygroundPath = useMatch(PLAYGROUND_PATH) !== null;
  const isInsideRealModelsPath = useMatch(`${INSIDE_REAL_MODELS_PATH}/*`) !== null;
  const isPlayground = oldHashPath ? oldHashPath === PLAYGROUND_PATH : isPlaygroundPath;
  const isInsideRealModels = !oldHashPath && isInsideRealModelsPath;
  const isLessons = !isPlayground && !isInsideRealModels;
  const [stageId, setStageId] = useState<StageId>("learn");
  return (
    <>
      {oldHashPath && <Navigate to={oldHashPath} replace />}
      <Sidebar isLessons={isLessons} stageId={stageId} onSelectStage={setStageId} />
      <LessonsPage isOpen={isLessons} stageId={stageId} onSelectStage={setStageId} />
      <PlaygroundPage isOpen={isPlayground} />
      <Suspense fallback={null}>
        <InsideRealModelsPage isOpen={isInsideRealModels} />
      </Suspense>
    </>
  );
}

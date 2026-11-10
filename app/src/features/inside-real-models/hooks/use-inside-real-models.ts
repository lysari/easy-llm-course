import { useEffect, useMemo, useRef } from "react";
import { useLocation, useMatch, useNavigate } from "react-router";
import type { ModeId, ModelPart } from "@/features/inside-real-models/interfaces/model-part.interface";
import { DEFAULT_MODEL, findModel, matchingPath, pathUrl, readPath } from "@/features/inside-real-models/models";
import { INSIDE_REAL_MODELS_PATH } from "@/features/inside-real-models/inside-real-models-path";
import { useDrawer } from "@/shared/hooks/use-drawer";
import { useTinyLab } from "./use-tiny-lab";

const scrollBehavior = (): ScrollBehavior => (matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth");

function scrollColumnIntoView(scroller: HTMLElement, column: HTMLElement) {
  const left = column.offsetLeft;
  const right = left + column.offsetWidth;
  if (left < scroller.scrollLeft) scroller.scrollTo({ left, behavior: scrollBehavior() });
  else if (right > scroller.scrollLeft + scroller.clientWidth) scroller.scrollTo({ left: right - scroller.clientWidth, behavior: scrollBehavior() });
}

export function useInsideRealModels(isOpen: boolean, isNarrow: boolean) {
  const match = useMatch(`${INSIDE_REAL_MODELS_PATH}/:model/*`);
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const model = findModel(match?.params.model) ?? DEFAULT_MODEL;
  const splat = match?.params["*"] ?? "";
  const path = useMemo(() => readPath(model, splat.split("/")), [model, splat]);
  const mode: ModeId = path[0] === model.modes.build ? "build" : "use";
  const url = pathUrl(model.id, path);
  const cards = useRef(new Map<string, HTMLElement>());
  const scrollerRef = useRef<HTMLDivElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const drawer = useDrawer(cards);
  const closeDrawer = drawer.close;
  const lab = useTinyLab(isOpen);
  const shownUrl = useRef(url);

  useEffect(() => {
    closeDrawer();
  }, [isOpen, url, closeDrawer]);

  useEffect(() => {
    if (shownUrl.current === url) return;
    shownUrl.current = url;
    if (document.activeElement === document.body) cards.current.get(path[path.length - 1].id)?.focus({ preventScroll: true });
  }, [url, path]);

  useEffect(() => {
    const newest = trackRef.current?.lastElementChild;
    if (!isOpen || !scrollerRef.current || !(newest instanceof HTMLElement)) return;
    if (isNarrow) newest.scrollIntoView({ behavior: scrollBehavior(), block: "start" });
    else scrollColumnIntoView(scrollerRef.current, newest);
  }, [isOpen, isNarrow, url]);

  function select(part: ModelPart, level: number) {
    if (!part.children) {
      drawer.toggle(part.id);
      return;
    }
    const next = pathUrl(model.id, [...path.slice(0, level + 1), part]);
    if (next !== url) navigate(next);
  }

  function showCard(id: string) {
    const card = cards.current.get(id);
    card?.focus({ preventScroll: true });
    card?.scrollIntoView({ behavior: scrollBehavior(), block: "nearest", inline: "nearest" });
  }

  function switchModel(id: string) {
    const next = findModel(id);
    if (next) navigate(pathUrl(next.id, matchingPath(next.modes[mode], path)));
  }

  function switchMode(next: ModeId) {
    if (next !== mode) navigate(pathUrl(model.id, matchingPath(model.modes[next], path)));
  }

  return {
    model,
    mode,
    path,
    redirect: isOpen && url !== pathname ? url : null,
    lab,
    cards,
    scrollerRef,
    trackRef,
    openId: drawer.openId,
    drawerRef: drawer.drawerRef,
    select,
    showCard,
    showDetails: drawer.toggle,
    closeDrawer,
    switchModel,
    switchMode,
  };
}

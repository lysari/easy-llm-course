import "./inside-real-models.css";
import { useMemo, useState } from "react";
import { Navigate } from "react-router";
import { NARROW_WIDTH, useWindowWidth } from "@/shared/hooks/use-window-width";
import { Breadcrumb } from "./breadcrumb";
import { useInsideRealModels } from "./hooks/use-inside-real-models";
import { activeIdOf, pathIdsOf } from "./models";
import { PartColumn } from "./part-column";
import { PartDrawer } from "./part-drawer";
import { placeOf } from "./real-place";
import { START_TEXTS, readSentence } from "./sentence";
import { SentenceBar } from "./sentence-bar";
import { InsideRealModelsHeader } from "./inside-real-models-header";
import { viewTiny } from "./tiny-parts";

export function InsideRealModelsPage({ isOpen }: { isOpen: boolean }) {
  const width = useWindowWidth();
  const isNarrow = width <= NARROW_WIDTH;
  const insideRealModels = useInsideRealModels(isOpen, isNarrow);
  const { model, mode, path, lab, openId } = insideRealModels;
  const [text, setText] = useState(START_TEXTS[0]);
  const sentence = useMemo(() => readSentence(text), [text]);
  const tiny = viewTiny(lab, sentence.words);
  const root = path[0];
  const columns = useMemo(() => path.map((part) => ({ parent: part, parts: part.children ?? [] })), [path]);
  const shownParts = columns.flatMap((column) => column.parts);
  const openPart = shownParts.find((part) => part.id === openId);
  const nextPart = shownParts.find((part) => part.id === openPart?.story?.nextId);
  const activeId = activeIdOf(path, openPart);
  const pathIds = pathIdsOf(path, openPart);

  return (
    <main className="page" id="inside-real-models" hidden={!isOpen}>
      {insideRealModels.redirect && <Navigate to={insideRealModels.redirect} replace />}
      <InsideRealModelsHeader model={model} mode={mode} lab={lab} onSwitchModel={insideRealModels.switchModel} onSwitchMode={insideRealModels.switchMode} />
      <SentenceBar text={text} sentence={sentence} onChange={setText} />
      <Breadcrumb path={path} onShow={insideRealModels.showCard} />
      <div className="column-scroll" ref={insideRealModels.scrollerRef}>
        <div className="column-track" ref={insideRealModels.trackRef}>
          {columns.map(({ parent, parts }, level) => (
            <PartColumn
              key={`${model.id}-${root.id}-${parent.id}`}
              model={model}
              root={root}
              parent={parent}
              parts={parts}
              selectedId={path[level + 1]?.id}
              activeId={activeId}
              pathIds={pathIds}
              openId={openId}
              cards={insideRealModels.cards}
              hasLinks={isOpen && !isNarrow}
              width={width}
              onSelect={(part) => insideRealModels.select(part, level)}
              onDetails={insideRealModels.showDetails}
            />
          ))}
        </div>
      </div>
      {openPart && <PartDrawer ref={insideRealModels.drawerRef} root={root} modelId={model.id} modelTitle={model.title} part={openPart} place={placeOf(model, root, openPart)} tiny={tiny} nextPart={nextPart} onOpen={insideRealModels.showDetails} onClose={insideRealModels.closeDrawer} />}
    </main>
  );
}

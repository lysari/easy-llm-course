import "./take-apart.css";
import { useMemo, useState } from "react";
import { Navigate } from "react-router";
import { NARROW_WIDTH, useWindowWidth } from "@/shared/hooks/use-window-width";
import { Breadcrumb } from "./breadcrumb";
import { useTakeApart } from "./hooks/use-take-apart";
import { PartColumn } from "./part-column";
import { PartDrawer } from "./part-drawer";
import { START_TEXTS, readSentence } from "./sentence";
import { SentenceBar } from "./sentence-bar";
import { TakeApartHeader } from "./take-apart-header";
import { viewTiny } from "./tiny-parts";

export function TakeApartPage({ isOpen }: { isOpen: boolean }) {
  const width = useWindowWidth();
  const isNarrow = width <= NARROW_WIDTH;
  const takeApart = useTakeApart(isOpen, isNarrow);
  const { model, mode, path, lab, openId } = takeApart;
  const [text, setText] = useState(START_TEXTS[0]);
  const sentence = useMemo(() => readSentence(text), [text]);
  const tiny = viewTiny(lab, sentence.words);
  const root = path[0];
  const columns = useMemo(() => path.map((part) => ({ parent: part, parts: part.children ?? [] })), [path]);
  const openPart = columns.flatMap((column) => column.parts).find((part) => part.id === openId);

  return (
    <main className="page" id="take-apart" hidden={!isOpen}>
      {takeApart.redirect && <Navigate to={takeApart.redirect} replace />}
      <TakeApartHeader model={model} mode={mode} lab={lab} onSwitchModel={takeApart.switchModel} onSwitchMode={takeApart.switchMode} />
      <SentenceBar text={text} sentence={sentence} onChange={setText} />
      <Breadcrumb path={path} onShow={takeApart.showCard} />
      <div className="column-scroll" ref={takeApart.scrollerRef}>
        <div className="column-track" ref={takeApart.trackRef}>
          {columns.map(({ parent, parts }, level) => (
            <PartColumn
              key={`${model.id}-${root.id}-${parent.id}`}
              root={root}
              modelTitle={model.title}
              parent={parent}
              parts={parts}
              selectedId={path[level + 1]?.id}
              openId={openId}
              cards={takeApart.cards}
              tiny={tiny}
              hasLinks={isOpen && !isNarrow}
              width={width}
              onSelect={(part) => takeApart.select(part, level)}
              onDetails={takeApart.showDetails}
            />
          ))}
        </div>
      </div>
      {openPart && <PartDrawer ref={takeApart.drawerRef} root={root} modelTitle={model.title} part={openPart} tiny={tiny} sentence={sentence} onClose={takeApart.closeDrawer} />}
    </main>
  );
}

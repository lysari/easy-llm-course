import "./playground.css";
import { useRef } from "react";
import { Link } from "react-router";
import { LEARNING_RATE } from "@/core/model/adam";
import { ConceptIcon } from "@/shared/components/concept-icon";
import { Builder } from "./builder";
import { useMenu } from "./hooks/use-menu";
import { usePlayground } from "./hooks/use-playground";
import { Menu } from "./menu";
import { Result } from "./result";

export function PlaygroundPage({ isOpen }: { isOpen: boolean }) {
  const rootRef = useRef<HTMLElement>(null);
  const playground = usePlayground(isOpen);
  const menu = useMenu(rootRef, isOpen);
  const { state } = playground;

  return (
    <main
      className="page page--playground"
      id="playground"
      hidden={!isOpen}
      ref={rootRef}
      onClickCapture={menu.closeUnlessInMenu}
      onKeyDown={menu.handleKeyDown}
      onScroll={() => menu.close()}
    >
      <div className="pg">
        <div className="pg-main">
          <header className="pg-head">
            <div>
              <h1 className="page-title">Playground</h1>
              <p className="pg-sub">Tiny GPT, the model from Lessons, trains in your browser. Change a setting and watch the loss and the writing change.</p>
              <p className="pg-sub">{`Try this: set Step size to 0.4. The loss jumps above random guessing, then settles worse than at ${LEARNING_RATE}.`}</p>
            </div>
            <label className="pg-switch" title="Retrain as soon as you change a setting">
              <span id="pg-switch-label">Auto train</span>
              <button type="button" className="switch" role="switch" aria-checked={state.isAutoTrain} aria-labelledby="pg-switch-label" onClick={playground.toggleAutoTrain} />
            </label>
          </header>
          <Builder
            settings={state.settings}
            addedRows={state.addedRows}
            openMenuId={menu.openMenu?.id ?? null}
            onToggleMenu={menu.toggle}
            onChange={playground.changeSettings}
            onRemoveRow={playground.removeRow}
          />
          <Link className="card pg-next" to="/inside-real-models">
            <span className="card-head">
              <ConceptIcon name="model" size="card" />
              <span className="card-title">Next: take GPT-2 and Llama 3 apart in Inside real models</span>
              <span aria-hidden="true">›</span>
            </span>
          </Link>
        </div>
        <Result state={state} onTrain={playground.train} onWriteAgain={playground.writeAgain} onReset={playground.reset} />
      </div>
      {menu.openMenu && (
        <Menu
          key={menu.openMenu.id}
          ref={menu.menuRef}
          id={menu.openMenu.id}
          chip={menu.openMenu.chip}
          settings={state.settings}
          addedRows={state.addedRows}
          onChange={playground.changeSettings}
          onAddRow={playground.addRow}
          onClose={menu.close}
        />
      )}
    </main>
  );
}

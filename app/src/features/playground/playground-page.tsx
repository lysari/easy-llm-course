import "./playground.css";
import { useRef } from "react";
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
              <p className="pg-sub">A real model trains in your browser · Change a setting and see what happens</p>
            </div>
            <label className="pg-switch">
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

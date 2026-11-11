import type { ReactNode, Ref } from "react";
import type { Status } from "@/shared/interfaces/card.interface";
import type { ConceptName } from "@/shared/interfaces/concept.interface";
import { ConceptIcon } from "./concept-icon";
import { Icon } from "./icon";
import { StatusPill } from "./status-pill";
import "./drawer.css";

interface DrawerProps {
  ref: Ref<HTMLElement>;
  id: string;
  icon: ConceptName;
  title: string;
  lesson?: string;
  status?: Status;
  bodyKey: string;
  onClose: () => void;
  children: ReactNode;
  panel?: ReactNode;
}

export function Drawer({ ref, id, icon, title, lesson, status, bodyKey, onClose, children, panel }: DrawerProps) {
  return (
    <aside className={panel ? "drawer drawer--wide" : "drawer"} id={id} ref={ref} tabIndex={-1} aria-label={`${title} details`}>
      <header className="card-head drawer-head">
        <ConceptIcon name={icon} size="drawer" />
        <div>
          <h2 className="card-title">{title}</h2>
          {lesson && <p className="card-lesson">{lesson}</p>}
        </div>
        {status && <StatusPill status={status} />}
        <button type="button" className="drawer-close" aria-label="Close details" onClick={onClose}>
          <Icon name="close" />
        </button>
      </header>
      {panel ? (
        <div className="drawer-columns" key={bodyKey}>
          <div className="drawer-body">{children}</div>
          <div className="drawer-panel">{panel}</div>
        </div>
      ) : (
        <div className="drawer-body" key={bodyKey}>
          {children}
        </div>
      )}
    </aside>
  );
}

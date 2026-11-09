import type { ReactNode, Ref } from "react";
import type { Status } from "@/shared/interfaces/card.interface";
import { Icon } from "./icon";
import { StatusPill } from "./status-pill";
import "./drawer.css";

interface DrawerProps {
  ref: Ref<HTMLElement>;
  id: string;
  icon: string;
  title: string;
  lesson: string;
  status: Status;
  bodyKey: string;
  onClose: () => void;
  children: ReactNode;
}

export function Drawer({ ref, id, icon, title, lesson, status, bodyKey, onClose, children }: DrawerProps) {
  return (
    <aside className="drawer" id={id} ref={ref} tabIndex={-1} aria-label={`${title} details`}>
      <header className="card-head drawer-head">
        <span className="tile"><Icon name={icon} /></span>
        <div>
          <h2 className="card-title">{title}</h2>
          <p className="card-lesson">{lesson}</p>
        </div>
        <StatusPill status={status} />
        <button type="button" className="drawer-close" aria-label="Close details" onClick={onClose}>
          <Icon name="close" />
        </button>
      </header>
      <div className="drawer-body" key={bodyKey}>
        {children}
      </div>
    </aside>
  );
}

import { NavLink } from "react-router";
import { STAGES } from "@/features/lessons/flow";
import { TAKE_APART_PATH } from "@/features/take-apart/models";
import type { StageId } from "@/features/lessons/interfaces/story.interface";
import { Icon, SparkleIcon } from "@/shared/components/icon";

interface SidebarProps {
  isLessons: boolean;
  stageId: StageId;
  onSelectStage: (id: StageId) => void;
}

export function Sidebar({ isLessons, stageId, onSelectStage }: SidebarProps) {
  return (
    <aside className="sidebar">
      <p className="brand"><SparkleIcon /> Tiny GPT</p>
      <nav className="side-nav" aria-label="Pages">
        <NavLink className={({ isActive }) => `side-link side-link--page${isActive ? " is-on" : ""}`} id="nav-lessons" to="/">
          <Icon name="book" /> Lessons
        </NavLink>
        <div className="side-stages" id="side-stages" hidden={!isLessons}>
          {STAGES.map((stage) => (
            <button
              key={stage.id}
              type="button"
              className={`side-link${stage.id === stageId ? " is-selected" : ""}`}
              id={`nav-${stage.id}`}
              onClick={() => onSelectStage(stage.id)}
            >
              <Icon name={stage.icon} /> {stage.title}
            </button>
          ))}
        </div>
        <NavLink className={({ isActive }) => `side-link side-link--page${isActive ? " is-on" : ""}`} id="nav-playground" to="/playground">
          <Icon name="play" /> Playground
        </NavLink>
        <NavLink className={({ isActive }) => `side-link side-link--page${isActive ? " is-on" : ""}`} id="nav-take-apart" to={TAKE_APART_PATH}>
          <Icon name="split" /> Take apart
        </NavLink>
      </nav>
    </aside>
  );
}

import { NavLink } from "react-router";
import { STAGES } from "@/features/lessons/flow";
import { INSIDE_REAL_MODELS_PATH } from "@/features/inside-real-models/inside-real-models-path";
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
        <NavLink className={({ isActive }) => `side-link side-link--page${isActive ? " is-on" : ""}`} id="nav-inside-real-models" to={INSIDE_REAL_MODELS_PATH}>
          <Icon name="split" />
          <span className="side-label--full">Inside real models</span>
          <span className="side-label--short">Real models</span>
        </NavLink>
      </nav>
    </aside>
  );
}

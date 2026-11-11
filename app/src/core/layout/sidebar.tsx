import { NavLink } from "react-router";
import { STAGES } from "@/features/lessons/flow";
import { INSIDE_REAL_MODELS_PATH } from "@/features/inside-real-models/inside-real-models-path";
import type { StageId } from "@/features/lessons/interfaces/story.interface";
import { ConceptIcon } from "@/shared/components/concept-icon";
import { Logo } from "./logo";
import { useBackdrop } from "./use-backdrop";

const PHOTO_URL = "https://unsplash.com/photos/blue-ocean-water-surface-is-calm-and-serene-yWCfIqyUapk";

interface SidebarProps {
  isLessons: boolean;
  stageId: StageId;
  onSelectStage: (id: StageId) => void;
}

const pageLinkClass = ({ isActive }: { isActive: boolean }) => `side-link side-link--page${isActive ? " is-on" : ""}`;

export function Sidebar({ isLessons, stageId, onSelectStage }: SidebarProps) {
  const { backdrop, setBackdrop } = useBackdrop();
  return (
    <aside className="sidebar">
      <p className="brand">
        <Logo />
        <span>LLM <span className="brand-soft">Lessons</span></span>
      </p>
      <nav className="side-nav" aria-label="Pages">
        <NavLink className={pageLinkClass} id="nav-lessons" to="/">
          <ConceptIcon name="lessons" size="nav" />
          <span className="side-text">
            <span className="side-title">Lessons</span>
            <span className="side-hint">Build a tiny model step by step</span>
          </span>
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
              <ConceptIcon name={stage.icon} size="chip" /> {stage.title}
            </button>
          ))}
        </div>
        <NavLink className={pageLinkClass} id="nav-playground" to="/playground">
          <ConceptIcon name="playground" size="nav" />
          <span className="side-text">
            <span className="side-title">Playground</span>
            <span className="side-hint">Change settings, watch it learn</span>
          </span>
        </NavLink>
        <NavLink className={pageLinkClass} id="nav-inside-real-models" to={INSIDE_REAL_MODELS_PATH}>
          <ConceptIcon name="model" size="nav" />
          <span className="side-text">
            <span className="side-title side-label--full">Inside real models</span>
            <span className="side-title side-label--short">Real models</span>
            <span className="side-hint">Take GPT-2 and Llama 3 apart</span>
          </span>
        </NavLink>
      </nav>
      <footer className="side-foot">
        <div className="backdrop-switch" role="group" aria-label="Background">
          <span className="backdrop-label" aria-hidden="true">Background</span>
          <button type="button" aria-pressed={backdrop === "wash"} onClick={() => setBackdrop("wash")}>Wash</button>
          <button type="button" aria-pressed={backdrop === "photo"} onClick={() => setBackdrop("photo")}>Photo</button>
        </div>
        {backdrop === "photo" && (
          <p className="photo-credit">
            Photo: <a href={PHOTO_URL} target="_blank" rel="noreferrer">Engin Akyurt, Unsplash</a>
          </p>
        )}
      </footer>
    </aside>
  );
}

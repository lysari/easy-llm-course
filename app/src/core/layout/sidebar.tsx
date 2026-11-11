import { NavLink } from "react-router";
import { STAGES } from "@/features/lessons/flow";
import { INSIDE_REAL_MODELS_PATH } from "@/features/inside-real-models/inside-real-models-path";
import type { StageId } from "@/features/lessons/interfaces/story.interface";
import { ConceptIcon } from "@/shared/components/concept-icon";
import { Logo } from "./logo";
import { type Backdrop, useBackdrop } from "./use-backdrop";
import { useLook } from "./use-look";

const PHOTO_URL = "https://unsplash.com/photos/blue-ocean-water-surface-is-calm-and-serene-yWCfIqyUapk";

const BACKDROP_LABELS: Record<Backdrop, string> = { wash: "Default", meadow: "Meadow", sky: "Sky", citrus: "Citrus", photo: "Photo" };

interface SidebarProps {
  isLessons: boolean;
  stageId: StageId;
  onSelectStage: (id: StageId) => void;
}

const pageLinkClass = ({ isActive }: { isActive: boolean }) => `side-link side-link--page${isActive ? " is-on" : ""}`;

export function Sidebar({ isLessons, stageId, onSelectStage }: SidebarProps) {
  const { backdrop, setBackdrop } = useBackdrop();
  const { look, setLook } = useLook();
  return (
    <aside className="sidebar">
      <p className="brand">
        <Logo />
        <span>Open<span className="brand-soft">Course</span></span>
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
        <div className="side-switch" role="group" aria-label="Look">
          <span className="side-switch-label" aria-hidden="true">Look</span>
          <button type="button" aria-pressed={look === "auto"} onClick={() => setLook("auto")}>Auto</button>
          <button type="button" aria-pressed={look === "light"} onClick={() => setLook("light")}>Light</button>
          <button type="button" aria-pressed={look === "dark"} onClick={() => setLook("dark")}>Dark</button>
          <button type="button" aria-pressed={look === "fresh"} onClick={() => setLook("fresh")}>Fresh</button>
        </div>
        <div className="side-swatches" role="group" aria-label="Background">
          <span className="side-switch-label" aria-hidden="true">Background · {BACKDROP_LABELS[backdrop]}</span>
          <div className="side-swatch-row">
            <button
              type="button"
              className="side-swatch side-swatch--wash"
              aria-label={BACKDROP_LABELS.wash}
              aria-pressed={backdrop === "wash"}
              title={BACKDROP_LABELS.wash}
              onClick={() => setBackdrop("wash")}
            />
            <button
              type="button"
              className="side-swatch side-swatch--meadow"
              aria-label={BACKDROP_LABELS.meadow}
              aria-pressed={backdrop === "meadow"}
              title={BACKDROP_LABELS.meadow}
              onClick={() => setBackdrop("meadow")}
            />
            <button
              type="button"
              className="side-swatch side-swatch--sky"
              aria-label={BACKDROP_LABELS.sky}
              aria-pressed={backdrop === "sky"}
              title={BACKDROP_LABELS.sky}
              onClick={() => setBackdrop("sky")}
            />
            <button
              type="button"
              className="side-swatch side-swatch--citrus"
              aria-label={BACKDROP_LABELS.citrus}
              aria-pressed={backdrop === "citrus"}
              title={BACKDROP_LABELS.citrus}
              onClick={() => setBackdrop("citrus")}
            />
            <button
              type="button"
              className="side-swatch side-swatch--photo"
              aria-label={BACKDROP_LABELS.photo}
              aria-pressed={backdrop === "photo"}
              title={BACKDROP_LABELS.photo}
              onClick={() => setBackdrop("photo")}
            />
          </div>
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

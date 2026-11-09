import { STAGES } from "@/features/lessons/flow";
import type { StageId } from "@/features/lessons/interfaces/story.interface";
import { Icon, SparkleIcon } from "@/shared/components/icon";

interface SidebarProps {
  isPlayground: boolean;
  stageId: StageId;
  onSelectStage: (id: StageId) => void;
}

export function Sidebar({ isPlayground, stageId, onSelectStage }: SidebarProps) {
  return (
    <aside className="sidebar">
      <p className="brand"><SparkleIcon /> Tiny GPT</p>
      <nav className="side-nav" aria-label="Pages">
        <a className={`side-link side-link--page${isPlayground ? "" : " is-on"}`} id="nav-lessons" href="#lessons" aria-current={isPlayground ? undefined : "page"}>
          <Icon name="book" /> Lessons
        </a>
        <div className="side-stages" id="side-stages" hidden={isPlayground}>
          {STAGES.map((stage) => (
            <button
              key={stage.id}
              type="button"
              className={`side-link${stage.id === stageId ? " is-selected" : ""}`}
              id={`nav-${stage.id}`}
              onClick={(event) => {
                event.stopPropagation();
                onSelectStage(stage.id);
              }}
            >
              <Icon name={stage.icon} /> {stage.title}
            </button>
          ))}
        </div>
        <a className={`side-link side-link--page${isPlayground ? " is-on" : ""}`} id="nav-playground" href="#playground" aria-current={isPlayground ? "page" : undefined}>
          <Icon name="play" /> Playground
        </a>
      </nav>
    </aside>
  );
}

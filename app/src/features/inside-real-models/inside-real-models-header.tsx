import type { ModeId, RealModel } from "@/features/inside-real-models/interfaces/model-part.interface";
import { count } from "@/shared/utils/format";
import { MODELS } from "./models";

interface InsideRealModelsHeaderProps {
  model: RealModel;
  mode: ModeId;
  onSwitchModel: (id: string) => void;
  onSwitchMode: (mode: ModeId) => void;
}

export function InsideRealModelsHeader({ model, mode, onSwitchModel, onSwitchMode }: InsideRealModelsHeaderProps) {
  return (
    <header className="top">
      <div className="top-title">
        <h1 className="page-title">Inside real models</h1>
        <div className="mode-switch" role="group" aria-label="Mode">
          <button type="button" aria-pressed={mode === "use"} onClick={() => onSwitchMode("use")}>
            {model.modes.use.title}
          </button>
          <button type="button" aria-pressed={mode === "build"} onClick={() => onSwitchMode("build")}>
            {model.modes.build.title}
          </button>
        </div>
        <label className="model-picker">
          Model
          <select name="model" value={model.id} onChange={(event) => onSwitchModel(event.target.value)}>
            {MODELS.map((option) => (
              <option key={option.id} value={option.id}>
                {option.title}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="top-stats">
        <span>{`${count(model.root.facts.weightCount)} numbers`}</span>
      </p>
    </header>
  );
}

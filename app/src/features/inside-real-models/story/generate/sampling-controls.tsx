import { useId } from "react";
import { END_TOKEN, VOCABULARY } from "@/core/model/corpus";
import type { ModelId } from "@/features/inside-real-models/interfaces/model-part.interface";
import type { Sampling } from "@/shared/interfaces/model.interface";
import { isGreedy } from "./decode";

const TOP_K_MOST = 40;

const PRESETS: { label: string; sampling: Sampling; models: ModelId[] }[] = [
  { label: "Plain roll", sampling: { temperature: 1, topK: 0, topP: 1 }, models: ["gpt-2-small", "llama-3-8b"] },
  { label: "GPT-2 paper: top-k 40", sampling: { temperature: 1, topK: 40, topP: 1 }, models: ["gpt-2-small"] },
  { label: "Llama 3 default: temperature 0.6, top-p 0.9", sampling: { temperature: 0.6, topK: 0, topP: 0.9 }, models: ["llama-3-8b"] },
  { label: "Greedy (always the top word): temperature 0", sampling: { temperature: 0, topK: 0, topP: 1 }, models: ["gpt-2-small", "llama-3-8b"] },
];

const STOP_CHOICES = VOCABULARY.filter((word) => word !== END_TOKEN);

interface SamplingControlsProps {
  modelId: ModelId;
  sampling: Sampling;
  yourStop: string | null;
  onSampling: (sampling: Sampling) => void;
  onYourStop: (word: string | null) => void;
}

export function SamplingControls({ modelId, sampling, yourStop, onSampling, onYourStop }: SamplingControlsProps) {
  const temperatureId = useId();
  const topKId = useId();
  const topPId = useId();
  const stopId = useId();
  const isTopOnly = isGreedy(sampling);
  const isPreset = (each: Sampling) => each.temperature === sampling.temperature && each.topK === sampling.topK && each.topP === sampling.topP;

  return (
    <div className="watch-box">
      <p className="section-title">How to pick each word</p>
      <p className="card-lesson">Temperature: lower = safer, higher = more surprising. Top-k / top-p: only pick from the likeliest words.</p>
      <div className="watch-controls" role="group" aria-label="Settings published by the makers">
        {PRESETS.filter((preset) => preset.models.includes(modelId)).map((preset) => (
          <button key={preset.label} type="button" className={`word-chip${isPreset(preset.sampling) ? " is-picked" : ""}`} aria-pressed={isPreset(preset.sampling)} onClick={() => onSampling(preset.sampling)}>
            {preset.label}
          </button>
        ))}
      </div>
      <div className="sampling-grid">
        <label htmlFor={temperatureId}>{`Temperature ${sampling.temperature.toFixed(1)}${isTopOnly ? ": always the top word" : ""}`}</label>
        <input id={temperatureId} type="range" min={0} max={2} step={0.1} value={sampling.temperature} onChange={(event) => onSampling({ ...sampling, temperature: Number(event.target.value) })} />
        <label htmlFor={topKId}>{`Top-k ${sampling.topK === 0 ? "off" : `${sampling.topK} words`}`}</label>
        <input id={topKId} type="range" min={0} max={TOP_K_MOST} step={1} value={sampling.topK} disabled={isTopOnly} onChange={(event) => onSampling({ ...sampling, topK: Number(event.target.value) })} />
        <label htmlFor={topPId}>{`Top-p ${sampling.topP === 1 ? "off" : sampling.topP.toFixed(2)}`}</label>
        <input id={topPId} type="range" min={0.05} max={1} step={0.05} value={sampling.topP} disabled={isTopOnly} onChange={(event) => onSampling({ ...sampling, topP: Number(event.target.value) })} />
        <label htmlFor={stopId}>Also stop at (your own stop word)</label>
        <select id={stopId} className="watch-select" value={yourStop ?? ""} onChange={(event) => onYourStop(event.target.value || null)}>
          <option value="">none</option>
          {STOP_CHOICES.map((word) => (
            <option key={word} value={word}>
              {word}
            </option>
          ))}
        </select>
      </div>
      <p className="card-lesson">{`Temperature divides every score before softmax: below 1 the top words grow, above 1 the chances even out. Top-k keeps only the k likeliest words; Tiny GPT's list has ${VOCABULARY.length}, so 40 keeps them all. Top-p keeps the fewest top words whose chances add up to p or more. The kept chances are scaled back up to 100%.`}</p>
    </div>
  );
}

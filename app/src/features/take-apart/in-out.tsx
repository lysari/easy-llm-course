import type { RealFacts } from "@/features/take-apart/interfaces/model-part.interface";

export function InOut({ facts, isDetailed }: { facts: RealFacts; isDetailed: boolean }) {
  return (
    <p className="in-out">
      <span className="in-out-chip">{isDetailed ? (facts.inputDetail ?? facts.input) : facts.input}</span>
      <span className="in-out-to">
        <span className="in-out-arrow">→</span>
        <span className="in-out-chip">{isDetailed ? (facts.outputDetail ?? facts.output) : facts.output}</span>
      </span>
    </p>
  );
}

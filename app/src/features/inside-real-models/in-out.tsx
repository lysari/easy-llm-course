import type { RealFacts } from "@/features/inside-real-models/interfaces/real-facts.interface";

export function InOut({ facts }: { facts: RealFacts }) {
  return (
    <p className="in-out">
      <span className="in-out-chip">{facts.input}</span>
      <span className="in-out-to">
        <span className="in-out-arrow">→</span>
        <span className="in-out-chip">{facts.output}</span>
      </span>
    </p>
  );
}

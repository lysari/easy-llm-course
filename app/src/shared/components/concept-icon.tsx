import { useId } from "react";
import type { ConceptName } from "@/shared/interfaces/concept.interface";
import { CONCEPT_SHAPES, type ConceptBadge } from "./concept-shapes";
import "./concept-icon.css";

type ConceptIconSize = "nav" | "chip" | "card" | "drawer";

const BADGE_MARKS: Record<ConceptBadge, string> = {
  plus: "M18.5 16v5M16 18.5h5",
  check: "m16.2 18.6 1.6 1.6 2.9-3.2",
  repeat: "M20.6 17.6a2.3 2.3 0 0 0-4.2-.4M16.4 19.4a2.3 2.3 0 0 0 4.2.4M16.3 15.8v1.6h1.6M20.7 21.2v-1.6h-1.6",
};

export function ConceptIcon({ name, size }: { name: ConceptName; size: ConceptIconSize }) {
  const id = useId();
  const shape = CONCEPT_SHAPES[name];
  return (
    <svg className={`concept-icon concept-icon--${size} concept-icon--${shape.family}`} viewBox="0 0 24 24" aria-hidden="true">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" className="ci-stop-top" />
          <stop offset="1" className="ci-stop-bottom" />
        </linearGradient>
      </defs>
      {shape.draw(`url(#${id})`)}
      {shape.badge && (
        <>
          <circle cx="18.5" cy="18.5" r="5.2" className={`ci-badge ci-badge--${shape.badge}`} strokeWidth="1.5" />
          <path d={BADGE_MARKS[shape.badge]} className="ci-badge-mark" strokeWidth="1.5" />
        </>
      )}
    </svg>
  );
}

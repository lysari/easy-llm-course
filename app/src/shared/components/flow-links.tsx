import type { LinkDrawing } from "@/shared/interfaces/link.interface";
import "./flow-links.css";

export const STAGGER_MS = 70;

export function FlowLinks({ id, drawing }: { id: string; drawing: LinkDrawing | null }) {
  return (
    <svg className="links" id={id} aria-hidden="true">
      {drawing?.paths.map((path, i) => (
        <path key={path.id} className="link-path" pathLength={1} style={{ animationDelay: `${i * STAGGER_MS}ms` }} d={path.d} />
      ))}
      {drawing?.dots.map((dot) => (
        <circle key={dot.id} className="link-dot" cx={dot.x} cy={dot.y} r={4} />
      ))}
    </svg>
  );
}

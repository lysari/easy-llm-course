import type { LinkDrawing } from "@/shared/interfaces/link.interface";
import "./flow-links.css";

export const STAGGER_MS = 70;

interface FlowLinksProps {
  id: string;
  drawing: LinkDrawing | null;
  activeIds: string[];
}

export function FlowLinks({ id, drawing, activeIds }: FlowLinksProps) {
  const paths = drawing?.paths ?? [];
  const isActive = (fromId: string) => activeIds.includes(fromId);
  const strokeOf = (pathId: string) => ({ stroke: `url(#${id}-gradient-${pathId})` });
  const delayOf = (i: number) => ({ animationDelay: `${i * STAGGER_MS}ms` });
  return (
    <svg className="links" id={id} aria-hidden="true">
      <defs>
        {paths.map(
          (path) =>
            isActive(path.fromId) && (
              <linearGradient key={path.id} id={`${id}-gradient-${path.id}`} gradientUnits="userSpaceOnUse" x1={path.startX} y1={0} x2={path.endX} y2={0}>
                <stop className="link-stop-indigo" offset={0} />
                <stop className="link-stop-sky" offset={0.38} />
                <stop className="link-stop-lavender" offset={0.64} />
                <stop className="link-stop-pink" offset={1} />
              </linearGradient>
            ),
        )}
      </defs>
      {paths.map((path, i) => (
        <path key={path.id} className="link-path" pathLength={1} style={isActive(path.fromId) ? { ...delayOf(i), ...strokeOf(path.id) } : delayOf(i)} d={path.d} />
      ))}
      {paths.map((path, i) => isActive(path.fromId) && <path key={path.id} className="link-glow" pathLength={1} style={{ ...delayOf(i), ...strokeOf(path.id) }} d={path.d} />)}
      {drawing?.dots.map((dot) => (
        <circle key={dot.id} className={`link-dot${isActive(dot.fromId) ? " is-active" : ""}`} cx={dot.x} cy={dot.y} r={4} />
      ))}
    </svg>
  );
}

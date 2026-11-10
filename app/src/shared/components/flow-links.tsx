import type { LinkDrawing, LinkPath } from "@/shared/interfaces/link.interface";
import "./flow-links.css";

export const STAGGER_MS = 70;

interface FlowLinksProps {
  id: string;
  drawing: LinkDrawing | null;
  pathIds: string[];
  glowIds: string[];
}

export function FlowLinks({ id, drawing, pathIds, glowIds }: FlowLinksProps) {
  const paths = drawing?.paths ?? [];
  const isGradient = (path: LinkPath) => pathIds.includes(path.fromId) && pathIds.includes(path.toId);
  const isGlowing = (path: LinkPath) => isGradient(path) && glowIds.includes(path.toId);
  const strokeOf = (path: LinkPath) => ({ stroke: `url(#${id}-gradient-${path.toId})` });
  const delayOf = (i: number) => ({ animationDelay: `${i * STAGGER_MS}ms` });
  return (
    <svg className="links" id={id} aria-hidden="true">
      <defs>
        {paths.map(
          (path) =>
            isGradient(path) && (
              <linearGradient key={path.toId} id={`${id}-gradient-${path.toId}`} gradientUnits="userSpaceOnUse" x1={path.startX} y1={0} x2={path.endX} y2={0}>
                <stop className="link-stop-indigo" offset={0} />
                <stop className="link-stop-sky" offset={0.38} />
                <stop className="link-stop-lavender" offset={0.64} />
                <stop className="link-stop-pink" offset={1} />
              </linearGradient>
            ),
        )}
      </defs>
      {paths.map((path, i) => (
        <path key={path.toId} className="link-path" pathLength={1} style={delayOf(i)} d={path.d} />
      ))}
      {paths.map((path) => isGradient(path) && <path key={path.toId} className="link-path" pathLength={1} style={strokeOf(path)} d={path.d} />)}
      {paths.map((path) => isGlowing(path) && <path key={path.toId} className="link-glow" pathLength={1} style={strokeOf(path)} d={path.d} />)}
      {drawing?.dots.map((dot) => (
        <circle key={dot.id} className={`link-dot${paths.some((path) => path.fromId === dot.fromId && isGradient(path)) ? " is-active" : ""}`} cx={dot.x} cy={dot.y} r={4} />
      ))}
    </svg>
  );
}

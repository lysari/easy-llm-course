import type { ModelPart } from "@/features/take-apart/interfaces/model-part.interface";
import { titleOf } from "./part-card";

export function Breadcrumb({ path, onShow }: { path: ModelPart[]; onShow: (id: string) => void }) {
  const [root, ...parts] = path;
  const last = parts.length - 1;
  return (
    <nav className="breadcrumb" aria-label="Selected path">
      <ol>
        <li aria-current={last < 0 ? "location" : undefined}>{root.title}</li>
        {parts.map((part, i) => (
          <li key={part.id}>
            <button type="button" aria-current={i === last ? "location" : undefined} onClick={() => onShow(part.id)}>
              {titleOf(part, root)}
            </button>
          </li>
        ))}
      </ol>
    </nav>
  );
}

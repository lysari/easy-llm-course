import { useEffect, useRef, useState } from "react";
import { Icon } from "./icon";

const COPIED_MS = 1500;

interface CopyButtonProps {
  text: string;
  label: string;
}

export function CopyButton({ text, label }: CopyButtonProps) {
  const [isCopied, setIsCopied] = useState(false);
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      return;
    }
    setIsCopied(true);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setIsCopied(false), COPIED_MS);
  }

  return (
    <span className="copy">
      <span className="copy-status" aria-live="polite">{isCopied ? "Copied" : ""}</span>
      <button type="button" className={`copy-button${isCopied ? " is-copied" : ""}`} aria-label={label} onClick={copy}>
        <span className="copy-icon" key={isCopied ? "check" : "copy"}>
          <Icon name={isCopied ? "check" : "copy"} />
        </span>
      </button>
    </span>
  );
}

import { useId } from "react";

export function Logo() {
  const id = useId();
  return (
    <svg className="logo" viewBox="0 0 24 24" aria-hidden="true">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" className="logo-stop-top" />
          <stop offset="1" className="logo-stop-bottom" />
        </linearGradient>
      </defs>
      <path d="M1.5 7a1.5 1.5 0 0 1 1.5-1.5h18A1.5 1.5 0 0 1 22.5 7v12.5A1.5 1.5 0 0 1 21 21h-6.4a2.6 2.6 0 0 1-5.2 0H3a1.5 1.5 0 0 1-1.5-1.5z" className="logo-cover" />
      <path d="M12 6.6C9.6 4.7 6.6 4 3.3 4.3v14.9c3.3-.2 6.3.5 8.7 2.3z" fill={`url(#${id})`} />
      <path d="M12 6.6c2.4-1.9 5.4-2.6 8.7-2.3v14.9c-3.3-.2-6.3.5-8.7 2.3z" className="logo-page" />
      <rect x="5" y="9.3" width="5" height="2.8" rx="1.1" className="logo-token-text" />
      <rect x="14" y="8.8" width="5" height="2.8" rx="1.1" className="logo-token-data" />
      <rect x="14" y="13.3" width="3.6" height="2.8" rx="1.1" className="logo-token-output" />
    </svg>
  );
}

import "./icon.css";

const ICON_PATHS = {
  repeat: "M3 7a5 5 0 0 1 9-2l1 1 M13 3v3h-3 M13 9a5 5 0 0 1-9 2l-1-1 M3 13v-3h3",
  chevron: "M4.500 6.500 8 10l3.500-3.500",
  check: "M3.500 8.500 6.500 11.500 12.500 4.500",
  close: "M4.500 4.500l7 7 M11.500 4.500l-7 7",
  copy: "M6 6h7v7H6z M10 6V3H3v7h3",
  plus: "M8 3.500v9 M3.500 8h9",
  trash: "M3 4.500h10 M6.500 4.500V3h3v1.500 M4.500 4.500l.5 8.500h6l.5-8.500",
};

export type IconName = keyof typeof ICON_PATHS;

export function Icon({ name }: { name: IconName }) {
  return (
    <svg className={`icon icon--${name}`} width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={ICON_PATHS[name]} />
    </svg>
  );
}

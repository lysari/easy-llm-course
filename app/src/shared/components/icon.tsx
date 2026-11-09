const ICON_PATHS: Record<string, string> = {
  line: "M3 13 L13 3 M3 3v10h10",
  gauge: "M3 11a5 5 0 0 1 10 0 M8 11l3-4",
  down: "M8 2v10 M4 8l4 4 4-4",
  network: "M3 8h2 M11 8h2 M5 8l3-4 3 4-3 4z",
  text: "M3 4h10 M3 8h10 M3 12h6",
  hash: "M6 2 5 14 M11 2l-1 12 M2.5 6h11 M2 10h11",
  question: "M5.5 6a2.5 2.5 0 1 1 3.6 2.2C8.3 8.7 8 9.2 8 10 M8 12.5v.5",
  grid: "M3 3h4v4H3z M9 3h4v4H9z M3 9h4v4H3z M9 9h4v4H9z",
  order: "M3 4h2 M3 8h2 M3 12h2 M8 4h5 M8 8h5 M8 12h5",
  eye: "M1.5 8S4 3.5 8 3.5 14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8z M8 6.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3z",
  merge: "M3 3c0 4 5 3 5 7v3 M13 3c0 4-5 3-5 7",
  repeat: "M3 7a5 5 0 0 1 9-2l1 1 M13 3v3h-3 M13 9a5 5 0 0 1-9 2l-1-1 M3 13v-3h3",
  percent: "M12 4 4 12 M5 3.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3z M11 9.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3z",
  play: "M5 3l8 5-8 5z",
  back: "M7 4 3 8l4 4 M3 8h10",
  sliders: "M4 2v12 M12 2v12 M8 2v12 M2.5 10h3 M6.5 5h3 M10.5 9h3",
  dice: "M3 3h10v10H3z M6 6v.5 M10 10v.5 M6 10v.5 M10 6v.5",
  pen: "M3 13l1-3 7-7 2 2-7 7z",
  book: "M8 4.500C6.800 3.500 5 3 2.500 3v9.500c2.500 0 4.300.5 5.500 1.500 1.200-1 3-1.500 5.500-1.500V3C11 3 9.200 3.500 8 4.500z M8 4.500V14",
  chevron: "M4.500 6.500 8 10l3.500-3.500",
  check: "M3.500 8.500 6.500 11.500 12.500 4.500",
  close: "M4.500 4.500l7 7 M11.500 4.500l-7 7",
  copy: "M6 6h7v7H6z M10 6V3H3v7h3",
  plus: "M8 3.500v9 M3.500 8h9",
  trash: "M3 4.500h10 M6.500 4.500V3h3v1.500 M4.500 4.500l.5 8.500h6l.5-8.500",
  layers: "M8 2.500 14 6l-6 3.500L2 6z M2 10l6 3.500 6-3.500",
};

export function Icon({ name }: { name: string }) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={ICON_PATHS[name]} />
    </svg>
  );
}

export function SparkleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor" aria-hidden="true">
      <path d="M8 1c.7 4 2 5.3 6 6-4 .7-5.3 2-6 6-.7-4-2-5.3-6-6 4-.7 5.3-2 6-6z" />
      <path d="M14 11c.3 1.700.8 2.200 2.500 2.500-1.700.3-2.200.8-2.500 2.500-.3-1.700-.8-2.200-2.500-2.500 1.700-.3 2.200-.8 2.500-2.500z" opacity=".6" />
    </svg>
  );
}

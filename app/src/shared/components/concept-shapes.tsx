import type { ReactNode } from "react";
import type { ConceptFamily, ConceptName } from "@/shared/interfaces/concept.interface";

export type ConceptBadge = "plus" | "check" | "repeat";

interface ConceptShape {
  family: ConceptFamily;
  badge?: ConceptBadge;
  draw: (body: string) => ReactNode;
}

const page = (body: string) => (
  <>
    <path d="M6.5 2.5h8l5.5 5.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 19V5a2.5 2.5 0 0 1 2.5-2.5z" fill={body} />
    <path d="M14.5 2.5v4a1.5 1.5 0 0 0 1.5 1.5h4z" className="ci-light" />
    <rect x="7" y="11.5" width="9" height="1.7" rx="0.85" className="ci-deep" />
    <rect x="7" y="15" width="6" height="1.7" rx="0.85" className="ci-deep" />
  </>
);

const merge = (body: string) => (
  <>
    <rect x="2.5" y="3" width="8.5" height="6" rx="2" className="ci-light" />
    <rect x="13" y="3" width="8.5" height="6" rx="2" className="ci-light" />
    <path d="M6.8 9.5 12 13.5l5.2-4" className="ci-line" strokeWidth="1.8" />
    <rect x="4" y="14.5" width="16" height="6.5" rx="2" fill={body} />
  </>
);

const pen = (body: string) => (
  <>
    <path d="M15.3 3.6a2 2 0 0 1 2.8 0l2.3 2.3a2 2 0 0 1 0 2.8L9.2 19.9l-5.1 1.2a.6.6 0 0 1-.7-.7l1.2-5.1z" fill={body} />
    <path d="m13.2 5.7 2.1-2.1a2 2 0 0 1 2.8 0l2.3 2.3a2 2 0 0 1 0 2.8l-2.1 2.1z" className="ci-light" />
    <path d="M4.6 15.4 3.4 20.4a.6.6 0 0 0 .7.7l5-1.2z" className="ci-deep" />
  </>
);

const medal = (body: string) => (
  <>
    <path d="m7.6 13.4-2.2 7.4 3.1-1.2 1.9 2.6 1.6-6.7zm8.8 0 2.2 7.4-3.1-1.2-1.9 2.6-1.6-6.7z" className="ci-deep" />
    <circle cx="12" cy="9.5" r="7" fill={body} />
    <path d="m12 5.4 1.3 2.6 2.9.4-2.1 2 .5 2.9-2.6-1.4-2.6 1.4.5-2.9-2.1-2 2.9-.4z" className="ci-light" />
  </>
);

const shield = (body: string) => (
  <>
    <path d="M11.4 2.7a1.6 1.6 0 0 1 1.2 0l6.9 2.6a1.6 1.6 0 0 1 1 1.5V11c0 5-3.4 8.6-8 10.4a1.4 1.4 0 0 1-1 0C6.9 19.6 3.5 16 3.5 11V6.8a1.6 1.6 0 0 1 1-1.5z" fill={body} />
    <path d="M12 2.6c.2 0 .4 0 .6.1l6.9 2.6a1.6 1.6 0 0 1 1 1.5V11c0 5-3.4 8.6-8 10.4a1.4 1.4 0 0 1-.5.1z" className="ci-light" />
  </>
);

const sheet = (body: string) => (
  <>
    <rect x="2.5" y="3.5" width="19" height="17" rx="3" fill={body} />
    <path d="M2.5 6.5a3 3 0 0 1 3-3h13a3 3 0 0 1 3 3V8h-19z" className="ci-deep" />
  </>
);

const axes = <path d="M4 3.5v15a1.5 1.5 0 0 0 1.5 1.5h15" className="ci-line" strokeWidth="1.8" />;

export const CONCEPT_SHAPES: Record<ConceptName, ConceptShape> = {
  documents: {
    family: "data",
    draw: (body) => (
      <>
        <rect x="8.5" y="1.5" width="12.5" height="16" rx="2.5" className="ci-deep" />
        <path d="M5 6h6.5l4.5 4.5V20a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2z" fill={body} />
        <path d="M11.5 6v3a1.5 1.5 0 0 0 1.5 1.5h3z" className="ci-light" />
        <rect x="5.5" y="13.5" width="7.5" height="1.6" rx="0.8" className="ci-deep" />
        <rect x="5.5" y="17" width="5" height="1.6" rx="0.8" className="ci-deep" />
      </>
    ),
  },
  "add-document": { family: "data", badge: "plus", draw: page },
  filter: {
    family: "data",
    draw: (body) => (
      <>
        <path d="M3.5 6.5h17l-6.3 7.6H9.8z" fill={body} stroke={body} strokeWidth="1.5" strokeLinejoin="round" />
        <path d="M9.8 14.1h4.4V20a1 1 0 0 1-.6.9l-2.6 1.2a.9.9 0 0 1-1.2-.8z" className="ci-deep" />
        <ellipse cx="12" cy="6" rx="8.8" ry="2.6" className="ci-light" />
      </>
    ),
  },
  duplicate: {
    family: "data",
    draw: (body) => (
      <>
        <path d="M10.5 1.5H16l4.5 4.5v9.5a2 2 0 0 1-2 2h-8a2 2 0 0 1-2-2v-12a2 2 0 0 1 2-2z" className="ci-light" />
        <path d="M5.5 6.5H11l4.5 4.5v9.5a2 2 0 0 1-2 2h-8a2 2 0 0 1-2-2v-12a2 2 0 0 1 2-2z" fill={body} />
        <path d="M11 6.5V10a1 1 0 0 0 1 1h3.5z" className="ci-light" />
        <rect x="6" y="14" width="6.5" height="1.6" rx="0.8" className="ci-deep" />
        <rect x="6" y="17.4" width="6.5" height="1.6" rx="0.8" className="ci-deep" />
      </>
    ),
  },
  pairs: {
    family: "data",
    draw: (body) => (
      <>
        <rect x="2" y="3.5" width="12" height="7" rx="2.2" fill={body} />
        <rect x="15.5" y="3.5" width="6.5" height="7" rx="2.2" className="ci-deep" />
        <rect x="2" y="13.5" width="12" height="7" rx="2.2" className="ci-light" />
        <rect x="15.5" y="13.5" width="6.5" height="7" rx="2.2" className="ci-deep" />
      </>
    ),
  },
  lessons: {
    family: "text",
    draw: (body) => (
      <>
        <path d="M6 11.5v4.3c0 1.8 2.7 3.2 6 3.2s6-1.4 6-3.2v-4.3z" fill={body} />
        <path d="M11.3 4.2a1.6 1.6 0 0 1 1.4 0l9 4.3a.8.8 0 0 1 0 1.5l-9 4.3a1.6 1.6 0 0 1-1.4 0l-9-4.3a.8.8 0 0 1 0-1.5z" className="ci-light" />
        <path d="M20 9.3v6.2" className="ci-line" strokeWidth="1.6" />
        <circle cx="20" cy="16.6" r="1.6" className="ci-deep" />
      </>
    ),
  },
  tokens: {
    family: "text",
    draw: (body) => (
      <>
        <rect x="2" y="4" width="9" height="6.5" rx="2" fill={body} />
        <rect x="12.5" y="4" width="9.5" height="6.5" rx="2" className="ci-light" />
        <rect x="2" y="13.5" width="6" height="6.5" rx="2" className="ci-light" />
        <rect x="9.5" y="13.5" width="12.5" height="6.5" rx="2" fill={body} />
      </>
    ),
  },
  split: {
    family: "text",
    draw: (body) => (
      <>
        <rect x="2" y="7.5" width="8.5" height="9" rx="2.5" fill={body} />
        <rect x="13.5" y="7.5" width="8.5" height="9" rx="2.5" className="ci-light" />
        <path d="m12 6-1.8-3h3.6zm0 12 1.8 3h-3.6z" className="ci-deep" />
      </>
    ),
  },
  merge: { family: "text", draw: merge },
  "merge-round": { family: "text", badge: "repeat", draw: merge },
  "pair-count": {
    family: "text",
    draw: (body) => (
      <>
        <rect x="2" y="3.5" width="9" height="7" rx="2" fill={body} />
        <rect x="13" y="3.5" width="9" height="7" rx="2" fill={body} />
        <rect x="2" y="14" width="20" height="6.5" rx="2" className="ci-light" />
        <rect x="2" y="14" width="12.5" height="6.5" rx="2" className="ci-deep" />
      </>
    ),
  },
  "token-id": {
    family: "text",
    draw: (body) => (
      <>
        <path d="M2.5 7a3 3 0 0 1 3-3h7.6a3 3 0 0 1 2.1.9l5.4 5.4a3.4 3.4 0 0 1 0 4.8l-5.4 5.4a3 3 0 0 1-2.1.9H5.5a3 3 0 0 1-3-3z" fill={body} />
        <path d="M2.5 13.5h14.2l-1.5 1.5-5.4 5.4a3 3 0 0 1-2.1.6H5.5a3 3 0 0 1-3-3z" className="ci-deep" />
        <circle cx="15.5" cy="11" r="2" className="ci-light" />
      </>
    ),
  },
  prompt: {
    family: "text",
    draw: (body) => (
      <>
        <rect x="2" y="8" width="6" height="8" rx="2" className="ci-light" />
        <rect x="9.5" y="8" width="7" height="8" rx="2" fill={body} />
        <path d="M17.5 4.5h4M17.5 19.5h4M19.5 4.5v15" className="ci-line" strokeWidth="1.9" />
      </>
    ),
  },
  model: {
    family: "model",
    draw: (body) => (
      <>
        <path d="M8.5 2v3m3.5-3v3m3.5-3v3m-7 14v3m3.5-3v3m3.5-3v3M2 8.5h3M2 12h3m-3 3.5h3m14-7h3m-3 3.5h3m-3 3.5h3" className="ci-line" strokeWidth="1.8" />
        <rect x="4.5" y="4.5" width="15" height="15" rx="3.5" fill={body} />
        <rect x="8.5" y="8.5" width="7" height="7" rx="1.8" className="ci-light" />
      </>
    ),
  },
  layers: {
    family: "model",
    draw: (body) => (
      <>
        <path d="m12 12.5 9 4.3-9 4.4-9-4.4z" className="ci-deep ci-round" strokeWidth="1.6" />
        <path d="m12 8 9 4.3-9 4.4-9-4.4z" className="ci-light ci-round" strokeWidth="1.6" />
        <path d="m12 3.2 9 4.3-9 4.4-9-4.4z" fill={body} stroke={body} strokeWidth="1.6" strokeLinejoin="round" />
      </>
    ),
  },
  layer: {
    family: "model",
    draw: (body) => (
      <>
        <path d="M3 10v4.5l9 4.5v-4.5z" fill={body} stroke={body} strokeWidth="1.6" strokeLinejoin="round" />
        <path d="M21 10v4.5L12 19v-4.5z" className="ci-deep ci-round" strokeWidth="1.6" />
        <path d="m12 5.5 9 4.5-9 4.5L3 10z" className="ci-light ci-round" strokeWidth="1.6" />
      </>
    ),
  },
  embedding: {
    family: "model",
    draw: (body) => (
      <>
        {sheet(body)}
        <path d="M4.5 9.6h4.3v2.6H4.5zm5.35 0h4.3v2.6h-4.3zm5.35 0h4.3v2.6h-4.3zM4.5 17h4.3v2H4.5zm5.35 0h4.3v2h-4.3zm5.35 0h4.3v2h-4.3z" className="ci-light" />
        <path d="M4.5 13.3h4.3v2.6H4.5zm5.35 0h4.3v2.6h-4.3zm5.35 0h4.3v2.6h-4.3z" className="ci-deep" />
      </>
    ),
  },
  position: {
    family: "model",
    draw: (body) => (
      <>
        {sheet(body)}
        <path d="M4.5 9.6h2.8v2.6H4.5zm0 3.7h2.8v2.6H4.5zm0 3.7h2.8v2H4.5z" className="ci-deep" />
        <path d="M8.5 9.6h11v2.6h-11zm0 3.7h11v2.6h-11zm0 3.7h11v2h-11z" className="ci-light" />
      </>
    ),
  },
  vector: {
    family: "model",
    draw: (body) => (
      <>
        <path d="M4 7H2.5v10H4m16-10h1.5v10H20" className="ci-line" strokeWidth="1.8" />
        <rect x="5" y="8.5" width="3" height="7" rx="1" fill={body} />
        <rect x="8.7" y="8.5" width="3" height="7" rx="1" className="ci-light" />
        <rect x="12.4" y="8.5" width="3" height="7" rx="1" fill={body} />
        <rect x="16.1" y="8.5" width="3" height="7" rx="1" className="ci-light" />
      </>
    ),
  },
  rope: {
    family: "model",
    draw: (body) => (
      <>
        <path d="M13.5 18.5A8.5 8.5 0 0 0 11 12.6" className="ci-light-line" strokeWidth="2" />
        <path d="M5 18.5h14" className="ci-line" strokeWidth="2.4" />
        <path d="m5 18.5 9.6-10.6" fill="none" stroke={body} strokeWidth="2.6" strokeLinecap="round" />
        <circle cx="19.5" cy="18.5" r="2.2" className="ci-deep" />
        <circle cx="15.5" cy="7" r="2.6" fill={body} />
      </>
    ),
  },
  attention: {
    family: "model",
    draw: (body) => (
      <>
        <path d="M19.2 13.5C19.2 5.5 4.8 5.5 4.8 13.5M19.2 13.5c0-4.5-7.2-4.5-7.2 0" className="ci-line" strokeWidth="1.8" />
        <rect x="2" y="15" width="5.6" height="5.5" rx="1.8" className="ci-light" />
        <rect x="9.2" y="15" width="5.6" height="5.5" rx="1.8" className="ci-light" />
        <rect x="16.4" y="15" width="5.6" height="5.5" rx="1.8" fill={body} />
      </>
    ),
  },
  head: {
    family: "model",
    draw: (body) => (
      <>
        <path d="M1.8 12C5 6.4 19 6.4 22.2 12 19 17.6 5 17.6 1.8 12z" className="ci-light" />
        <circle cx="12" cy="12" r="4.4" fill={body} />
        <circle cx="12" cy="12" r="2" className="ci-deep" />
      </>
    ),
  },
  query: {
    family: "model",
    draw: (body) => (
      <>
        <path d="m15.5 15.5 5 5" className="ci-line" strokeWidth="3.4" />
        <circle cx="10" cy="10" r="5.6" className="ci-light" />
        <circle cx="10" cy="10" r="6.6" fill="none" stroke={body} strokeWidth="2.6" />
      </>
    ),
  },
  key: {
    family: "model",
    draw: (body) => (
      <>
        <path d="M11.5 10.6h9a1.4 1.4 0 0 1 0 2.8h-1v2.6a1.2 1.2 0 0 1-2.4 0v-2.6h-1.2v1.6a1.2 1.2 0 0 1-2.4 0v-1.6h-2z" className="ci-deep" />
        <circle cx="7.5" cy="12" r="5.5" fill={body} />
        <circle cx="7" cy="12" r="2.1" className="ci-light" />
      </>
    ),
  },
  value: {
    family: "model",
    draw: (body) => (
      <>
        <rect x="3.5" y="8" width="17" height="13" rx="2.5" fill={body} />
        <rect x="2.5" y="3.5" width="19" height="5.5" rx="2" className="ci-light" />
        <rect x="10.8" y="3.5" width="2.4" height="17.5" className="ci-deep" />
      </>
    ),
  },
  "kv-group": {
    family: "model",
    draw: (body) => (
      <>
        <rect x="3" y="12" width="18" height="9.5" rx="2.5" className="ci-light" />
        <rect x="10.9" y="12" width="2.2" height="9.5" className="ci-deep" />
        <path d="M10.5 5.6h9a1.3 1.3 0 0 1 0 2.6h-.9v1.6a1.1 1.1 0 0 1-2.2 0V8.2h-5.9z" fill={body} />
        <circle cx="7" cy="6.9" r="4.4" fill={body} />
        <circle cx="6.7" cy="6.9" r="1.7" className="ci-deep" />
      </>
    ),
  },
  "feed-forward": {
    family: "model",
    draw: (body) => (
      <>
        <path d="M5 8 12 4.8M5 8l7 7.2M5 16l7-11.2M5 16l7-.8M12 4.8 19 10m-7 5.2L19 10" className="ci-line" strokeWidth="1.5" />
        <circle cx="5" cy="8" r="2.6" className="ci-light" />
        <circle cx="5" cy="16" r="2.6" className="ci-light" />
        <circle cx="12" cy="4.8" r="2.6" fill={body} />
        <circle cx="12" cy="15.2" r="2.6" fill={body} />
        <circle cx="19" cy="10" r="2.9" className="ci-deep" />
      </>
    ),
  },
  expand: {
    family: "model",
    draw: (body) => (
      <>
        <path d="M7 8.5 16 3v18l-9-5.5z" className="ci-light" />
        <rect x="3" y="8.5" width="4" height="3.2" rx="1" className="ci-deep" />
        <rect x="3" y="12.3" width="4" height="3.2" rx="1" className="ci-deep" />
        <path d="M16 3h4a1 1 0 0 1 1 1v3.3h-5zm0 4.6h5v4.2h-5zm0 4.5h5v4.3h-5zm0 4.6h5V20a1 1 0 0 1-1 1h-4z" fill={body} />
      </>
    ),
  },
  contract: {
    family: "model",
    draw: (body) => (
      <>
        <path d="M17 8.5 8 3v18l9-5.5z" className="ci-light" />
        <rect x="17" y="8.5" width="4" height="3.2" rx="1" className="ci-deep" />
        <rect x="17" y="12.3" width="4" height="3.2" rx="1" className="ci-deep" />
        <path d="M8 3H4a1 1 0 0 0-1 1v3.3h5zm0 4.6H3v4.2h5zm0 4.5H3v4.3h5zm0 4.6H3V20a1 1 0 0 0 1 1h4z" fill={body} />
      </>
    ),
  },
  gate: {
    family: "model",
    draw: (body) => (
      <>
        <rect x="7" y="7" width="10" height="3" rx="1" className="ci-light" />
        <rect x="7" y="13.5" width="10" height="3" rx="1" className="ci-light" />
        <rect x="2.5" y="3" width="5" height="18" rx="2" fill={body} />
        <rect x="16.5" y="3" width="5" height="18" rx="2" fill={body} />
        <circle cx="19" cy="12" r="1.3" className="ci-deep" />
      </>
    ),
  },
  activation: {
    family: "model",
    draw: (body) => (
      <>
        {axes}
        <path d="M7 15.5c4.8 0 5.6-9 12.5-9" fill="none" stroke={body} strokeWidth="2.8" strokeLinecap="round" />
        <circle cx="13" cy="11" r="1.9" className="ci-light" />
      </>
    ),
  },
  norm: {
    family: "model",
    draw: (body) => (
      <>
        <rect x="2" y="3.5" width="20" height="1.9" rx="0.95" className="ci-deep" />
        <rect x="3.5" y="7.5" width="4.6" height="13" rx="1.6" fill={body} />
        <rect x="9.7" y="7.5" width="4.6" height="13" rx="1.6" className="ci-light" />
        <rect x="15.9" y="7.5" width="4.6" height="13" rx="1.6" fill={body} />
      </>
    ),
  },
  mix: {
    family: "model",
    draw: (body) => (
      <>
        <circle cx="5" cy="6" r="2" className="ci-light" />
        <circle cx="12" cy="5.5" r="3.3" className="ci-deep" />
        <circle cx="19" cy="6" r="1.5" className="ci-light" />
        <path d="M3 11.5h18v5a4.5 4.5 0 0 1-4.5 4.5h-9A4.5 4.5 0 0 1 3 16.5z" fill={body} />
        <path d="M3 11.5h18v1.5c-2.2 1.6-4 1.6-6 0s-3.8-1.6-6 0-4 1.6-6 0z" className="ci-light" />
      </>
    ),
  },
  cache: {
    family: "model",
    draw: (body) => (
      <>
        <path d="M3.5 5.5v13c0 1.9 3.8 3.5 8.5 3.5s8.5-1.6 8.5-3.5v-13z" fill={body} />
        <path d="M3.5 11.5c0 1.9 3.8 3.5 8.5 3.5s8.5-1.6 8.5-3.5v1.8c0 1.9-3.8 3.5-8.5 3.5s-8.5-1.6-8.5-3.5z" className="ci-deep" />
        <ellipse cx="12" cy="5.5" rx="8.5" ry="3.5" className="ci-light" />
      </>
    ),
  },
  experts: {
    family: "model",
    draw: (body) => (
      <>
        <path d="M12 7.5 4.8 14M12 7.5V14m0-6.5 7.2 6.5" className="ci-line" strokeWidth="1.6" />
        <circle cx="12" cy="5" r="3" className="ci-deep" />
        <rect x="2" y="14" width="5.6" height="7" rx="1.8" className="ci-light" />
        <rect x="9.2" y="14" width="5.6" height="7" rx="1.8" fill={body} />
        <rect x="16.4" y="14" width="5.6" height="7" rx="1.8" className="ci-light" />
      </>
    ),
  },
  "train-loop": {
    family: "train",
    draw: (body) => (
      <>
        <path d="M12 4a8 8 0 1 1-8 8" fill="none" stroke={body} strokeWidth="3.4" strokeLinecap="round" />
        <path d="M4 6.5 7.8 12H.2z" className="ci-deep ci-round" strokeWidth="1.2" />
        <circle cx="12" cy="12" r="2.8" className="ci-light" />
      </>
    ),
  },
  loss: {
    family: "train",
    draw: (body) => (
      <>
        <path d="M2.5 16.5a9.5 9.5 0 0 1 19 0v1.5A1.5 1.5 0 0 1 20 19.5H4A1.5 1.5 0 0 1 2.5 18z" fill={body} />
        <path d="M5.5 16.5a6.5 6.5 0 0 1 13 0z" className="ci-light" />
        <path d="m12 16.5 4-5" className="ci-line" strokeWidth="2.2" />
        <circle cx="12" cy="16.5" r="1.9" className="ci-deep" />
      </>
    ),
  },
  slope: {
    family: "train",
    draw: (body) => (
      <>
        {axes}
        <path d="M6.5 5c2.5 11.5 10 11.5 12.5 0" className="ci-light-line" strokeWidth="2.2" />
        <path d="m11 17 9-8" fill="none" stroke={body} strokeWidth="2.6" strokeLinecap="round" />
        <circle cx="15.6" cy="12.9" r="2" className="ci-deep" />
      </>
    ),
  },
  descent: {
    family: "train",
    draw: (body) => (
      <>
        <path d="M2 7c4.5 0 5.5 11 10 11s5.5-11 10-11v12.5a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2z" fill={body} />
        <path d="M2 7c4.5 0 5.5 11 10 11s5.5-11 10-11v2c-4 0-5 11-10 11S6 9 2 9z" className="ci-light" />
        <circle cx="6.2" cy="5.2" r="2.6" className="ci-deep" />
      </>
    ),
  },
  backprop: {
    family: "train",
    draw: (body) => (
      <>
        <path d="M5 7h14" className="ci-line" strokeWidth="1.6" />
        <circle cx="5" cy="7" r="2.8" fill={body} />
        <circle cx="12" cy="7" r="2.8" fill={body} />
        <circle cx="19" cy="7" r="2.8" className="ci-deep" />
        <path d="M21 15.2H10v-2.9a.6.6 0 0 0-1-.4l-5.6 4.5a.6.6 0 0 0 0 .9L9 21.8a.6.6 0 0 0 1-.4v-2.9h11z" className="ci-light" />
      </>
    ),
  },
  optimizer: {
    family: "train",
    draw: (body) => (
      <>
        <rect x="5.4" y="3" width="2.2" height="18" rx="1.1" className="ci-light" />
        <rect x="10.9" y="3" width="2.2" height="18" rx="1.1" className="ci-light" />
        <rect x="16.4" y="3" width="2.2" height="18" rx="1.1" className="ci-light" />
        <rect x="3.5" y="12" width="6" height="4" rx="1.6" fill={body} />
        <rect x="9" y="5.5" width="6" height="4" rx="1.6" className="ci-deep" />
        <rect x="14.5" y="14" width="6" height="4" rx="1.6" fill={body} />
      </>
    ),
  },
  scale: {
    family: "train",
    draw: (body) => (
      <>
        {axes}
        <rect x="6.5" y="14" width="3" height="4.5" rx="1" className="ci-light" />
        <rect x="11" y="10.5" width="3" height="8" rx="1" className="ci-light" />
        <rect x="15.5" y="5.5" width="3" height="13" rx="1" fill={body} />
        <path d="M7 11.5c3-3.5 6.5-6.5 13-8.5" className="ci-line" strokeWidth="1.6" />
      </>
    ),
  },
  playground: {
    family: "train",
    draw: (body) => (
      <>
        <path d="M7 6.5h10a5 5 0 0 1 4.9 4l.9 5.4a2.6 2.6 0 0 1-4.5 2.2l-2.3-2.6H7.9l-2.3 2.6a2.6 2.6 0 0 1-4.5-2.2l.9-5.4A5 5 0 0 1 7 6.5z" fill={body} />
        <path d="M6.6 9.1a.5.5 0 0 1 .5-.5h.8a.5.5 0 0 1 .5.5v1.5h1.5a.5.5 0 0 1 .5.5v.8a.5.5 0 0 1-.5.5H8.4v1.5a.5.5 0 0 1-.5.5h-.8a.5.5 0 0 1-.5-.5v-1.5H5.1a.5.5 0 0 1-.5-.5v-.8a.5.5 0 0 1 .5-.5h1.5z" className="ci-deep" />
        <circle cx="16" cy="10" r="1.4" className="ci-light" />
        <circle cx="18.4" cy="12.6" r="1.4" className="ci-light" />
      </>
    ),
  },
  chat: {
    family: "tune",
    draw: (body) => (
      <>
        <path d="M11 2.5h8.5a2.5 2.5 0 0 1 2.5 2.5v5a2.5 2.5 0 0 1-2.5 2.5H19v2.2l-3-2.2h-5A2.5 2.5 0 0 1 8.5 10V5A2.5 2.5 0 0 1 11 2.5z" className="ci-light" />
        <path d="M4.5 8h9A2.5 2.5 0 0 1 16 10.5v6a2.5 2.5 0 0 1-2.5 2.5H8.5l-3.6 2.6a.5.5 0 0 1-.8-.4V19A2.5 2.5 0 0 1 2 16.5v-6A2.5 2.5 0 0 1 4.5 8z" fill={body} />
        <circle cx="6" cy="13.5" r="1.2" className="ci-deep" />
        <circle cx="9" cy="13.5" r="1.2" className="ci-deep" />
        <circle cx="12" cy="13.5" r="1.2" className="ci-deep" />
      </>
    ),
  },
  "fine-tune": {
    family: "tune",
    badge: "check",
    draw: (body) => (
      <>
        <path d="M5.5 3h13A3 3 0 0 1 21.5 6v7a3 3 0 0 1-3 3H10l-4.2 3.4a.6.6 0 0 1-1-.5V16A3 3 0 0 1 2.5 13V6a3 3 0 0 1 3-3z" fill={body} />
        <path d="M2.5 9.5h19V13a3 3 0 0 1-3 3H10l-4.2 3.4a.6.6 0 0 1-1-.5V16A3 3 0 0 1 2.5 13z" className="ci-light" />
        <rect x="6" y="5.8" width="11" height="1.8" rx="0.9" className="ci-deep" />
      </>
    ),
  },
  preference: {
    family: "tune",
    draw: (body) => (
      <>
        <rect x="2" y="4.5" width="9" height="15" rx="2.5" className="ci-light" />
        <rect x="4.2" y="8" width="4.6" height="1.6" rx="0.8" className="ci-deep" />
        <rect x="4.2" y="11.2" width="4.6" height="1.6" rx="0.8" className="ci-deep" />
        <rect x="13" y="4.5" width="9" height="15" rx="2.5" fill={body} />
        <path d="m17.5 15.2-2.6-2.5a1.6 1.6 0 1 1 2.6-1.9 1.6 1.6 0 1 1 2.6 1.9z" className="ci-deep" />
      </>
    ),
  },
  reward: { family: "tune", draw: medal },
  rlhf: { family: "tune", badge: "repeat", draw: medal },
  "in-context": {
    family: "tune",
    draw: (body) => (
      <>
        <circle cx="12" cy="9.5" r="7" fill={body} />
        <path d="M9.6 9.6 12 12l2.4-2.4M12 12v3.5" className="ci-light-line" strokeWidth="1.6" />
        <rect x="8.8" y="16.5" width="6.4" height="2" rx="1" className="ci-deep" />
        <rect x="9.6" y="19.5" width="4.8" height="2" rx="1" className="ci-deep" />
      </>
    ),
  },
  evaluation: {
    family: "test",
    draw: (body) => (
      <>
        <rect x="3.5" y="3.5" width="17" height="18.5" rx="3" fill={body} />
        <rect x="6" y="6.5" width="12" height="13" rx="1.5" className="ci-light" />
        <rect x="8.5" y="1.8" width="7" height="4" rx="1.5" className="ci-deep" />
        <path d="m8 11 1.3 1.3L11.8 9.8M8 15.6l1.3 1.3 2.5-2.5" className="ci-line" strokeWidth="1.5" />
      </>
    ),
  },
  benchmark: {
    family: "test",
    draw: (body) => (
      <>
        <rect x="3" y="12.5" width="5" height="7.5" rx="1.5" className="ci-light" />
        <rect x="9.5" y="8" width="5" height="12" rx="1.5" fill={body} />
        <rect x="16" y="3.5" width="5" height="16.5" rx="1.5" fill={body} />
        <rect x="2" y="20" width="20" height="2" rx="1" className="ci-deep" />
      </>
    ),
  },
  "test-question": {
    family: "test",
    draw: (body) => (
      <>
        <path d="M6.5 2.5h8l5.5 5.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 19V5a2.5 2.5 0 0 1 2.5-2.5z" className="ci-light" />
        <path d="M14.5 2.5v4a1.5 1.5 0 0 0 1.5 1.5h4z" fill={body} />
        <path d="M9.5 11a2.5 2.5 0 1 1 3.6 2.2c-.8.4-1.1.9-1.1 1.8" className="ci-line" strokeWidth="2" />
        <circle cx="12" cy="18" r="1.2" className="ci-deep" />
      </>
    ),
  },
  safety: { family: "test", draw: shield },
  "safety-check": { family: "test", badge: "check", draw: shield },
  quantize: {
    family: "test",
    draw: (body) => (
      <>
        <path d="M2.5 21v-4.5H7V12h4.5V7.5H16V3h5.5v16a2 2 0 0 1-2 2z" fill={body} stroke={body} strokeWidth="1" strokeLinejoin="round" />
        <path d="M2.5 15.5h4.5V11h4.5V6.5H16V2h5.5v1H16v4.5h-4.5V12H7v4.5H2.5z" className="ci-light" />
        <path d="M3 18C9 17 14 11 21.5 5" className="ci-line" strokeWidth="1.5" />
      </>
    ),
  },
  scores: {
    family: "output",
    draw: (body) => (
      <>
        <rect x="3" y="4.5" width="3.8" height="7" rx="1.2" fill={body} />
        <rect x="8" y="12.5" width="3.8" height="5" rx="1.2" className="ci-light" />
        <rect x="13" y="2.5" width="3.8" height="9" rx="1.2" fill={body} />
        <rect x="18" y="12.5" width="3.8" height="8" rx="1.2" className="ci-light" />
        <rect x="2" y="11.1" width="20" height="1.8" rx="0.9" className="ci-deep" />
      </>
    ),
  },
  chances: {
    family: "output",
    draw: (body) => (
      <>
        <path d="M2.5 6a2 2 0 0 1 2-2h15a2 2 0 0 1 0 4h-15a2 2 0 0 1-2-2zm0 6a2 2 0 0 1 2-2h15a2 2 0 0 1 0 4h-15a2 2 0 0 1-2-2zm0 6a2 2 0 0 1 2-2h15a2 2 0 0 1 0 4h-15a2 2 0 0 1-2-2z" className="ci-light" />
        <rect x="2.5" y="4" width="15" height="4" rx="2" fill={body} />
        <rect x="2.5" y="10" width="8.5" height="4" rx="2" fill={body} />
        <rect x="2.5" y="16" width="4.5" height="4" rx="2" className="ci-deep" />
      </>
    ),
  },
  generate: { family: "output", draw: pen },
  "write-round": { family: "output", badge: "repeat", draw: pen },
  dice: {
    family: "output",
    draw: (body) => (
      <>
        <path d="M3 8.5 10 4l11 3.5-7 4.5z" className="ci-light ci-round" strokeWidth="1.6" />
        <path d="M3 8.5 14 12v10L3 18.5z" fill={body} stroke={body} strokeWidth="1.6" strokeLinejoin="round" />
        <path d="m14 12 7-4.5v10L14 22z" className="ci-deep ci-round" strokeWidth="1.6" />
        <circle cx="6" cy="12.5" r="1.3" className="ci-light" />
        <circle cx="11" cy="17.6" r="1.3" className="ci-light" />
        <circle cx="8.5" cy="15" r="1.3" className="ci-light" />
      </>
    ),
  },
  append: {
    family: "output",
    badge: "plus",
    draw: (body) => (
      <>
        <rect x="2" y="4" width="8" height="6.5" rx="2" className="ci-light" />
        <rect x="11.5" y="4" width="10.5" height="6.5" rx="2" className="ci-light" />
        <rect x="2" y="13.5" width="10" height="6.5" rx="2" fill={body} />
      </>
    ),
  },
  stop: {
    family: "output",
    draw: (body) => (
      <>
        <rect x="11" y="14" width="2" height="8" rx="1" className="ci-deep" />
        <path d="M9 1.8h6l4.2 4.2v6L15 16.2H9L4.8 12V6z" fill={body} stroke={body} strokeWidth="1.6" strokeLinejoin="round" />
        <path d="M9.7 3.6h4.6l3.1 3.1v4.6l-3.1 3.1H9.7l-3.1-3.1V6.7z" className="ci-light-line" strokeWidth="1.4" />
      </>
    ),
  },
  temperature: {
    family: "output",
    draw: (body) => (
      <>
        <rect x="9" y="2" width="6" height="14" rx="3" className="ci-light" />
        <rect x="10.6" y="8" width="2.8" height="8" rx="1.4" fill={body} />
        <circle cx="12" cy="17.5" r="4.5" fill={body} />
        <path d="M16.5 5h2M16.5 8.5h2M16.5 12h2" className="ci-line" strokeWidth="1.5" />
      </>
    ),
  },
  "top-k": {
    family: "output",
    draw: (body) => (
      <>
        <rect x="2.5" y="3.5" width="3.3" height="17" rx="1.2" fill={body} />
        <rect x="6.7" y="7.5" width="3.3" height="13" rx="1.2" fill={body} />
        <rect x="10.9" y="10.5" width="3.3" height="10" rx="1.2" className="ci-deep" />
        <rect x="15.1" y="14" width="3.3" height="6.5" rx="1.2" className="ci-muted" />
        <rect x="19.3" y="16.5" width="2.5" height="4" rx="1" className="ci-muted" />
      </>
    ),
  },
  "top-p": {
    family: "output",
    draw: (body) => (
      <>
        <circle cx="12" cy="12" r="9.5" fill={body} />
        <path d="M12 12V2.5a9.5 9.5 0 0 1 9.5 9.5z" className="ci-muted" />
        <path d="M12 12 4.6 18A9.5 9.5 0 0 1 2.5 12z" className="ci-deep" />
      </>
    ),
  },
  number: {
    family: "math",
    draw: (body) => (
      <>
        <path d="M3 4.5A1.5 1.5 0 0 1 4.5 3H8v5H3zm6.5-1.5h5v5h-5zm6.5 0h3.5A1.5 1.5 0 0 1 21 4.5V8h-5zM3 9.5h5v5H3zm13 0h5v5h-5zM3 16h5v5H4.5A1.5 1.5 0 0 1 3 19.5zm6.5 0h5v5h-5zm6.5 0h5v3.5a1.5 1.5 0 0 1-1.5 1.5H16z" className="ci-light" />
        <rect x="8.6" y="8.6" width="6.8" height="6.8" rx="1.6" fill={body} />
      </>
    ),
  },
  "multiply-add": {
    family: "math",
    draw: (body) => (
      <>
        <path d="m3.5 3.5 7 7m0-7-7 7" fill="none" stroke={body} strokeWidth="3" strokeLinecap="round" />
        <path d="M17 12.5v9m-4.5-4.5h9" className="ci-line" strokeWidth="3" />
      </>
    ),
  },
  bias: {
    family: "math",
    draw: (body) => (
      <>
        <path d="M8 6.5v11M2.5 12h11" fill="none" stroke={body} strokeWidth="3" strokeLinecap="round" />
        <path d="M18.5 20V5m-3.2 3.2 3.2-3.2 3.2 3.2" className="ci-line" strokeWidth="2.2" />
      </>
    ),
  },
  function: {
    family: "math",
    draw: (body) => (
      <>
        <path d="M18.2 14h3.6m-2.2-2.4L22 14l-2.4 2.4" className="ci-line" strokeWidth="2" />
        <path d="M6.5 2.5h11l-3 5h-5z" className="ci-light ci-round" strokeWidth="1.6" />
        <rect x="4" y="8" width="14.5" height="13" rx="3" fill={body} />
        <rect x="7" y="17" width="8.5" height="2" rx="1" className="ci-deep" />
      </>
    ),
  },
  "fit-line": {
    family: "math",
    draw: (body) => (
      <>
        {axes}
        <path d="M7 17.5 20.5 5.5" fill="none" stroke={body} strokeWidth="2.4" strokeLinecap="round" />
        <circle cx="8.5" cy="12.5" r="1.7" className="ci-light" />
        <circle cx="13" cy="15.5" r="1.7" className="ci-light" />
        <circle cx="13.5" cy="8.5" r="1.7" className="ci-light" />
        <circle cx="18.5" cy="11.5" r="1.7" className="ci-light" />
      </>
    ),
  },
  matrix: {
    family: "math",
    draw: (body) => (
      <>
        <path d="M6 3.5H3.5v17H6m12-17h2.5v17H18" className="ci-line" strokeWidth="2" />
        <rect x="6.8" y="6.8" width="4.6" height="4.6" rx="1.2" fill={body} />
        <rect x="12.6" y="6.8" width="4.6" height="4.6" rx="1.2" className="ci-light" />
        <rect x="6.8" y="12.6" width="4.6" height="4.6" rx="1.2" className="ci-light" />
        <rect x="12.6" y="12.6" width="4.6" height="4.6" rx="1.2" fill={body} />
      </>
    ),
  },
  pipeline: {
    family: "math",
    draw: (body) => (
      <>
        <path d="M10 6.5h4M6 10v2.5a2 2 0 0 0 2 2M18 10v2.5a2 2 0 0 1-2 2" className="ci-line" strokeWidth="1.8" />
        <rect x="2" y="3" width="8" height="7" rx="2" fill={body} />
        <rect x="14" y="3" width="8" height="7" rx="2" className="ci-light" />
        <rect x="8" y="14" width="8" height="7" rx="2" fill={body} />
      </>
    ),
  },
};

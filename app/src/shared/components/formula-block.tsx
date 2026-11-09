import { Fragment, type ReactNode } from "react";
import type { Formula } from "@/shared/interfaces/formula.interface";
import { CopyButton } from "./copy-button";
import "./snippet-blocks.css";

const MATH_PART = /\s+|\^|\p{L}[\p{L}\p{M}\p{N}_]*|\d+(?:\.\d+)?|./gu;
const OPERATORS = new Set(["=", "≈", "+", "−", "×", "·", "/", "→", "@"]);
const PREFIXES = new Set(["Σ", "√", "∂"]);
const LETTER = /[\p{Lu}\p{Ll}]/gu;

function isOneLetter(word: string) {
  return word.match(LETTER)?.length === 1;
}

function takePower(parts: string[], start: number): [string, number] {
  let end = start + (parts[start] === "(" ? 0 : 1);
  if (parts[end] === "(") {
    let depth = 0;
    do {
      depth += parts[end] === "(" ? 1 : parts[end] === ")" ? -1 : 0;
      end++;
    } while (depth > 0 && end < parts.length);
  }
  const power = parts.slice(start, end).join("");
  const isWrapped = parts[start] === "(" && parts[end - 1] === ")";
  return [isWrapped ? power.slice(1, -1) : power, end];
}

function mathWord(word: string, key: number): ReactNode {
  const [base, sub] = word.split(/_(.*)/s);
  if (sub && isOneLetter(base)) {
    return <Fragment key={key}><var>{base}</var><sub>{sub}</sub></Fragment>;
  }
  return isOneLetter(word) ? <var key={key}>{word}</var> : word;
}

function mathNodes(text: string): ReactNode[] {
  const parts = text.match(MATH_PART) ?? [];
  const nodes: ReactNode[] = [];
  let previous = "";
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    const isPrefix = PREFIXES.has(part) || (part === "−" && (!previous || previous === "(" || OPERATORS.has(previous)));
    if (!/^\s/.test(part)) previous = part;
    if (part === "^") {
      const [power, end] = takePower(parts, i + 1);
      nodes.push(<sup key={i}>{mathNodes(power)}</sup>);
      i = end - 1;
    } else if (/^\s/.test(part)) {
      nodes.push(" ");
    } else if (isPrefix) {
      nodes.push(<span key={i} className="math-sign">{part}</span>);
    } else if (OPERATORS.has(part)) {
      nodes.push(<span key={i} className="math-op">{part}</span>);
    } else if (/^\p{L}/u.test(part)) {
      nodes.push(mathWord(part, i));
    } else {
      nodes.push(part);
    }
  }
  return nodes;
}

export function FormulaBlock({ formula }: { formula: Formula }) {
  const legend = formula.legend ?? [];
  const text = [...formula.lines, ...legend.map(({ term, meaning }) => `${term}: ${meaning}`)].join("\n");

  return (
    <section className="section">
      <header className="snippet-head">
        <p className="section-title">Formula</p>
        <CopyButton text={text} label="Copy formula" />
      </header>
      <div className="snippet-box formula">
        {formula.lines.map((line, i) => (
          <p key={i} className="formula-line">{mathNodes(line)}</p>
        ))}
        {legend.length > 0 && (
          <dl className="formula-legend">
            {legend.map(({ term, meaning }) => (
              <Fragment key={term}>
                <dt>{mathNodes(term)}</dt>
                <dd>{meaning}</dd>
              </Fragment>
            ))}
          </dl>
        )}
      </div>
    </section>
  );
}

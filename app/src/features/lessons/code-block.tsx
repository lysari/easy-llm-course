import { parse } from "sugar-high/core";
import { tokenize as tokenizeTypeScript } from "sugar-high/lang/typescript";
import { CopyButton } from "./copy-button";
import "./snippet-blocks.css";

export function CodeBlock({ code }: { code: string }) {
  const { lines } = parse(code, { tokenize: tokenizeTypeScript });

  return (
    <section className="section">
      <header className="snippet-head">
        <p className="section-title">Code</p>
        <span className="snippet-language">TypeScript</span>
        <CopyButton text={code} label="Copy code" />
      </header>
      <pre className="snippet-box snippet-code">
        <code>
          {lines.map((line) => (
            <span key={line.index} className="code-line">
              <span className="code-line-number" aria-hidden="true">{line.index + 1}</span>
              <span>
                {line.tokens.map((token, i) => (
                  <span key={i} className={`code-token--${token.type}`}>{token.value}</span>
                ))}
              </span>
            </span>
          ))}
        </code>
      </pre>
    </section>
  );
}

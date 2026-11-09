import { percent, showWord } from "@/shared/utils/format";

interface ChanceBarsProps {
  options: { token: string; chance: number }[];
  answer: string;
  limit?: number;
}

export function ChanceBars({ options, answer, limit = 4 }: ChanceBarsProps) {
  const top = options.slice(0, limit);
  const answerOption = options.find((option) => option.token === answer);
  const shown = answerOption && !top.includes(answerOption) ? [...top, answerOption] : top;
  return (
    <div className="chances">
      {shown.map((option) => (
        <div key={option.token} className={`chance${option.token === answer ? " is-key" : ""}`}>
          <span>{showWord(option.token)}</span>
          <span className="chance-track">
            <i className="chance-fill" style={{ width: `${Math.max(1, Math.round(option.chance * 100))}%` }} />
          </span>
          <span>{percent(option.chance)}</span>
        </div>
      ))}
    </div>
  );
}

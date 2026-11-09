export function FactRows({ rows }: { rows: [string, string][] }) {
  return (
    <div className="tasks">
      {rows.map(([text, value], i) => (
        <div key={i} className="task">
          <span>{text}</span>
          <span>{value}</span>
        </div>
      ))}
    </div>
  );
}

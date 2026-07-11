function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const trainData: Array<[string, number]> = [
  ["i love this movie", 1],["what a great film", 1],["really wonderful acting", 1],
  ["great story and great cast", 1],["i enjoyed every minute", 1],["a wonderful little gem", 1],
  ["lovely and moving", 1],["great fun for everyone", 1],
  ["terrible plot and bad acting", 0],["i hated this film", 0],["what a boring mess", 0],
  ["awful from start to finish", 0],["bad script bad pacing", 0],["truly terrible movie", 0],
  ["boring and predictable", 0],["i hated every minute", 0],["an awful waste of time", 0],
  ["dull boring and bad", 0],["the worst film this year", 0],["bad acting ruined it", 0],
  ["a terrible boring slog", 0],["worst script i have seen", 0],
];
const testData: Array<[string, number]> = [
  ["a great and lovely film", 1],["i love the wonderful story", 1],
  ["boring plot terrible cast", 0],["i hated the script", 0],
  ["the worst mess this year", 0],["bad pacing and a bad ending", 0],
  ["truly a GREAT film", 1],["a WONDERFUL story indeed", 1],
  ["a BORING story and cast", 0],["the acting is TERRIBLE really", 0],
  ["DULL film for everyone", 0],["every minute is AWFUL", 0],
];
function bigrams(t: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < t.length - 1; i++) out.push(t.slice(i, i + 2));
  return out;
}
function train(norm: boolean, len: boolean) {
  const nz = (s: string) => (norm ? s.toLowerCase() : s);
  const vocab = new Map<string, number>();
  for (const [t] of trainData) for (const bg of bigrams(nz(t))) if (!vocab.has(bg)) vocab.set(bg, vocab.size);
  const dim = vocab.size + (len ? 1 : 0);
  const feat = (raw: string): number[] => {
    const f = Array<number>(dim).fill(0);
    for (const bg of bigrams(nz(raw))) { const j = vocab.get(bg); if (j !== undefined) f[j] = 1; }
    if (len) f[dim - 1] = raw.length / 30;
    return f;
  };
  const rand = mulberry32(42);
  const w = Array.from({ length: dim }, () => (rand() - 0.5) * 0.01);
  let b = 0;
  for (let e = 0; e < 150; e++)
    for (const [t, y] of trainData) {
      const x = feat(t);
      let z = b;
      for (let j = 0; j < dim; j++) z += w[j]! * (x[j] ?? 0);
      const g = 1 / (1 + Math.exp(-z)) - y;
      for (let j = 0; j < dim; j++) w[j]! -= 0.5 * g * (x[j] ?? 0);
      b -= 0.5 * g;
    }
  return { feat, w, b, dim };
}
for (const [norm, len, label] of [[true,true,"full"],[true,false,"-len"],[false,true,"-norm"]] as const) {
  const m = train(norm, len);
  let ok = 0;
  console.log("== " + label + " ==  b=" + m.b.toFixed(2));
  for (const [t, y] of testData) {
    const x = m.feat(t);
    let z = m.b;
    for (let j = 0; j < m.dim; j++) z += m.w[j]! * (x[j] ?? 0);
    const pred = z > 0 ? 1 : 0;
    if (pred === y) ok++;
    console.log(`${pred===y?"ok ":"BAD"} z=${z.toFixed(2).padStart(6)} y=${y} ${t}`);
  }
  console.log("acc", (ok / testData.length).toFixed(3));
}

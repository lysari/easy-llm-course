import type { RankedToken, Sampling, TinyGptWeights } from "@/shared/interfaces/model.interface";
import { END_TOKEN, MAX_CONTEXT, VOCABULARY } from "./corpus";
import { predictNext } from "./tiny-gpt";

const NO_SAMPLING: Sampling = { temperature: 1, topK: 0, topP: 1 };

export function pickToken(options: RankedToken[], random: () => number): RankedToken {
  let roll = random();
  for (const option of options) {
    roll -= option.chance;
    if (roll <= 0) return option;
  }
  return options[0];
}

export function nextOptions(tokens: string[], weights: TinyGptWeights): RankedToken[] {
  return predictNext(tokens.map((token) => VOCABULARY.indexOf(token)), weights);
}

export function applySampling(options: RankedToken[], { temperature, topK, topP }: Sampling): RankedToken[] {
  const toChances = (list: { option: RankedToken; raw: number }[]) => {
    const total = list.reduce((sum, item) => sum + item.raw, 0);
    return list.map(({ option, raw }) => ({ option: { ...option, chance: raw / total }, raw }));
  };
  const scaled = options.map((option) => ({ option, raw: Math.exp((option.score - options[0].score) / temperature) }));
  const kept = toChances(topK > 0 ? scaled.slice(0, topK) : scaled);
  if (topP >= 1) return kept.map(({ option }) => option);
  let running = 0;
  const nucleus = kept.filter(({ option }) => {
    const keep = running < topP;
    running += option.chance;
    return keep;
  });
  return toChances(nucleus).map(({ option }) => option);
}

export function writeText(start: string[], weights: TinyGptWeights, random: () => number, sampling = NO_SAMPLING): string[] {
  const tokens = [...start];
  while (tokens.length < MAX_CONTEXT) {
    const picked = pickToken(applySampling(nextOptions(tokens, weights), sampling), random);
    if (picked.token === END_TOKEN) break;
    tokens.push(picked.token);
  }
  return tokens;
}

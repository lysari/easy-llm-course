import type { RankedToken, TinyGptWeights } from "../interfaces/model.interface";
import { END_TOKEN, MAX_CONTEXT, VOCABULARY } from "./corpus";
import { predictNext } from "./tiny-gpt";

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

export function writeText(start: string[], weights: TinyGptWeights, random: () => number): string[] {
  const tokens = [...start];
  while (tokens.length < MAX_CONTEXT) {
    const picked = pickToken(nextOptions(tokens, weights), random);
    if (picked.token === END_TOKEN) break;
    tokens.push(picked.token);
  }
  return tokens;
}

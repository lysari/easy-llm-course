import { count, percent } from "@/shared/utils/format";

export const GPT_2_KARMA = 3;
const MIN_WORDS = 50;
const WORD_LENGTH = { min: 3, max: 10 };
const MIN_SENTENCE_LINES = 0.5;
const MIN_LETTERS = 0.8;

export interface QualityRule {
  name: string;
  measured: string;
  passes: boolean;
}

const wordsOf = (text: string) => text.match(/[\p{L}\p{M}\p{N}]+(?:['’-][\p{L}\p{M}\p{N}]+)*/gu) ?? [];

const linesOf = (text: string) =>
  text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

const shareOf = (part: number, whole: number) => (whole === 0 ? 0 : part / whole);

export function checkQuality(text: string): QualityRule[] {
  const words = wordsOf(text);
  const lines = linesOf(text);
  const averageLength = shareOf(
    words.reduce((sum, word) => sum + [...word].length, 0),
    words.length,
  );
  const sentenceLines = shareOf(lines.filter((line) => /[.!?]["”’')\]]?$/u.test(line)).length, lines.length);
  const letters = shareOf((text.match(/[\p{L}\p{M}]/gu) ?? []).length, text.replace(/\s/gu, "").length);
  return [
    { name: `At least ${MIN_WORDS} words`, measured: `${count(words.length)} words`, passes: words.length >= MIN_WORDS },
    { name: `Words ${WORD_LENGTH.min} to ${WORD_LENGTH.max} letters long on average`, measured: `${averageLength.toFixed(1)} letters`, passes: averageLength >= WORD_LENGTH.min && averageLength <= WORD_LENGTH.max },
    { name: `At least ${percent(MIN_SENTENCE_LINES)} of lines end like a sentence (. ! ?)`, measured: `${percent(sentenceLines)} of ${count(lines.length)} lines`, passes: sentenceLines >= MIN_SENTENCE_LINES },
    { name: `At least ${percent(MIN_LETTERS)} of characters are letters`, measured: percent(letters), passes: letters >= MIN_LETTERS },
  ];
}

export const keepsDocument = (rules: QualityRule[]) => rules.every((rule) => rule.passes);

export const passesKarma = (karma: number) => karma >= GPT_2_KARMA;

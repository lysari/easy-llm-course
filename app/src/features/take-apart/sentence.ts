import { END_TOKEN, MAX_CONTEXT, START_CHOICES, VOCABULARY, joinTokens } from "@/core/model/corpus";
import type { Sentence } from "@/features/take-apart/interfaces/sentence.interface";

const MAX_WORDS = MAX_CONTEXT - 1;
const SAMPLE_WORDS = START_CHOICES[0].split(" ");

export const START_TEXTS = START_CHOICES.map((choice) => joinTokens(choice.split(" ")));

const isKnown = (word: string) => word !== END_TOKEN && VOCABULARY.includes(word);

export function readSentence(text: string): Sentence {
  const pieces = text.toLowerCase().replace(/([,.])/g, " $1 ").split(/\s+/).filter(Boolean);
  const known = pieces.filter(isKnown);
  const unknown = [...new Set(pieces.filter((word) => !isKnown(word)))];
  if (known.length === 0) return { words: SAMPLE_WORDS, unknown, isSample: true, isCut: false };
  return { words: known.slice(-MAX_WORDS), unknown, isSample: false, isCut: known.length > MAX_WORDS };
}

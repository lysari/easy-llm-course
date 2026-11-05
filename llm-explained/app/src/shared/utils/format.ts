import { END_TOKEN } from "@/core/model/corpus";

/** Shows the stop word in plain language and leaves all other words as they are. */
export const showWord = (word: string) => (word === END_TOKEN ? "stop" : word);

/** Formats a 0-1 chance as a short percent, for example 74%. */
export const percent = (chance: number) => (chance < 0.01 ? "<1%" : `${Math.round(chance * 100)}%`);

/** Writes a count with thousands separators. */
export const count = (value: number) => value.toLocaleString("en-US");

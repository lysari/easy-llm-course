export const END_TOKEN = "<end>";
export const MAX_CONTEXT = 16;

export const TRAINING_LINES: { text: string; times: number }[] = [
  { text: "my dog is hungry , so i give it a bone .", times: 3 },
  { text: "my dog is hungry , so i give it a treat .", times: 1 },
  { text: "my cat is hungry , so i give it a fish .", times: 3 },
  { text: "my cat is hungry , so i give it a treat .", times: 1 },
  { text: "my dog is bored , so i give it a ball .", times: 3 },
  { text: "my dog is bored , so i give it a stick .", times: 1 },
  { text: "my cat is bored , so i give it a toy .", times: 3 },
  { text: "my cat is bored , so i give it a ball .", times: 1 },
  { text: "the dog likes the bone .", times: 1 },
  { text: "the cat likes the fish .", times: 1 },
  { text: "the dog chases the ball .", times: 1 },
  { text: "the cat chases the toy .", times: 1 },
];

export const VOCABULARY: string[] = [
  ...new Set(TRAINING_LINES.flatMap((line) => line.text.split(" "))),
  END_TOKEN,
];

export const END_ID = VOCABULARY.indexOf(END_TOKEN);

export const toSequences = (lines: { text: string; times: number }[]): number[][] =>
  lines.flatMap((line) => {
    const ids = [...line.text.split(" ").map((token) => VOCABULARY.indexOf(token)), END_ID];
    return Array.from({ length: line.times }, () => ids);
  });

export const TRAINING_SEQUENCES: number[][] = toSequences(TRAINING_LINES);

export function joinTokens(tokens: string[]): string {
  return tokens
    .filter((token) => token !== END_TOKEN)
    .reduce((text, token) => (token === "," || token === "." || !text ? text + token : `${text} ${token}`), "");
}

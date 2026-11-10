import { VOCABULARY } from "@/core/model/corpus";
import { allTables, lossAndGradients, predictNext } from "@/core/model/tiny-gpt";
import type { TinyGptWeights } from "@/shared/interfaces/model.interface";

const PUSH_STEP = 0.05;

export const LLAMA_3_BASE_WORDS = 128_000;

const LLAMA_3_SPECIAL_WORDS = ["<|begin_of_text|>", "<|end_of_text|>", "<|reserved_special_token_0|>", "<|reserved_special_token_1|>", "<|reserved_special_token_2|>", "<|reserved_special_token_3|>", "<|start_header_id|>", "<|end_header_id|>", "<|reserved_special_token_4|>", "<|eot_id|>"];

export type ChatRole = "system" | "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export interface ChatPiece {
  text: string;
  id: number | null;
}

export const specialId = (word: string) => LLAMA_3_BASE_WORDS + LLAMA_3_SPECIAL_WORDS.indexOf(word);

const special = (word: string): ChatPiece => ({ text: word, id: specialId(word) });

const header = (role: ChatRole): ChatPiece[] => [special("<|start_header_id|>"), { text: role, id: null }, special("<|end_header_id|>"), { text: "\n\n", id: null }];

export function chatPieces(messages: ChatMessage[]): ChatPiece[] {
  return [
    special("<|begin_of_text|>"),
    ...messages.flatMap((message) => [...header(message.role), { text: message.content.trim(), id: null }, special("<|eot_id|>")]),
    ...header("assistant"),
  ];
}

export function answerChance(prompt: string[], answer: string, weights: TinyGptWeights) {
  return predictNext(prompt.map((word) => VOCABULARY.indexOf(word)), weights).find((option) => option.token === answer)?.chance ?? 0;
}

export const preferenceGap = (prompt: string[], better: string, worse: string, weights: TinyGptWeights) => Math.log(answerChance(prompt, better, weights)) - Math.log(answerChance(prompt, worse, weights));

export function pushPreference(weights: TinyGptWeights, prompt: string[], better: string, worse: string, stepSize = PUSH_STEP) {
  const ids = prompt.map((word) => VOCABULARY.indexOf(word));
  lossAndGradients([[...ids, VOCABULARY.indexOf(better)]], weights);
  const betterSlopes = allTables(weights).map(({ tensor }) => tensor.grad.slice());
  lossAndGradients([[...ids, VOCABULARY.indexOf(worse)]], weights);
  allTables(weights).forEach(({ tensor }, t) => {
    for (let i = 0; i < tensor.data.length; i++) tensor.data[i] -= stepSize * (betterSlopes[t][i] - tensor.grad[i]);
  });
}

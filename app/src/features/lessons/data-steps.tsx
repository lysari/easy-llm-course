import { END_TOKEN, TRAINING_LINES, TRAINING_SEQUENCES, VOCABULARY, joinTokens } from "@/core/model/corpus";
import type { FlowStep } from "@/features/lessons/interfaces/story.interface";
import { count } from "@/shared/utils/format";
import { WordCards } from "./lesson-plots";

const tokenTotal = TRAINING_SEQUENCES.reduce((sum, sequence) => sum + sequence.length, 0);
const questionTotal = TRAINING_SEQUENCES.reduce((sum, sequence) => sum + sequence.length - 1, 0);
const FIRST_LINE = TRAINING_LINES[0].text.split(" ");

export const dataSteps: FlowStep[] = [
  {
    id: "text",
    stage: "data",
    icon: "text",
    title: "Training text",
    lesson: "Lessons 09, 14",
    goal: "Collect the text the model will learn from.",
    from: () => "Nothing",
    to: () => `${count(tokenTotal)} tokens of text`,
    status: () => "done",
    rowsTitle: `All ${TRAINING_LINES.length} kinds of sentence`,
    rows: () => TRAINING_LINES.map((line) => ({ text: joinTokens(line.text.split(" ")), value: `${line.times}×` })),
    note: () => "A hungry dog gets a bone 3 times and a treat 1 time. Watch for that later.",
    why: "A model only knows what is in its text. GPT-1 read about 800 million words. Ours reads a few sentences, so it can only memorise them.",
  },
  {
    id: "tokenizer",
    stage: "data",
    icon: "hash",
    title: "Tokenizer",
    lesson: "Lessons 09, 18",
    goal: "Turn text into numbers, and back again.",
    from: () => "Text",
    to: () => `${VOCABULARY.length} token IDs`,
    status: () => "done",
    visual: () => <WordCards words={VOCABULARY} numbers={VOCABULARY.map((_, i) => i)} />,
    rowsTitle: "Three ways to cut text",
    rows: () => [
      { text: "Characters (Lesson 09)", value: "small dictionary, long text" },
      { text: "Whole words (this model)", value: `${VOCABULARY.length} tokens, unknown words fail` },
      { text: "BPE word pieces (Lesson 18)", value: "GPT-2: 50,257 tokens, any text" },
      { text: `Encode "my dog"`, value: `[${VOCABULARY.indexOf("my")}, ${VOCABULARY.indexOf("dog")}]` },
      { text: `Decode [${VOCABULARY.indexOf("my")}, ${VOCABULARY.indexOf("dog")}]`, value: `"my dog"` },
    ],
    note: () => `"stop" (ID ${VOCABULARY.indexOf(END_TOKEN)}) is a real token. It is how a model learns to finish.`,
    code: "class Tokenizer {\n  encode(text: string): number[] {\n    return text\n      .split(/\\s+/)\n      .map((w) => this.wordToId.get(w));\n  }\n  decode(ids: number[]): string {\n    return ids\n      .map((i) => this.idToWord.get(i))\n      .join(\" \");\n  }\n}",
    why: "A token number has no meaning. It is only an address, like a locker number.",
  },
  {
    id: "pairs",
    stage: "data",
    icon: "question",
    title: "Training pairs",
    lesson: "Lessons 09, 11",
    goal: "Make questions: given these tokens, what comes next?",
    from: () => "Token IDs",
    to: () => `${count(questionTotal)} questions`,
    status: () => "done",
    rowsTitle: "From the first sentence",
    rows: () => FIRST_LINE.slice(0, -1).map((_, i) => ({ text: joinTokens(FIRST_LINE.slice(0, i + 1)), value: `→ ${FIRST_LINE[i + 1]}` })),
    code: "const inputs = tokens.slice(0, -1);\nconst targets = tokens.slice(1);",
    why: "Nobody writes the answers. The next word in the text is always the answer. This is the only task a base model trains on.",
  },
];

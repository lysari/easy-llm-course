import type { ModelPart } from "@/features/take-apart/interfaces/model-part.interface";
import { count } from "@/shared/utils/format";

interface AttentionMathSpec {
  headSize: number;
  context: number;
}

export function attentionMathParts({ headSize, context }: AttentionMathSpec): ModelPart[] {
  const size = count(headSize);
  const words = count(context);
  return [
    {
      id: "scores",
      icon: "percent",
      title: "Match scores",
      copies: 1,
      facts: { shape: `up to ${words} × ${words}`, weightCount: 0, input: "this word's query and every earlier word's key", output: "one share per earlier word, adding up to 1" },
      tiny: { part: "scores", level: "whole" },
      lesson: "Lessons 11, 12",
      reverseLesson: "RL-03",
      explanation: "The query is compared with the key of every earlier word by a dot product. Softmax then turns the scores into shares that add up to 1. It stores no numbers of its own.",
      why: "This is all \"attention\" is: dot products and one softmax. A high score only means two lists of numbers point the same way.",
      formula: { lines: [`score = q · k / √${headSize}`, "shares = softmax(scores)"] },
      code: `const scores = keys.map(\n  (k) => dot(q, k) / Math.sqrt(${headSize}),\n);\nconst shares = softmax(scores);`,
      children: [
        {
          id: "score",
          icon: "hash",
          title: "One score",
          copies: context,
          facts: { shape: "1", weightCount: 0, input: `${size} query numbers and ${size} key numbers`, output: "1 score" },
          tiny: { part: "scores", level: "number" },
          lesson: "Lesson 12",
          reverseLesson: "RL-03",
          explanation: `How well this word's query matches one earlier word's key: ${size} pairs of numbers, multiplied and added up.`,
          why: "One number per pair of words. Up to the full context, every word gets one against every earlier word.",
          formula: { lines: [`score = q1 × k1 + q2 × k2 + … + q${headSize} × k${headSize}`] },
          code: `let sum = 0;\nfor (let i = 0; i < ${headSize}; i++) {\n  sum = sum + q[i] * k[i];\n}`,
          children: [
            {
              id: "score-multiply-add",
              icon: "plus",
              title: "One multiply-and-add",
              copies: headSize,
              facts: { shape: "1 × 1", weightCount: 0, input: "1 query number and 1 key number", output: "the running sum" },
              tiny: { part: "scores", level: "multiply-add" },
              lesson: "Lessons 00, 12",
              reverseLesson: "RL-06",
              explanation: "Ground zero. One query number times one key number, added to the running sum. Neither is a stored number: both were made a moment ago from the words.",
              why: "The smallest piece of attention is the same multiply-and-add as everywhere else.",
              formula: { lines: ["sum = sum + q × k"] },
              code: "sum = sum + q * k;",
              groundZero: "multiply-add",
            },
          ],
        },
      ],
    },
    {
      id: "mix",
      icon: "merge",
      title: "Weighted mix",
      copies: 1,
      facts: { shape: `${words} × ${size}`, weightCount: 0, input: "the shares and every earlier word's value", output: `${size} numbers` },
      tiny: { part: "mix", level: "whole" },
      lesson: "Lesson 12",
      reverseLesson: "RL-03",
      explanation: "Each earlier word's value is multiplied by its share, and they are all added together. Words with a big share count more. It stores no numbers of its own.",
      why: "The result is a weighted average of values. Information moves between words only through this mix.",
      formula: { lines: ["out = Σ share(i) × value(i)"] },
      code: `const out = values.reduce(\n  (sum, v, i) => add(sum, scale(v, shares[i])),\n  zeros(${headSize}),\n);`,
      children: [
        {
          id: "mix-number",
          icon: "hash",
          title: "One mixed number",
          copies: headSize,
          facts: { shape: "1", weightCount: 0, input: `up to ${words} shares and ${words} value numbers`, output: "1 number" },
          tiny: { part: "mix", level: "number" },
          lesson: "Lesson 12",
          reverseLesson: "RL-03",
          explanation: `One of the ${size} numbers that come out of the head: every earlier word's share times that word's value number, added up.`,
          why: "Same shape as every other number here: multiply, add, repeat.",
          formula: { lines: ["out = s1 × v1 + s2 × v2 + …"] },
          code: "let sum = 0;\nfor (let i = 0; i < shares.length; i++) {\n  sum = sum + shares[i] * v[i];\n}",
          children: [
            {
              id: "mix-multiply-add",
              icon: "plus",
              title: "One multiply-and-add",
              copies: context,
              facts: { shape: "1 × 1", weightCount: 0, input: "1 share and 1 value number", output: "the running sum" },
              tiny: { part: "mix", level: "multiply-add" },
              lesson: "Lessons 00, 12",
              reverseLesson: "RL-06",
              explanation: "Ground zero. One share times one value number, added to the running sum.",
              why: "No stored number is used here. The shares and values were both made from the words a moment ago.",
              formula: { lines: ["sum = sum + s × v"] },
              code: "sum = sum + s * v;",
              groundZero: "multiply-add",
            },
          ],
        },
      ],
    },
  ];
}

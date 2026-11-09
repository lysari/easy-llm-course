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
      summary: "Checks how well this word matches each earlier word, as shares that add up to 1.",
      copies: 1,
      facts: { shape: `up to ${words} × ${words}`, weightCount: 0, headline: { value: context, unit: "words at most" }, input: "query and keys", output: "one share per word", inputDetail: "this word's query and every earlier word's key", outputDetail: "one share per earlier word, adding up to 1" },
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
          summary: "How well this word's question matches one earlier word's label, as one number.",
          copies: context,
          facts: { shape: "1", weightCount: 0, headline: { value: headSize, unit: "pairs multiplied" }, input: `${size} query, ${size} key`, output: "1 score", inputDetail: `${size} query numbers and ${size} key numbers` },
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
              summary: "Multiplies one query number by one key number and adds it to the sum.",
              copies: headSize,
              facts: { shape: "1 × 1", weightCount: 0, headline: { value: headSize, unit: "per score" }, input: "1 query, 1 key", output: "the running sum", inputDetail: "1 query number and 1 key number" },
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
      summary: "Blends the earlier words' values, giving more weight to the best matches.",
      copies: 1,
      facts: { shape: `${words} × ${size}`, weightCount: 0, headline: { value: headSize, unit: "numbers per word" }, input: "shares and values", output: `${size} numbers`, inputDetail: "the shares and every earlier word's value" },
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
          summary: "One number of the blend: each earlier word's share times its value, added up.",
          copies: headSize,
          facts: { shape: "1", weightCount: 0, headline: { value: context, unit: "words at most" }, input: "shares and values", output: "1 number", inputDetail: `up to ${words} shares and ${words} value numbers` },
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
              summary: "Multiplies one share by one value number and adds it to the sum.",
              copies: context,
              facts: { shape: "1 × 1", weightCount: 0, headline: { value: context, unit: "per mixed number" }, input: "1 share, 1 value", output: "the running sum", inputDetail: "1 share and 1 value number" },
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

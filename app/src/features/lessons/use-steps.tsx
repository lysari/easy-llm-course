import { END_TOKEN, MAX_CONTEXT, START_CHOICES, joinTokens } from "@/core/model/corpus";
import { TARGET_ROUNDS } from "@/core/model/lab";
import type { FlowStep } from "@/features/lessons/interfaces/story.interface";
import { ChanceBars } from "@/shared/components/chance-bars";
import type { Status } from "@/shared/interfaces/card.interface";
import { count, percent, showWord } from "@/shared/utils/format";
import { WordCards } from "./lesson-plots";
import type { Session } from "./session";

const useStatus = (session: Session, whenTrained: Status): Status => (session.lab.isTrained ? whenTrained : "queued");
const trainFirst = (session: Session) => (session.lab.isTrained ? "" : `Train Tiny GPT first (${TARGET_ROUNDS} rounds). Until then, these numbers come from an unfinished model.`);
const LESSON_LOGITS = [2, 1, 0];
const softmaxWith = (temperature: number) => {
  const raw = LESSON_LOGITS.map((score) => Math.exp(score / temperature));
  const total = raw.reduce((sum, value) => sum + value, 0);
  return raw.map((value) => `${((value / total) * 100).toFixed(1)}%`).join(", ");
};

export const useSteps: FlowStep[] = [
  {
    id: "prompt",
    stage: "use",
    icon: "prompt",
    title: "You send a prompt",
    lesson: "Lesson 14",
    goal: "The same path runs, but with no right answer to compare.",
    from: () => "Your text",
    to: (session) => `Top guess: ${showWord(session.options()[0].token)}`,
    status: (session) => useStatus(session, session.chosenStart ? "done" : "needs"),
    rowsTitle: "Choose a start",
    rows: (session) => START_CHOICES.map((start) => ({ text: `${joinTokens(start.split(" "))} …`, command: `choose:${start}`, isOn: session.chosenStart?.join(" ") === start })),
    visual: (session) => {
      const options = session.options();
      return <ChanceBars options={options} answer={options[0].token} limit={5} />;
    },
    note: (session) => trainFirst(session) || "Training has guess, loss, backprop and update. Using keeps only the guess, so the numbers never change. It does not learn from you.",
    code: "const context = tokens.slice(-blockSize);\nconst { logits } = forward(context);\nconst lastLogits = logits[logits.length - 1];",
    why: "This is called inference. The model cannot look things up. It only calculates what is likely to come next.",
  },
  {
    id: "sampling",
    stage: "use",
    icon: "dice",
    title: "Sampling",
    lesson: "Lesson 23",
    goal: "Pick one token from the percents.",
    from: () => "Percents",
    to: () => "1 token",
    status: (session) => useStatus(session, session.picks.length > 0 ? "done" : "needs"),
    actions: (session) => (session.lab.isTrained ? [{ label: "Pick 20 times", command: "pick-many", isStrong: true }, { label: "Clear", command: "pick-clear" }] : []),
    rowsTitle: "Settings. Press to change.",
    rows: ({ temperature, topK, topP }) => [
      { text: "Temperature 0.5: safer", value: "sharper", command: "temperature:0.5", isOn: temperature === 0.5 },
      { text: "Temperature 1: normal", value: "no change", command: "temperature:1", isOn: temperature === 1 },
      { text: "Temperature 2: more random", value: "flatter", command: "temperature:2", isOn: temperature === 2 },
      { text: "Top-K off", value: "all tokens", command: "topk:0", isOn: topK === 0 },
      { text: "Top-K 2", value: "best 2 tokens only", command: "topk:2", isOn: topK === 2 },
      { text: "Top-P off", value: "all tokens", command: "topp:1", isOn: topP === 1 },
      { text: "Top-P 0.9", value: "until they reach 90%", command: "topp:0.9", isOn: topP === 0.9 },
    ],
    visual: (session) => {
      const options = session.sampledOptions();
      const picks = session.pickCounts();
      return (
        <>
          <ChanceBars options={options} answer={options[0].token} limit={5} />
          {picks.length > 0 && (
            <>
              <p className="section-title">{`What got picked (${session.picks.length})`}</p>
              <WordCards words={picks.map(([word]) => word)} numbers={picks.map(([, number]) => `${number}×`)} />
            </>
          )}
        </>
      );
    },
    note: () => `The lesson's logits [2, 1, 0]: T = 0.5 gives ${softmaxWith(0.5)}. T = 1 gives ${softmaxWith(1)}. T = 2 gives ${softmaxWith(2)}.`,
    formula: {
      lines: ["p = softmax(logits / T)"],
      legend: [
        { term: "Top-K", meaning: "keep the K highest, then sample" },
        { term: "Top-P", meaning: "keep tokens until they total p" },
      ],
    },
    code: "let r = Math.random();\nfor (let i = 0; i < probs.length; i++) {\n  r -= probs[i];\n  if (r <= 0) return i;\n}",
    why: "Always taking the top word makes writing repeat itself. Sampling is why the same question can get different answers.",
  },
  {
    id: "generate",
    stage: "use",
    icon: "generate",
    title: "Generate the reply",
    lesson: "Lesson 14",
    goal: "Add the token, then run the whole path again.",
    from: () => "1 token",
    to: (session) => `${session.written.filter((word) => word !== END_TOKEN).length} new words`,
    status: (session) => {
      if (session.isRunning("write")) return "running";
      return useStatus(session, session.isWritingDone ? "done" : "needs");
    },
    actions: (session) => (session.lab.isTrained && !session.isRunning("write") ? [{ label: session.written.length > 0 ? "Write again" : "Write a reply", command: "start:write", isStrong: true }] : []),
    visual: (session) => <WordCards words={[...session.startWords, ...session.written]} newFrom={session.startWords.length} />,
    rowsTitle: "Each new token",
    rows: (session) => session.writtenSteps().map((step, i) => ({ text: `Trip ${i + 1}: top ${showWord(step.top)} ${percent(step.topChance)}`, value: `picked ${showWord(step.word)}` })),
    note: (session) => trainFirst(session) || `It stops when it picks "stop", or at the context window of ${MAX_CONTEXT} tokens.`,
    code: "function generate(\n  seed: string,\n  length: number,\n  temperature = 0.8,\n) {\n  let tokens = tokenizer.encode(seed);\n  for (let i = 0; i < length; i++) {\n    const context = tokens.slice(-blockSize);\n    const { logits } = forward(context);\n    tokens.push(\n      sampleToken(\n        logits[logits.length - 1],\n        temperature,\n      ),\n    );\n  }\n  return tokenizer.decode(tokens);\n}",
    why: "Each token is a full trip through the model. That is why chat apps show words one by one.",
  },
  {
    id: "kv-cache",
    stage: "use",
    icon: "cache",
    title: "KV cache",
    lesson: "Lesson 21",
    goal: "Save earlier K and V, so only the new token needs work.",
    from: () => "Repeated work",
    to: () => "Far less work",
    status: () => "lesson",
    rowsTitle: "Word-steps of attention work (smaller = faster)",
    rows: (session) => {
      const length = session.startWords.length + session.written.length;
      const work = (n: number) => (n * (n + 1)) / 2;
      return [
        { text: `Your reply (${length} tokens), no cache`, value: count(work(length)) },
        { text: "Your reply, with cache", value: count(length) },
        { text: "1,000 tokens, no cache", value: count(work(1000)) },
        { text: "1,000 tokens, with cache", value: "1,000" },
        { text: "10,000 tokens, no cache", value: count(work(10000)) },
      ];
    },
    note: () => "Why not Q too? Q is only needed for the newest token. K and V of old tokens never change. This small app runs without a cache.",
    formula: {
      lines: ["No cache: 1 + 2 + … + N = N(N + 1) / 2", "With cache: N"],
      legend: [
        { term: "Prefill", meaning: "run the whole prompt once" },
        { term: "Decode", meaning: "one token at a time" },
      ],
    },
    why: "Without it, the waste grows with the square of the length. The cost is memory, which real systems manage carefully.",
  },
];

export const beyondSteps: FlowStep[] = [
  {
    id: "scale",
    stage: "beyond",
    icon: "scale",
    title: "Scaling laws",
    lesson: "Lesson 22",
    goal: "Bigger models improve in a predictable way.",
    from: () => "A small model",
    to: () => "A very large one",
    status: () => "lesson",
    rowsTitle: "Numbers in each model (Lesson 30)",
    rows: () => [
      { text: "GPT-1 (2018)", value: "117 million" },
      { text: "GPT-2 (2019)", value: "1.5 billion" },
      { text: "GPT-3 (2020)", value: "175 billion" },
      { text: "GPT-4 (2023)", value: "unknown, rumored ~1 trillion" },
      { text: "Doubling the numbers gives", value: `about ${Math.round((1 - 2 ** -0.076) * 100)}% lower loss` },
    ],
    formula: { lines: ["L(N) ≈ (Nc / N)^0.076"] },
    why: "You can train a small model, measure it, and predict a much bigger one before paying for it. Chinchilla (2022) added how much data to use.",
  },
  {
    id: "in-context",
    stage: "beyond",
    icon: "in-context",
    title: "In-context learning",
    lesson: "Lesson 24",
    goal: "A large model can follow examples inside your prompt.",
    from: () => "Examples in the prompt",
    to: () => "A new task, no training",
    status: () => "lesson",
    rowsTitle: "Three modes",
    rows: () => [
      { text: "Zero-shot", value: "only an instruction" },
      { text: "One-shot", value: "1 example" },
      { text: "Few-shot", value: "a few examples" },
    ],
    code: "const prompt = `Translate English to French:\ncheese →`;",
    note: () => "No numbers change. It just continues the pattern. Tiny GPT is far too small to do this.",
    why: "GPT-3's big discovery. It needs a model that is large enough. Prompt engineering comes from this.",
  },
  {
    id: "finetune",
    stage: "beyond",
    icon: "fine-tune",
    title: "Instruction fine-tuning",
    lesson: "Lesson 26",
    goal: "Teach a text completer to follow instructions.",
    from: () => "Base model",
    to: () => "A model that replies",
    status: () => "lesson",
    rows: () => [
      { text: "Base model asked for a poem", value: "writes more requests" },
      { text: "Training data", value: "instruction + ideal reply" },
      { text: "Same loop", value: "smaller learning rate" },
      { text: "Risk", value: "forgets old skills" },
      { text: "LoRA", value: "trains only small add-ons" },
    ],
    note: (session) => `Tiny GPT is a base model too. Given "my dog", it writes "${session.sampleWriting()}". It continues the text and does not answer.`,
    why: "Base models learned to predict the next token, not to be helpful. Fine-tuning is much cheaper than pre-training.",
  },
  {
    id: "reward",
    stage: "beyond",
    icon: "reward",
    title: "Reward model",
    lesson: "Lesson 27",
    goal: "Learn which replies people prefer.",
    from: () => "Pairs of replies",
    to: () => "A score for any reply",
    status: () => "lesson",
    rows: () => [
      { text: "People are asked", value: "which of 2 is better?" },
      { text: "Reward model outputs", value: "1 score per reply" },
      { text: "Goal", value: "r_A above r_B" },
      { text: "Experts disagree", value: "about 20–30% of the time" },
      { text: "Danger", value: "reward hacking" },
    ],
    formula: { lines: ["P(A preferred) = sigmoid(r_A − r_B)"] },
    why: "People find it easier to recognise a good reply than to describe one. This turns that judgment into a number.",
  },
  {
    id: "rlhf",
    stage: "beyond",
    icon: "rlhf",
    title: "RLHF with PPO",
    lesson: "Lesson 28",
    goal: "Nudge the model toward replies that score well.",
    from: () => "Fine-tuned model",
    to: () => "A helpful assistant",
    status: () => "lesson",
    rows: () => [
      { text: "Agent", value: "the LLM" },
      { text: "Action", value: "generate the next token" },
      { text: "Reward", value: "the reward model's score" },
      { text: "Clip each update", value: "avoids big changes" },
      { text: "KL penalty", value: "stays near the old model" },
    ],
    formula: { lines: ["maximize E[reward_model(prompt, completion)]", "r(t) = π_new(token) / π_old(token)", "A = R − V(state)"] },
    why: "This is how InstructGPT and ChatGPT became helpful. DPO is a simpler alternative that the lesson also covers.",
  },
  {
    id: "moe",
    stage: "beyond",
    icon: "experts",
    title: "Mixture of Experts",
    lesson: "Lesson 29",
    goal: "Have many numbers, but use only some per token.",
    from: () => "One big FFN",
    to: () => "N experts, K active",
    status: () => "lesson",
    rows: () => [
      { text: "Replace the FFN with", value: "N smaller experts" },
      { text: "A router picks", value: "K experts per token" },
      { text: "Output", value: "weighted sum of the K" },
      { text: "Problem to manage", value: "load balancing" },
    ],
    formula: { lines: ["Input → Router → K of N experts → weighted sum → Output"] },
    why: "In a normal model, every token passes through every number. This breaks that link, so capacity grows without the same cost.",
  },
  {
    id: "capstone",
    stage: "beyond",
    icon: "pipeline",
    title: "The complete picture",
    lesson: "Lesson 30",
    goal: "Everything you built, from y = wx + b to GPT-4.",
    from: () => "y = w × x + b",
    to: () => "GPT-4",
    status: () => "lesson",
    rows: () => [
      { text: "Lessons 00–04", value: "weights, gradients, learning loop" },
      { text: "Lessons 05–07", value: "networks, backprop" },
      { text: "Lessons 08–11", value: "tokens → vectors → percents" },
      { text: "Lessons 12–17", value: "GPT-1" },
      { text: "Lessons 18–21", value: "GPT-2" },
      { text: "Lessons 22–25", value: "GPT-3" },
      { text: "Lessons 26–28", value: "the helpful assistant" },
      { text: "Lesson 29", value: "GPT-4" },
    ],
    note: () => "Lesson 30 also points to what to read and build next.",
    formula: { lines: ["tokens → model → probability of the next token"] },
    why: "Every language model is a function that takes tokens and gives a probability for the next one. The learning is in the numbers.",
  },
];

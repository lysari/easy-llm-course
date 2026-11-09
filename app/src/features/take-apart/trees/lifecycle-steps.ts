import type { ModelPart, StepFacts } from "@/features/take-apart/interfaces/model-part.interface";
import { evaluationStep, postTrainingStep, preTrainingStep } from "./training-steps";

interface LifecycleSpec {
  model: ModelPart;
  data: StepFacts;
  tokenizer: StepFacts;
  preTraining: StepFacts;
  postTraining: StepFacts;
  benchmarks: StepFacts;
  safety: StepFacts;
  quantization: StepFacts;
}

const HAND_WRITTEN = "Our tiny corpus is 12 sentences written by hand, so nothing is filtered out.";
const REPEATS_ON_PURPOSE = "Our tiny corpus repeats some sentences on purpose, up to 3 times, so their endings come out more often. It skips this check.";
const NO_LEARNED_MERGES = "Our tiny model learns no merge rules. Its list is every word in its sentences, split on spaces.";

function dataStep(facts: StepFacts): ModelPart {
  return {
    id: "data",
    icon: "book",
    title: "Data collection and prep",
    summary: "Gathers a huge pile of text and cleans it up for training.",
    copies: 1,
    facts: { shape: facts.shape, weightCount: 0, headline: facts.headline, input: "web pages", output: "clean training text", notes: facts.notes },
    tiny: { part: "corpus", level: "whole" },
    lesson: "Lessons 22, 30",
    reverseLesson: "RL-04",
    explanation: "Gather a huge amount of text, drop the low-quality and harmful parts, and remove copies. What is left is all the model will ever learn from.",
    why: "The model can only echo patterns in this text. Its \"knowledge\" is whatever this pile happened to contain.",
    formula: { lines: ["gather → filter → remove copies"] },
    children: [
      {
        id: "gather",
        icon: "down",
        title: "Copy one document in",
        summary: "Copies the plain text of one web page into the pile.",
        copies: 1,
        facts: { shape: "1 document", weightCount: 0, headline: { value: 1, unit: "document at a time" }, input: "one web page", output: "its plain text" },
        tiny: { part: "corpus", level: "row" },
        lesson: "Lesson 30",
        reverseLesson: "RL-04",
        explanation: "Ground zero. Take one web page, strip out the menus and code, and keep its plain text as one document.",
        why: "Nobody reads it first. It goes in as a long run of characters.",
        formula: { lines: ["document = text(page)"] },
        groundZero: "action",
      },
      {
        id: "filter",
        icon: "sliders",
        title: "One quality check",
        summary: "Checks one document against one rule, then keeps it or drops it.",
        copies: 1,
        facts: { shape: "1 rule", weightCount: 0, headline: { value: 1, unit: "rule" }, input: "one document", output: "keep or drop" },
        tiny: { part: "corpus", level: "row", missing: HAND_WRITTEN },
        lesson: "No lesson yet",
        reverseLesson: "RL-04",
        explanation: "Ground zero. Check one document against one rule, such as \"too many repeated lines\", and keep it or drop it.",
        why: "Rules and small scoring models decide what counts as good text, long before training starts.",
        formula: { lines: ["keep = passes(document, rule)"] },
        groundZero: "action",
      },
      {
        id: "dedupe",
        icon: "copy",
        title: "One deduplication check",
        summary: "Drops a document if it copies, or nearly copies, one already kept.",
        copies: 1,
        facts: { shape: "1 comparison", weightCount: 0, headline: { value: 1, unit: "comparison" }, input: "one document", output: "keep or drop", inputDetail: "one document and those kept", outputDetail: "keep, or drop as a copy" },
        tiny: { part: "corpus", level: "row", missing: REPEATS_ON_PURPOSE },
        lesson: "No lesson yet",
        reverseLesson: "RL-04",
        explanation: "Ground zero. Compare one document with the ones already kept, and drop it if it is a copy or nearly one.",
        why: "Text seen many times gets copied back word for word, so copies are removed.",
        formula: { lines: ["drop if seen.has(fingerprint(document))"] },
        groundZero: "action",
      },
    ],
  };
}

function tokenizerStep(facts: StepFacts): ModelPart {
  return {
    id: "tokenizer",
    icon: "text",
    title: "Tokenizer",
    summary: "Builds the word list the model will use, by counting common pairs of letters.",
    copies: 1,
    facts: { shape: facts.shape, weightCount: 0, headline: facts.headline, input: "clean text", output: "a word list", inputDetail: "clean training text", outputDetail: "a word list: piece ↔ ID", notes: facts.notes },
    tiny: { part: "vocabulary", level: "whole" },
    lesson: "Lessons 09, 18",
    reverseLesson: "RL-01",
    explanation: "Build the list of words the model will use. BPE starts from single bytes, then keeps joining the most common pair of neighbouring pieces into a new piece, until the list is full.",
    why: "The list is made by counting, not by meaning. Common words become one piece; rare words stay in parts.",
    formula: { lines: ["repeat: count pairs, merge the top pair", "until the list is full"] },
    children: [
      {
        id: "merge-round",
        icon: "repeat",
        title: "One merge round",
        summary: "Joins the most common pair of pieces into one new piece.",
        copies: 1,
        facts: { shape: "1 new piece per round", weightCount: 0, headline: { value: 1, unit: "new piece" }, input: "text in pieces", output: "1 new piece, 1 rule", inputDetail: "the text, split into pieces", outputDetail: "one new piece and one rule" },
        tiny: { part: "vocabulary", level: "whole", missing: NO_LEARNED_MERGES },
        lesson: "Lesson 18",
        reverseLesson: "RL-01",
        explanation: "Count every pair of neighbouring pieces in the text, join the most common pair into one new piece, and write that down as a merge rule.",
        why: "The rules are kept in order. Using the model later replays them on new text.",
        formula: { lines: ["top = most common pair", "rules.push(top)"] },
        children: [
          {
            id: "count-pair",
            icon: "hash",
            title: "Count one pair",
            summary: "Adds 1 to the count for one pair of side-by-side pieces.",
            copies: 1,
            facts: { shape: "1 count", weightCount: 0, headline: { value: 1, unit: "count" }, input: "two pieces", output: "their count, plus 1", inputDetail: "two neighbouring pieces" },
            tiny: { part: "vocabulary", level: "whole", missing: NO_LEARNED_MERGES },
            lesson: "Lesson 18",
            reverseLesson: "RL-01",
            explanation: "Ground zero. Add 1 to the count for one pair of neighbouring pieces, such as \"d\" then \"o\".",
            why: "The tokenizer is built from counts like this, nothing more.",
            formula: { lines: ["count[a + b] = count[a + b] + 1"] },
            groundZero: "action",
          },
          {
            id: "learn-merge",
            icon: "merge",
            title: "One merge rule",
            summary: "Writes down one join rule and adds the joined piece to the list.",
            copies: 1,
            facts: { shape: "1 rule", weightCount: 0, headline: { value: 1, unit: "rule" }, input: "the most common pair", output: "one new piece" },
            tiny: { part: "vocabulary", level: "whole", missing: NO_LEARNED_MERGES },
            lesson: "Lesson 18",
            reverseLesson: "RL-01",
            explanation: "Ground zero. Write down one rule, such as \"do\" + \"g\" → \"dog\", and add the joined piece to the list.",
            why: "\"dog\" becomes one piece because it was common, not because anything knows it is an animal.",
            formula: { lines: ["rule: \"do\" + \"g\" → \"dog\""] },
            groundZero: "action",
          },
        ],
      },
      {
        id: "assign-id",
        icon: "hash",
        title: "Give one piece an ID",
        summary: "Writes one piece into the list, and its place number becomes its ID.",
        copies: 1,
        facts: { shape: "1 row of the list", weightCount: 0, headline: { value: facts.headline.value, unit: "IDs given" }, input: "one piece", output: "its number" },
        tiny: { part: "vocabulary", level: "number" },
        lesson: "Lesson 09",
        reverseLesson: "RL-01",
        explanation: "Ground zero. Write one piece and its place number into the list. That number is its ID from now on.",
        why: "The ID is only a place in a list. Nearby IDs mean nothing.",
        formula: { lines: ["list[id] = piece"] },
        groundZero: "action",
      },
    ],
  };
}

function architectureStep(model: ModelPart): ModelPart {
  return {
    id: "architecture",
    icon: "network",
    title: "Architecture",
    summary: "Writes the model as code, with every stored number starting out random.",
    copies: 1,
    facts: { shape: model.facts.shape, weightCount: model.facts.weightCount, headline: model.facts.headline, input: "sizes and parts", output: "a random model", outputDetail: "code with random numbers" },
    tiny: { part: "model", level: "whole" },
    lesson: "Lessons 13, 14, 20",
    reverseLesson: "RL-05",
    explanation: "Write the model as code: the word table, the layers of attention and feed-forward with their norms and add-backs, and the output head. Every stored number starts random.",
    why: "Before training, this exact code with random numbers writes nonsense. The shape alone knows nothing.",
    formula: { lines: ["numbers = random()", "model = layers(numbers)"] },
    children: [model],
  };
}

export function lifecycleRoot(spec: LifecycleSpec): ModelPart {
  return {
    id: "build",
    icon: "pen",
    title: "Build a model",
    summary: "How a model is made, from raw text to a model you can use, in 6 stages.",
    copies: 1,
    facts: { shape: "6 stages", weightCount: spec.model.facts.weightCount, headline: spec.model.facts.headline, input: "web text", output: "a model ready to use", inputDetail: "text from the web" },
    tiny: { part: "model", level: "whole" },
    lesson: "Lessons 14, 30",
    reverseLesson: "RL-05",
    explanation: "How a model is made: collect text, build the word list, write the model code, train it to guess the next word, teach it to answer, then test and shrink it.",
    why: "Only the architecture stage holds stored numbers. Every other stage shapes what those numbers end up being.",
    formula: { lines: ["data → word list → code → pre-train → post-train → test"] },
    children: [
      dataStep(spec.data),
      tokenizerStep(spec.tokenizer),
      architectureStep(spec.model),
      preTrainingStep(spec.preTraining),
      postTrainingStep(spec.postTraining),
      evaluationStep(spec.benchmarks, spec.safety, spec.quantization),
    ],
  };
}

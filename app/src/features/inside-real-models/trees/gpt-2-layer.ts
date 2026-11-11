import type { ModelPart } from "@/features/inside-real-models/interfaces/model-part.interface";
import { gpt2Attention } from "./gpt-2-attention";
import { gpt2FeedForward } from "./gpt-2-feed-forward";
import { GPT_2_ADD_BACK, GPT_2_DEPTH, GPT_2_LAYER_LOOP, GPT_2_NORM_CODE, GPT_2_NORM_PLACE, GPT_2_SIZE_HELPS } from "./gpt-2-reasons";
import { normPart } from "./part-builders";
import { GPT_2_SETTINGS } from "./sources";

const WIDTH = 768;
const LAYER_COUNT = 12;

const layer: ModelPart = {
  id: "layer",
  icon: "layer",
  title: "One layer",
  summary: "Looks back at earlier words, then picks out patterns, adding each result back on.",
  copies: LAYER_COUNT,
  facts: { shape: "norm, attention, norm, feed-forward", weightCount: 7_087_872, headline: { value: 7_087_872, unit: "numbers", isComputed: true }, input: "768 per word", output: "768, updated", inputDetail: "768 numbers per word", outputDetail: "768 numbers per word, with attention's and the feed-forward's results added on", sources: [...GPT_2_SETTINGS, "gpt-2-paper"], reasons: [GPT_2_ADD_BACK, GPT_2_NORM_PLACE] },
  tiny: { part: "layer", level: "whole" },
  explanation:
    "One of 12 identical layers. A norm, then attention, added back on; a second norm, then the feed-forward, added back on. GPT-2 puts each norm before its part, not after.",
  why: "Adding each result back (x + part(x)) lets the numbers flow straight through 12 layers, so training can reach the first one.",
  story: {
    problem: "A layer is the unit that repeats: look back at other words, then work on each word, each time adding the result back on. Without the adding back, each layer would overwrite the one before.",
    demo: "tiny-part",
    when: "Written once in the code and stacked 12 times; every word goes through it each time text is read.",
    feeds: "Its output goes into the next layer, or, after the 12th, into the final norm.",
    nextId: "norm-1",
  },
  formula: { lines: ["x = x + attention(norm1(x))", "x = x + ffn(norm2(x))"] },
  code: "x = add(x, attention(norm1(x)));\nx = add(x, feedForward(norm2(x)));",
  children: [
    normPart({
      id: "norm-1",
      icon: "norm",
      title: "Norm before attention",
      summary: "Keeps a word's numbers in a steady range before attention reads them.",
      copies: 1,
      size: WIDTH,
      hasShift: true,
      tinyPart: "norm",
      sources: [...GPT_2_SETTINGS, "gpt-2-paper"],
      explanation: "LayerNorm. Shifts a word's 768 numbers to an average of 0 and a spread of 1, then scales and shifts each one by its own 2 stored numbers.",
      why: "Keeps the numbers in a steady range however many layers came before, so training stays stable.",
      story: {
        problem: "As layers add onto a word, its numbers can drift very big or very small. Without a norm, attention would read numbers of wildly different sizes from layer to layer.",
        demo: "tiny-part",
        when: "At the start of every layer, before attention, for every word.",
        feeds: "The steadied numbers go into the query, key and value tables.",
        nextId: "attention",
      },
      reasons: [GPT_2_NORM_PLACE, GPT_2_NORM_CODE],
      codeReasons: [GPT_2_NORM_CODE],
    }),
    gpt2Attention,
    normPart({
      id: "norm-2",
      icon: "norm",
      title: "Norm before feed-forward",
      summary: "Keeps a word's numbers in a steady range before the feed-forward reads them.",
      copies: 1,
      size: WIDTH,
      hasShift: true,
      tinyPart: "norm",
      sources: [...GPT_2_SETTINGS, "gpt-2-paper"],
      explanation: "The same kind of LayerNorm with its own 768 scale and 768 shift numbers, run before the feed-forward.",
      why: "Each part gets inputs in a steady range, whatever attention just added.",
      story: {
        problem: "Attention has just added its result onto the word, which changes the numbers' size again. This norm steadies them before the feed-forward reads them.",
        demo: "tiny-part",
        when: "In every layer, between attention and the feed-forward, for every word.",
        feeds: "The steadied numbers go into the expand table.",
        nextId: "feed-forward",
      },
      reasons: [GPT_2_NORM_PLACE, GPT_2_NORM_CODE],
      codeReasons: [GPT_2_NORM_CODE],
    }),
    gpt2FeedForward,
  ],
};

export const gpt2Layers: ModelPart = {
  id: "layers",
  icon: "layers",
  title: "Stack of 12 layers",
  summary: "Runs every word through 12 layers of the same shape, one after another.",
  copies: 1,
  facts: { shape: "12 layers", weightCount: 85_054_464, headline: { value: 85_054_464, unit: "numbers", isComputed: true }, input: "word vectors", output: "updated word vectors", sources: GPT_2_SETTINGS, reasons: [GPT_2_LAYER_LOOP, GPT_2_DEPTH, GPT_2_SIZE_HELPS] },
  tiny: { part: "layers", level: "whole" },
  explanation: "12 layers of the same shape, one after another. Each one reads every word's 768 numbers and adds a little more onto them.",
  why: `The same small recipe, stacked. Each of the ${LAYER_COUNT} layers has its own numbers, so each can learn something different.`,
  story: {
    problem: "A single layer can only look back and work on each word once. Stacking 12 lets each layer build on what the ones before it found.",
    demo: "tiny-part",
    when: "You pick the number of layers (12) when you write the architecture; a word goes through all of them every time it is read.",
    feeds: "The 12th layer's output goes to the final norm, then to the output table.",
    nextId: "layer",
  },
  formula: { lines: ["x12 = layer12(… layer2(layer1(x0)) …)"] },
  code: "for (const layer of layers) {\n  x = layer(x);\n}",
  children: [layer],
};

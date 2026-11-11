import type { ModelPart } from "@/features/inside-real-models/interfaces/model-part.interface";
import { llama3Attention } from "./llama-3-attention";
import { llama3FeedForward } from "./llama-3-feed-forward";
import { LLAMA_3_ADD_BACK, LLAMA_3_LAYER_LOOP, LLAMA_3_NO_REASON, LLAMA_3_RMS_NORM } from "./llama-3-reasons";
import { normPart } from "./part-builders";
import { LLAMA_3_SETTINGS } from "./sources";

const WIDTH = 4096;
const LAYER_COUNT = 32;

const layer: ModelPart = {
  id: "layer",
  icon: "layer",
  title: "One layer",
  summary: "Looks back at earlier words, then picks out patterns, adding each result back on.",
  copies: LAYER_COUNT,
  facts: { shape: "norm, attention, norm, feed-forward", weightCount: 218_112_000, headline: { value: 218_112_000, unit: "numbers", isComputed: true }, input: "4,096 per word", output: "4,096 per word", inputDetail: "4,096 numbers per word", outputDetail: "4,096 numbers per word", sources: LLAMA_3_SETTINGS, reasons: [LLAMA_3_ADD_BACK, LLAMA_3_NO_REASON] },
  tiny: { part: "layer", level: "whole" },
  explanation: "One of 32 identical layers. An RMSNorm, then attention, added back on; a second RMSNorm, then the feed-forward, added back on.",
  why: "Adding each result back (x + part(x)) lets the numbers flow straight through 32 layers, so training can reach the first one.",
  story: {
    problem: "A layer is the unit that repeats: look back at other words, then work on each word, each time adding the result back on. Without the adding back, each layer would overwrite the one before.",
    demo: "tiny-part",
    when: "Written once in the code and stacked 32 times; every word goes through it each time text is read.",
    feeds: "Its output goes into the next layer, or, after the 32nd, into the final norm.",
    nextId: "norm-1",
  },
  formula: { lines: ["x = x + attention(norm1(x))", "x = x + ffn(norm2(x))"] },
  code: "x = add(x, attention(norm1(x)));\nx = add(x, feedForward(norm2(x)));",
  children: [
    normPart({
      id: "norm-1",
      icon: "norm",
      title: "RMSNorm before attention",
      summary: "Keeps a word's numbers at a steady size before attention reads them.",
      copies: 1,
      size: WIDTH,
      hasShift: false,
      tinyPart: "norm",
      sources: LLAMA_3_SETTINGS,
      explanation: "RMSNorm. Divides a word's 4,096 numbers by their root mean square, so their size is about 1, then scales each one by its own stored number. It does not subtract the average and has no shift.",
      why: "Llama 3's code writes it as x × 1/√(mean(x²) + 0.00001), then times each scale number.",
      story: {
        problem: "As layers add onto a word, its numbers can drift very big or very small. Without a norm, attention would read numbers of wildly different sizes from layer to layer.",
        demo: "tiny-part",
        when: "At the start of every layer, before attention, for every word.",
        feeds: "The steadied numbers go into the query, key and value tables.",
        nextId: "attention",
      },
      reasons: [LLAMA_3_RMS_NORM, LLAMA_3_NO_REASON],
      codeReasons: [LLAMA_3_RMS_NORM],
    }),
    llama3Attention,
    normPart({
      id: "norm-2",
      icon: "norm",
      title: "RMSNorm before feed-forward",
      summary: "Keeps a word's numbers at a steady size before the feed-forward reads them.",
      copies: 1,
      size: WIDTH,
      hasShift: false,
      tinyPart: "norm",
      sources: LLAMA_3_SETTINGS,
      explanation: "The same kind of RMSNorm with its own 4,096 scale numbers, run before the feed-forward.",
      why: "Each part gets inputs of a steady size, whatever attention just added.",
      story: {
        problem: "Attention has just added its result onto the word, which changes the numbers' size again. This norm steadies them before the feed-forward reads them.",
        demo: "tiny-part",
        when: "In every layer, between attention and the feed-forward, for every word.",
        feeds: "The steadied numbers go into the gate and up tables.",
        nextId: "feed-forward",
      },
      reasons: [LLAMA_3_RMS_NORM, LLAMA_3_NO_REASON],
      codeReasons: [LLAMA_3_RMS_NORM],
    }),
    llama3FeedForward,
  ],
};

export const llama3Layers: ModelPart = {
  id: "layers",
  icon: "layers",
  title: "Layers × 32",
  summary: "Runs every word through 32 layers of the same shape, one after another.",
  copies: 1,
  facts: { shape: "32 layers", weightCount: 6_979_584_000, headline: { value: 6_979_584_000, unit: "numbers", isComputed: true }, input: "word vectors", output: "updated word vectors", sources: LLAMA_3_SETTINGS, reasons: [LLAMA_3_LAYER_LOOP, LLAMA_3_NO_REASON] },
  tiny: { part: "layers", level: "whole" },
  explanation: "32 layers of the same shape, one after another. Each one reads every word's 4,096 numbers and adds a little more onto them.",
  why: `The same small recipe, stacked. Each of the ${LAYER_COUNT} layers has its own numbers, so each can learn something different.`,
  story: {
    problem: "A single layer can only look back and work on each word once. Stacking 32 lets each layer build on what the ones before it found.",
    demo: "tiny-part",
    when: "You pick the number of layers (32) when you write the architecture; a word goes through all of them every time it is read.",
    feeds: "The 32nd layer's output goes to the final norm, then to the output table.",
    nextId: "layer",
  },
  formula: { lines: ["x32 = layer32(… layer2(layer1(x0)) …)"] },
  code: "for (const layer of layers) {\n  x = layer(x);\n}",
  children: [layer],
};

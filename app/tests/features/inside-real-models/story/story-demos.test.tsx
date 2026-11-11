import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { Window } from "happy-dom";
import { Suspense } from "react";
import { prerender } from "react-dom/static";
import { ModelLab } from "@/core/model/lab";
import type { DemoId, ModelId, ModelPart, TinyView } from "@/features/inside-real-models/interfaces/model-part.interface";
import { MODELS } from "@/features/inside-real-models/models";
import { START_TEXTS, readSentence } from "@/features/inside-real-models/sentence";
import { DEMOS, isUsedOn } from "@/features/inside-real-models/story/story-demos";
import { viewTiny } from "@/features/inside-real-models/tiny-parts";

const OWN_TEXT_BOX = 'class="watch-input"';

const window = new Window({ settings: { disableJavaScriptEvaluation: true, disableJavaScriptFileLoading: true, disableCSSFileLoading: true, disableIframePageLoading: true } });

beforeAll(() => {
  globalThis.DOMParser = window.DOMParser as unknown as typeof DOMParser;
});

afterAll(() => window.happyDOM.close());

const allParts = (part: ModelPart): ModelPart[] => [part, ...(part.children ?? []).flatMap(allParts)];

async function render(demo: DemoId, part: ModelPart, modelId: ModelId, tiny: TinyView) {
  const Demo = DEMOS[demo].component;
  const { prelude } = await prerender(
    <Suspense>
      <Demo part={part} modelId={modelId} tiny={tiny} />
    </Suspense>,
  );
  return new Response(prelude).text();
}

function trainedLab() {
  const lab = new ModelLab();
  lab.trainRounds(3);
  return lab;
}

describe.each(MODELS)("$title demos", (model) => {
  const parts = [...new Map([model.modes.use, model.modes.build].flatMap(allParts).map((part) => [part.id, part])).values()];
  const fresh = new ModelLab();
  const trained = trainedLab();
  const [firstWords, otherWords] = [START_TEXTS[0], START_TEXTS[1]].map((text) => readSentence(text).words);

  test.each(Object.keys(DEMOS) as DemoId[])("%s says truly whether it reads the sentence and the training", async (demo) => {
    const shown = parts.filter((part) => part.story?.demo === demo);
    let readsSentence = false;
    let readsTraining = false;
    let hasOwnTextBox = false;
    for (const part of shown) {
      const first = await render(demo, part, model.id, viewTiny(fresh, firstWords));
      expect(await render(demo, part, model.id, viewTiny(fresh, firstWords)), part.id).toBe(first);
      readsSentence ||= first !== (await render(demo, part, model.id, viewTiny(fresh, otherWords)));
      readsTraining ||= first !== (await render(demo, part, model.id, viewTiny(trained, firstWords)));
      hasOwnTextBox ||= first.includes(OWN_TEXT_BOX);
    }
    expect(shown.length).toBeGreaterThan(0);
    expect(isUsedOn(DEMOS[demo].usesTraining, model.id)).toBe(readsTraining);
    if (isUsedOn(DEMOS[demo].usesSentence, model.id)) expect(readsSentence).toBe(true);
    else expect(!readsSentence || hasOwnTextBox).toBe(true);
  });
});

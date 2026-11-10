import { useEffect, useState } from "react";
import { type Gpt2Vocab, readVocab } from "@/features/inside-real-models/story/tokenizer/gpt-2-bpe";

const GPT_2_VOCAB_URL = `${import.meta.env.BASE_URL}gpt-2/vocab.bpe`;

let loading: Promise<Gpt2Vocab> | null = null;

function loadVocab() {
  loading ??= fetch(GPT_2_VOCAB_URL)
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.text();
    })
    .then(readVocab)
    .catch((error: unknown) => {
      loading = null;
      throw error;
    });
  return loading;
}

export function useGpt2Vocab() {
  const [vocab, setVocab] = useState<Gpt2Vocab | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    let isCurrent = true;
    loadVocab().then(
      (loaded) => isCurrent && setVocab(loaded),
      () => isCurrent && setProblem("GPT-2's merge rules could not be loaded from this app. Reload the page to try again."),
    );
    return () => {
      isCurrent = false;
    };
  }, []);

  return { vocab, problem };
}

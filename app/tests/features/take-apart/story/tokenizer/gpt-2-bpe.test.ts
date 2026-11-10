import { describe, expect, test } from "bun:test";
import { learnMerges, countFirstPairs, countPairs, pairPlaces, pieceCount, trainingWords } from "@/features/take-apart/story/tokenizer/bpe-train";
import { BASE_PIECES, END_OF_TEXT, encodeText, mergeSteps, readVocab, splitWords, toBytePieces } from "@/features/take-apart/story/tokenizer/gpt-2-bpe";

const vocab = readVocab(await Bun.file(`${import.meta.dir}/../../../../../public/gpt-2/vocab.bpe`).text());
const idsOf = (text: string) => encodeText(text, vocab).flatMap((word) => word.tokens.map((token) => token.id));

describe("GPT-2's real tokenizer", () => {
  test("reads all 50,000 merge rules and gives 50,257 IDs, first rule Ġ + t", () => {
    expect(vocab.merges.length).toBe(50_000);
    expect(vocab.pieces.length).toBe(50_257);
    expect(vocab.merges[0]).toEqual(["Ġ", "t"]);
    expect(vocab.ids.get("Ġt")).toBe(256);
    expect(vocab.ids.get(END_OF_TEXT)).toBe(50_256);
  });

  test("starts from the 256 bytes in encoder.json's order", () => {
    expect(BASE_PIECES.length).toBe(256);
    expect(new Set(BASE_PIECES).size).toBe(256);
    expect(BASE_PIECES.slice(0, 3)).toEqual(["!", '"', "#"]);
    expect(toBytePieces(" ")).toEqual(["Ġ"]);
    expect(vocab.ids.get("Ġ")).toBe(220);
  });

  test("gives the same IDs as OpenAI's encoder.json for known text", () => {
    expect(idsOf("Hello world")).toEqual([15496, 995]);
    expect(idsOf("Lightning is a giant spark.")).toEqual([15047, 768, 318, 257, 6175, 9009, 13]);
    expect(idsOf(" unbelievably")).toEqual([48943]);
    expect(idsOf(" café")).toEqual([40304]);
  });

  test("splits words with GPT-2's rule and merges the lowest-ranked pair first", () => {
    expect(splitWords("It's a dog!")).toEqual(["It", "'s", " a", " dog", "!"]);
    const steps = mergeSteps(" the", vocab);
    expect(steps[0].pieces).toEqual(["Ġ", "t", "h", "e"]);
    expect(steps[1].rule).toEqual(["Ġ", "t"]);
    expect(steps[steps.length - 1].pieces).toEqual(["Ġthe"]);
    for (let i = 2; i < steps.length; i++) expect(steps[i].rank ?? 0).toBeGreaterThan(steps[i - 1].rank ?? -1);
  });

  test("handles empty text and text in any script", () => {
    expect(encodeText("", vocab)).toEqual([]);
    expect(idsOf("១២៣ 123").every((id) => id >= 0)).toBe(true);
  });
});

describe("learning merge rules", () => {
  const words = trainingWords("the dog and the dog and the cat");

  test("counts each pair once per time its word appears", () => {
    const top = countPairs(words)[0];
    expect(top).toEqual({ pair: ["t", "h"], count: 3 });
    expect(pieceCount(words)).toBe(new TextEncoder().encode("the dog and the dog and the cat").length);
  });

  test("joins the most common pair each round, and fewer pieces are left", () => {
    const { learned } = learnMerges(words, 4);
    expect(learned.length).toBe(4);
    for (let i = 1; i < learned.length; i++) expect(learned[i].pieceCount).toBeLessThan(learned[i - 1].pieceCount);
    expect(learned[0].pieceCount).toBe(pieceCount(words) - learned[0].count);
  });

  test("stops when every word is one piece", () => {
    expect(learnMerges(trainingWords("aa"), 10).learned.length).toBe(1);
    expect(learnMerges([], 3).learned).toEqual([]);
  });

  test("counting pairs one at a time ends with the same counts", () => {
    const rows = [toBytePieces("dog"), toBytePieces(" dog")];
    const places = pairPlaces(rows);
    expect(places.length).toBe(5);
    expect(countFirstPairs(rows, places).slice(0, 2)).toEqual([
      { pair: ["d", "o"], count: 2 },
      { pair: ["o", "g"], count: 2 },
    ]);
    expect(countFirstPairs(rows, places.slice(0, 1))).toEqual([{ pair: ["d", "o"], count: 1 }]);
  });
});

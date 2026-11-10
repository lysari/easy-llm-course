import { describe, expect, test } from "bun:test";
import { TEST_WORDS } from "@/core/model/corpus";
import { ModelLab } from "@/core/model/lab";
import { copyWeights } from "@/core/model/tiny-gpt";
import { chatPieces, preferenceGap, pushPreference, specialId } from "@/features/inside-real-models/story/post-training/post-training-math";

describe("Llama 3's chat layout", () => {
  test("the marker IDs come after the 128,000 regular words, in tokenizer.py's order", () => {
    expect(specialId("<|begin_of_text|>")).toBe(128_000);
    expect(specialId("<|end_of_text|>")).toBe(128_001);
    expect(specialId("<|start_header_id|>")).toBe(128_006);
    expect(specialId("<|end_header_id|>")).toBe(128_007);
    expect(specialId("<|eot_id|>")).toBe(128_009);
  });

  test("builds the prompt the way ChatFormat.encode_dialog_prompt does", () => {
    const pieces = chatPieces([{ role: "user", content: "  Hi there  " }]);
    expect(pieces.map((piece) => piece.text)).toEqual(["<|begin_of_text|>", "<|start_header_id|>", "user", "<|end_header_id|>", "\n\n", "Hi there", "<|eot_id|>", "<|start_header_id|>", "assistant", "<|end_header_id|>", "\n\n"]);
  });
});

describe("the preference nudge", () => {
  test("grows the gap between the better and the worse answer", () => {
    const lab = new ModelLab();
    lab.trainRounds(20);
    const weights = copyWeights(lab.live);
    const before = preferenceGap(TEST_WORDS, "stick", "bone", weights);
    pushPreference(weights, TEST_WORDS, "stick", "bone");
    expect(preferenceGap(TEST_WORDS, "stick", "bone", weights)).toBeGreaterThan(before);
  });
});

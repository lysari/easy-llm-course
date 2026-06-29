// RL-01 demo: tokenization strips meaning
//
// Run: npx ts-node reverse-lessons/RL-01-text-is-numbers/demo.ts
//
// We use character-level tokenization here (each character = one token)
// to keep it self-contained. Real models use BPE subword tokens,
// but the core point is the same: text → integers → meaning gone.

// Build a vocabulary from the characters we'll use
const text = "the cat sat on the mat";

const chars = Array.from(new Set(text.split(""))).sort();
const vocab: Record<string, number> = {};
chars.forEach((ch, i) => { vocab[ch] = i; });

function tokenize(s: string): number[] {
  return s.split("").map(ch => vocab[ch] ?? -1);
}

console.log("=== Vocabulary (character → integer) ===");
for (const [ch, id] of Object.entries(vocab)) {
  console.log(`  "${ch === " " ? "space" : ch}"  →  ${id}`);
}
console.log();

console.log("=== Tokenizing sentences ===");

const sentences = [
  "the cat sat on the mat",
  "the cat sat on the hat",  // "mat" → "hat": one character different
  "the dog sat on the mat",  // "cat" → "dog": completely different animal
];

for (const s of sentences) {
  const tokens = tokenize(s);
  console.log(`"${s}"`);
  console.log(`  → [${tokens.join(", ")}]`);
  console.log();
}

// Key demo: the model cannot tell "cat" and "dog" are both animals
// from the token IDs alone. The integers are just integers.
const catTokens = tokenize("cat");
const dogTokens = tokenize("dog");
const hatTokens = tokenize("hat");

console.log("=== Does tokenization carry meaning? ===");
console.log(`"cat" tokens: [${catTokens.join(", ")}]`);
console.log(`"dog" tokens: [${dogTokens.join(", ")}]`);
console.log(`"hat" tokens: [${hatTokens.join(", ")}]`);
console.log();
console.log("To a human: cat and dog are both animals; hat is not.");
console.log("To the tokenizer: they are all just different sequences of integers.");
console.log("The tokenizer has no concept of 'animal' or 'object'.");
console.log();

// Show that negation is invisible to the tokenizer
const happy = tokenize("happy");
const notHappy = tokenize("not happy");
console.log("=== Negation is invisible ===");
console.log(`"happy"     → [${happy.join(", ")}]`);
console.log(`"not happy" → [${notHappy.join(", ")}]`);
console.log(`"unhappy"   → (would be split into subwords in real BPE)`);
console.log();
console.log("The tokenizer does not know 'not' negates 'happy'.");
console.log("It just sees different integers.");
console.log("The model must learn this from patterns — it is NOT built in.");

// Tokenization: convert text into integers so a model can process it

class CharTokenizer {
  private charToId: Map<string, number> = new Map();
  private idToChar: Map<number, string> = new Map();

  constructor(text: string) {
    const chars = [...new Set(text.split(""))].sort();
    chars.forEach((c, i) => {
      this.charToId.set(c, i);
      this.idToChar.set(i, c);
    });
  }

  get vocabSize(): number {
    return this.charToId.size;
  }

  encode(text: string): number[] {
    return text.split("").map(c => {
      if (!this.charToId.has(c)) throw new Error(`Unknown token: "${c}"`);
      return this.charToId.get(c)!;
    });
  }

  decode(tokens: number[]): string {
    return tokens.map(t => this.idToChar.get(t) ?? "?").join("");
  }

  showVocab(): void {
    console.log(`Vocabulary (size ${this.vocabSize}):`);
    this.charToId.forEach((id, char) => {
      const display = char === "\n" ? "\\n" : char === " " ? "·" : char;
      process.stdout.write(`  "${display}"=${id}  `);
    });
    console.log();
  }
}

// Training corpus (tiny — normally you'd use a book or dataset)
const corpus = "the quick brown fox jumps over the lazy dog";
const tokenizer = new CharTokenizer(corpus);

tokenizer.showVocab();

// Encode
const text = "the fox";
const tokens = tokenizer.encode(text);
console.log(`\nEncoded "${text}":`, tokens);

// Decode
const decoded = tokenizer.decode(tokens);
console.log(`Decoded back: "${decoded}"`);
console.log(`Round-trip ok: ${decoded === text}`);

// Show next-token prediction pairs
console.log("\n=== Training pairs (input → target) ===");
const all = tokenizer.encode(corpus);
for (let i = 0; i < 8; i++) {
  const input = tokenizer.decode([all[i]!]);
  const target = tokenizer.decode([all[i + 1]!]);
  console.log(`  "${input}" (${all[i]}) → "${target}" (${all[i + 1]})`);
}

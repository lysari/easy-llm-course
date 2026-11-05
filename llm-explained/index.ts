const book = [
  "the cat ate the fish",
  "the cat sat on the mat",
  "the dog ate the bone",
];

const cups: Record<string, string[]> = {};

for (const sentence of book) {
  const words = sentence.split(" ");
  for (let i = 0; i < words.length - 1; i++) {
    const word = words[i];
    cups[word] ??= [];
    cups[word].push(words[i + 1]);
  }
}

console.log("Guessy's cups:");
for (const [word, slips] of Object.entries(cups)) {
  console.log(`  ${word}: ${slips.join(", ")}`);
}

const firstWord = book[0].split(" ")[0];

console.log("\nGuessy writes 5 sentences:");
for (let turn = 0; turn < 5; turn++) {
  const sentence = [firstWord];
  let cup = cups[firstWord];
  while (cup && sentence.length < 15) {
    const slip = cup[Math.floor(Math.random() * cup.length)];
    sentence.push(slip);
    cup = cups[slip];
  }
  console.log(`  ${sentence.join(" ")}`);
}

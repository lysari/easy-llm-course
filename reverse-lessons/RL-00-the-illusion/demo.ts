// RL-00 demo: the gap between what you see and what's inside
//
// Run: npx ts-node reverse-lessons/RL-00-the-illusion/demo.ts

// This is the sentence that "sounds like understanding":
const aiOutput = "The capital of France is Paris.";

// Now look at what the model actually worked with.
// Each word is not a word inside the model — it's an integer.
// These are approximate GPT-2 token IDs for illustration:
const tokenIds = [464, 3139, 286, 4881, 318, 6342, 13];
//                The capital of France is Paris  .

console.log("=== What you see ===");
console.log(aiOutput);
console.log();

console.log("=== What the model worked with ===");
console.log("Token IDs:", tokenIds);
console.log();

console.log("=== The gap ===");
console.log(`You see:        "${aiOutput}"`);
console.log(`Model sees:     [${tokenIds.join(", ")}]`);
console.log();
console.log("The model never read the word 'France'.");
console.log("It processed the integer 4881.");
console.log("It never 'knew' France is a country.");
console.log("It learned that 4881 is often followed by 318 then 6342.");
console.log();

// Let's make this even clearer with a nonsense substitution.
// If we replace "France" with a made-up word that has the same
// statistical position in sentences, the model would respond the same way.
const swapped = "The capital of Blorvia is ___";
console.log("=== Thought experiment ===");
console.log(`If the training data had: "${swapped}"`);
console.log("...and 'Krapton' always followed it...");
console.log("...the model would answer 'Krapton' with equal confidence.");
console.log();
console.log("The model does not know France exists.");
console.log("It knows that token 4881 is usually followed by token 6342.");
console.log("That's it. That's the whole trick.");

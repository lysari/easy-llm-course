// RL-04 demo: the model predicts, it does not know
//
// Run: npx ts-node reverse-lessons/RL-04-prediction-not-knowing/demo.ts

function softmax(logits: number[], temperature = 1.0): number[] {
  const scaled = logits.map(l => l / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map(l => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function sampleFromDistribution(probs: number[], vocab: string[]): string {
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i]!;
    if (r <= 0) return vocab[i]!;
  }
  return vocab[vocab.length - 1]!;
}

// Simulated vocab and logits for: "The capital of France is ___"
// These logits represent what a trained model might output after seeing that prompt.
// Higher logit = more consistent in training data.
const vocab = ["Paris", "Lyon", "Berlin", "London", "Madrid", "Rome", "Vienna"];
const logits = [8.2, 2.1, 0.5, 0.3, 0.1, 0.2, 0.1];
//              ↑ very high: "Paris" was consistently next in training data

console.log("=== Simulated output: 'The capital of France is ___' ===\n");

// Show at different temperatures
const temperatures = [0.3, 1.0, 2.0];
for (const temp of temperatures) {
  const probs = softmax(logits, temp);
  console.log(`Temperature = ${temp}:`);
  vocab.forEach((word, i) => {
    const bar = "█".repeat(Math.round(probs[i]! * 40));
    console.log(`  ${word.padEnd(8)} ${(probs[i]! * 100).toFixed(1).padStart(5)}%  ${bar}`);
  });
  console.log();
}

// Now show a "hallucination" scenario
console.log("=== Simulated output: 'The population of Australia in 1923 was ___' ===\n");
// This fact is rare/inconsistent in training data → flat distribution → any answer
const rareVocab = ["4", "5", "6", "7", "million", "thousand"];
const rareLogits = [2.1, 2.3, 2.0, 2.2, 3.1, 1.8];
// note: no dominant answer → model will say something but it's a guess

const rareProbs = softmax(rareLogits, 1.0);
console.log("Temperature = 1.0:");
rareVocab.forEach((word, i) => {
  const bar = "█".repeat(Math.round(rareProbs[i]! * 40));
  console.log(`  ${word.padEnd(10)} ${(rareProbs[i]! * 100).toFixed(1).padStart(5)}%  ${bar}`);
});
console.log();

// Sample 5 times to show randomness
console.log("Sampling 5 times from this distribution:");
for (let i = 0; i < 5; i++) {
  console.log(`  → "${sampleFromDistribution(rareProbs, rareVocab)}"`);
}
console.log();
console.log("Each run produces a different answer.");
console.log("The model is not retrieving a fact. It is sampling from a probability distribution.");
console.log("When that distribution is flat (uncertain training data), any token can win.");
console.log("The model states it with equal confidence regardless.");
console.log();
console.log("=== The core lesson ===");
console.log("Model confidence (high probability) = training data consistency");
console.log("Model confidence ≠ factual correctness");

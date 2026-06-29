// Softmax + cross-entropy: how a model outputs probabilities and measures its own error

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits); // subtract max for numerical stability
  const exps = logits.map(l => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function crossEntropyLoss(probs: number[], targetIndex: number): number {
  return -Math.log((probs[targetIndex] ?? 0) + 1e-9);
}

// Sample the next token: higher probability = more likely to be picked
function sample(probs: number[]): number {
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i] ?? 0;
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

// Temperature scaling: controls randomness
// low temp → confident (more peaked), high temp → creative (more uniform)
function softmaxWithTemp(logits: number[], temperature: number): number[] {
  return softmax(logits.map(l => l / temperature));
}

const vocab = ["the", "cat", "sat", "on", "mat"];
const logits = [1.0, 3.5, 0.5, -0.5, 2.0]; // model's raw output

console.log("=== Softmax ===");
const probs = softmax(logits);
vocab.forEach((word, i) => {
  const bar = "█".repeat(Math.round((probs[i] ?? 0) * 20));
  console.log(`  "${word}": ${((probs[i] ?? 0) * 100).toFixed(1).padStart(5)}%  ${bar}`);
});
console.log(`  Sum: ${probs.reduce((a, b) => a + b, 0).toFixed(6)}`);

// Loss when correct answer is "cat" (index 1) — low loss = good prediction
console.log("\n=== Cross-Entropy Loss ===");
console.log(`Loss for "cat" (index 1, prob=${(probs[1] ?? 0).toFixed(3)}): ${crossEntropyLoss(probs, 1).toFixed(4)}`);
console.log(`Loss for "on"  (index 3, prob=${(probs[3] ?? 0).toFixed(3)}): ${crossEntropyLoss(probs, 3).toFixed(4)}`);

// Temperature demo
console.log("\n=== Temperature Scaling ===");
for (const temp of [0.3, 1.0, 2.0]) {
  const p = softmaxWithTemp(logits, temp);
  console.log(`temp=${temp}: [${p.map(v => (v * 100).toFixed(1) + "%").join(", ")}]`);
}

// Sample the distribution 20 times
console.log("\n=== Sampling (20 times, temp=1.0) ===");
const samples = Array.from({ length: 20 }, () => vocab[sample(probs)]);
console.log(samples.join(", "));

// RL-05 demo: "learning" is adjusting numbers to minimize a loss
//
// Run: npx ts-node reverse-lessons/RL-05-weights-not-wisdom/demo.ts
//
// We train the simplest possible model to predict whether a token ID is
// "large" (>= 500) or "small" (< 500). This is meaningless classification,
// but it shows the exact same mechanism as GPT — gradient descent on numbers.

const LR = 0.1;

// Model: one weight + bias. Predicts P(large) given a token ID.
let weight = 0.0;   // starts at zero — no knowledge
let bias   = 0.0;

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function predict(tokenId: number): number {
  const normalized = tokenId / 1000;  // normalize input
  return sigmoid(normalized * weight + bias);
}

function loss(probLarge: number, label: number): number {
  // cross-entropy: -log(P(correct))
  return label === 1
    ? -Math.log(probLarge + 1e-10)
    : -Math.log(1 - probLarge + 1e-10);
}

// Training data: (tokenId, label) where label=1 if tokenId >= 500
const trainingData: [number, number][] = [
  [100, 0], [200, 0], [300, 0], [400, 0],
  [600, 1], [700, 1], [800, 1], [900, 1],
];

console.log("=== Training: adjusting numbers to minimize loss ===\n");
console.log(`Initial weight: ${weight.toFixed(4)},  bias: ${bias.toFixed(4)}`);
console.log(`Initial prediction for tokenId=750: ${(predict(750)*100).toFixed(1)}% (should be high)`);
console.log(`Initial prediction for tokenId=250: ${(predict(250)*100).toFixed(1)}% (should be low)`);
console.log();

for (let epoch = 0; epoch < 100; epoch++) {
  let totalLoss = 0;

  for (const [tokenId, label] of trainingData) {
    const normalized = tokenId / 1000;
    const p = predict(tokenId);
    totalLoss += loss(p, label);

    // Gradient descent: compute gradients manually
    // d_loss/d_output = p - label (for cross-entropy + sigmoid)
    const error = p - label;
    const dWeight = error * normalized;
    const dBias   = error;

    weight -= LR * dWeight;
    bias   -= LR * dBias;
  }

  if (epoch % 20 === 0) {
    console.log(`Epoch ${String(epoch).padStart(3)}: loss = ${(totalLoss / trainingData.length).toFixed(4)},  weight = ${weight.toFixed(4)},  bias = ${bias.toFixed(4)}`);
  }
}

console.log();
console.log("=== After training ===");
console.log(`Weight: ${weight.toFixed(4)},  Bias: ${bias.toFixed(4)}`);
console.log();

for (const [tokenId] of trainingData) {
  const p = predict(tokenId);
  const label = tokenId >= 500 ? 1 : 0;
  const correct = (p > 0.5) === (label === 1);
  console.log(`  tokenId=${tokenId}  →  P(large)=${(p*100).toFixed(1)}%  label=${label}  ${correct ? "✓" : "✗"}`);
}

console.log();
console.log("=== What just happened? ===");
console.log("The model started with weight=0 and bias=0 — no knowledge.");
console.log("Gradient descent adjusted those two numbers to minimize loss.");
console.log("The model now correctly classifies token IDs as large or small.");
console.log();
console.log("Did it learn what 'large' means? No.");
console.log("Did it understand numbers? No.");
console.log("Did it acquire any semantic knowledge? No.");
console.log();
console.log("It found two numbers (weight and bias) that minimize prediction error.");
console.log("GPT-4 does the exact same thing, with 1.8 trillion numbers instead of 2.");

// Backpropagation: compute gradients via chain rule, update all weights

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}
function sigmoidDerivative(s: number): number {
  return s * (1 - s); // s must already be sigmoid(z), not z
}

// Training data
const xData = [1, 2, 3, 4, 5, 6, 7];
const yData = [40, 50, 60, 70, 80, 100, 115];

// Scale y to (0,1) for sigmoid output layer
const yMax = Math.max(...yData);
const yNorm = yData.map(v => v / yMax);

const lr = 0.05;

// Network: 1 input → 3 hidden neurons → 1 output
// Hidden layer
let w1 = Math.random() - 0.5, b1 = 0.0;
let w2 = Math.random() - 0.5, b2 = 0.0;
let w3 = Math.random() - 0.5, b3 = 0.0;
// Output layer
let wO1 = Math.random() - 0.5;
let wO2 = Math.random() - 0.5;
let wO3 = Math.random() - 0.5;
let bOut = 0.0;

function forward(xi: number): { h1: number; h2: number; h3: number; out: number } {
  const h1 = sigmoid(w1 * xi + b1);
  const h2 = sigmoid(w2 * xi + b2);
  const h3 = sigmoid(w3 * xi + b3);
  const out = sigmoid(wO1 * h1 + wO2 * h2 + wO3 * h3 + bOut);
  return { h1, h2, h3, out };
}

function backward(xi: number, yi: number, h1: number, h2: number, h3: number, out: number): void {
  // Output error → output gradient
  const error = yi - out;
  const dOut = error * sigmoidDerivative(out);

  // Update output weights
  wO1 += lr * dOut * h1;
  wO2 += lr * dOut * h2;
  wO3 += lr * dOut * h3;
  bOut += lr * dOut;

  // Backprop through hidden layer (chain rule)
  const dH1 = dOut * wO1 * sigmoidDerivative(h1);
  const dH2 = dOut * wO2 * sigmoidDerivative(h2);
  const dH3 = dOut * wO3 * sigmoidDerivative(h3);

  w1 += lr * dH1 * xi; b1 += lr * dH1;
  w2 += lr * dH2 * xi; b2 += lr * dH2;
  w3 += lr * dH3 * xi; b3 += lr * dH3;
}

// Training loop
for (let epoch = 0; epoch < 50000; epoch++) {
  let totalError = 0;
  for (let i = 0; i < xData.length; i++) {
    const { h1, h2, h3, out } = forward(xData[i] ?? 0);
    totalError += (yNorm[i] ?? 0 - out) ** 2;
    backward(xData[i] ?? 0, yNorm[i] ?? 0, h1, h2, h3, out);
  }
  if (epoch % 10000 === 0) {
    console.log(`epoch ${epoch}: MSE=${(totalError / xData.length).toFixed(5)}`);
  }
}

// Predictions (scale back up)
console.log("\n=== Predictions ===");
for (let i = 0; i < xData.length; i++) {
  const { out } = forward(xData[i] ?? 0);
  const pred = (out * yMax).toFixed(1);
  console.log(`x=${xData[i]} → true=${yData[i]}, pred=${pred}`);
}

// Activation functions add non-linearity — without them, deep nets are just one linear function

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

function relu(z: number): number {
  return Math.max(0, z);
}

function tanhFn(z: number): number {
  return (Math.exp(z) - Math.exp(-z)) / (Math.exp(z) + Math.exp(-z));
}

// Derivatives (needed for backpropagation)
function sigmoidDerivative(s: number): number {
  return s * (1 - s); // s is sigmoid(z) already
}

function reluDerivative(z: number): number {
  return z > 0 ? 1 : 0;
}

// Compare activations across a range of inputs
console.log("z\t\tsigmoid\t\trelu\t\ttanh");
console.log("─".repeat(55));
for (const z of [-5, -2, -1, 0, 1, 2, 5]) {
  console.log(
    `${z}\t\t${sigmoid(z).toFixed(4)}\t\t${relu(z).toFixed(4)}\t\t${tanhFn(z).toFixed(4)}`
  );
}

// Vanishing gradient: sigmoid gradient near 0 for large inputs
console.log("\n=== Vanishing Gradient (sigmoid) ===");
for (const z of [-10, -5, 0, 5, 10]) {
  const s = sigmoid(z);
  console.log(`z=${z.toString().padStart(3)}: sigmoid=${s.toFixed(4)}, grad=${sigmoidDerivative(s).toFixed(6)}`);
}
// Notice: gradient at z=±10 is nearly 0 → learning stops in deep nets

// ReLU gradient is always 0 or 1 → no vanishing problem
console.log("\n=== ReLU gradient ===");
for (const z of [-2, -1, 0, 1, 2]) {
  console.log(`z=${z}: relu=${relu(z)}, grad=${reluDerivative(z)}`);
}

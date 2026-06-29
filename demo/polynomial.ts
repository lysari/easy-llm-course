import { exp } from 'mathjs';

// Our training data
const x = [1, 2, 3, 4, 5, 6, 7];
const y = [40, 50, 60, 70, 80, 100, 115];

// Small helper functions
function sigmoid(z: number): number {
  return 1 / (1 + exp(-z));
}
function sigmoidDerivative(z: number): number {
  return z * (1 - z);
}

// Random initial weights and biases
let w1 = Math.random(), w2 = Math.random(), w3 = Math.random(); // hidden layer
let b1 = Math.random(), b2 = Math.random(), b3 = Math.random();
let wOut1 = Math.random(), wOut2 = Math.random(), wOut3 = Math.random(); // output layer weights
let bOut = Math.random();

const learningRate = 0.01;

// Training loop
for (let epoch = 0; epoch < 3000000; epoch++) {
  let totalError = 0;

  for (let i = 0; i < x.length; i++) {
    const xi = x[i] || 0;
    const yi = (y[i] || 0) / 120; // scale output between 0–1 (important for sigmoid)

    // ---- Forward pass ----
    const h1 = sigmoid(w1 * xi + b1);
    const h2 = sigmoid(w2 * xi + b2);
    const h3 = sigmoid(w3 * xi + b3);
    const output = sigmoid(wOut1 * h1 + wOut2 * h2 + wOut3 * h3 + bOut);

    // ---- Error ----
    const error = yi - output;
    totalError += error ** 2;

    // ---- Backpropagation (manual update) ----
    const dOut = error * sigmoidDerivative(output);

    // Gradients for output weights
    wOut1 += learningRate * dOut * h1;
    wOut2 += learningRate * dOut * h2;
    wOut3 += learningRate * dOut * h3;
    bOut += learningRate * dOut;

    // Gradients for hidden layer
    const dH1 = dOut * wOut1 * sigmoidDerivative(h1);
    const dH2 = dOut * wOut2 * sigmoidDerivative(h2);
    const dH3 = dOut * wOut3 * sigmoidDerivative(h3);

    w1 += learningRate * dH1 * xi;
    w2 += learningRate * dH2 * xi;
    w3 += learningRate * dH3 * xi;
    b1 += learningRate * dH1;
    b2 += learningRate * dH2;
    b3 += learningRate * dH3;
  }

  if (epoch % 10000 === 0) {
    console.log(`Epoch ${epoch}, Error: ${(totalError / x.length).toFixed(5)}`);
  }
}

// ---- Test the model ----
function predict(xi: number): number {
  const h1 = sigmoid(w1 * xi + b1);
  const h2 = sigmoid(w2 * xi + b2);
  const h3 = sigmoid(w3 * xi + b3);
  const output = sigmoid(wOut1 * h1 + wOut2 * h2 + wOut3 * h3 + bOut);
  return output * 120; // scale back up
}

console.log("Prediction for x=1:", y[1-1] , predict(1).toFixed(2));
console.log("Prediction for x=2:", y[2-1] , predict(2).toFixed(2));
console.log("Prediction for x=3:", y[3-1] , predict(3).toFixed(2));
console.log("Prediction for x=4:", y[4-1] , predict(4).toFixed(2));
console.log("Prediction for x=5:", y[5-1] , predict(5).toFixed(2));
console.log("Prediction for x=6:", y[6-1] , predict(6).toFixed(2));
console.log("Prediction for x=7:", y[7-1], predict(7).toFixed(2));
console.log("Prediction for x=8:", "__", predict(8).toFixed(2));
console.log("Prediction for x=50:", "__", predict(50).toFixed(2));

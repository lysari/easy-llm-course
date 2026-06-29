// Gradient descent: iteratively move w and b to reduce the loss

const x = [1, 2, 3, 4, 5, 6, 7];
const y = [40, 50, 60, 70, 80, 100, 115];

let w = 0;
let b = 0;
const lr = 0.01;

function predict(xv: number): number {
  return w * xv + b;
}

function mse(): number {
  const n = x.length;
  return x.reduce((sum, xi, i) => sum + (predict(xi) - (y[i] ?? 0)) ** 2, 0) / n;
}

function step(): void {
  const n = x.length;
  let dw = 0;
  let db = 0;

  for (let i = 0; i < n; i++) {
    const error = predict(x[i] ?? 0) - (y[i] ?? 0);
    dw += error * (x[i] ?? 0);
    db += error;
  }

  // Move opposite to the gradient (downhill)
  w -= lr * (2 / n) * dw;
  b -= lr * (2 / n) * db;
}

console.log("=== Training ===");
for (let i = 0; i < 10000; i++) {
  step();
  if (i % 1000 === 0) {
    console.log(`iter ${i.toString().padStart(5)}: w=${w.toFixed(3)}, b=${b.toFixed(3)}, loss=${mse().toFixed(3)}`);
  }
}

console.log("\n=== Predictions ===");
for (const xi of [6, 7, 10, 15]) {
  console.log(`x=${xi} → ${predict(xi).toFixed(1)}`);
}

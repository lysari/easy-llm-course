// Linear regression: find the best line y = w*x + b through data points

const x = [1, 2, 3, 4, 5, 6, 7];
const y = [40, 50, 60, 70, 80, 100, 115];

function predict(xValue: number, w: number, b: number): number {
  return w * xValue + b;
}

// Try different lines and compare errors
const candidates = [
  { w: 10, b: 30 },
  { w: 12, b: 26 },
  { w: 13, b: 25 },
];

for (const { w, b } of candidates) {
  let totalError = 0;
  for (let i = 0; i < x.length; i++) {
    totalError += Math.abs(predict(x[i], w, b) - y[i]);
  }
  console.log(`w=${w}, b=${b} → total error: ${totalError.toFixed(1)}`);
}

// Print the best one's predictions
console.log("\n=== Best line (w=13, b=25) ===");
for (let i = 0; i < x.length; i++) {
  const pred = predict(x[i], 13, 25);
  console.log(`x=${x[i]} → pred=${pred}, true=${y[i]}, diff=${(pred - y[i]).toFixed(1)}`);
}

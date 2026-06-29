// A model is just a function: f(x) = output
// w (weight) and b (bias) are the parameters we learn

function predict(x: number, w: number, b: number): number {
  return w * x + b;
}

// Before training: guessing
console.log("=== Guessing parameters ===");
console.log("f(3) with w=5,  b=10:", predict(3, 5, 10));  // 25
console.log("f(3) with w=10, b=5: ", predict(3, 10, 5));  // 35
console.log("f(3) with w=12, b=4: ", predict(3, 12, 4));  // 40 ← closest to true answer

// The goal: find w and b so predict(x) ≈ true y for all x
// That search is what "training" means
const trueW = 12;
const trueB = 4;
const xs = [1, 2, 3, 4, 5];
const ys = [16, 28, 40, 52, 64]; // y = 12x + 4

console.log("\n=== With correct parameters ===");
for (let i = 0; i < xs.length; i++) {
  const pred = predict(xs[i], trueW, trueB);
  console.log(`x=${xs[i]} → pred=${pred}, true=${ys[i]}, match=${pred === ys[i]}`);
}

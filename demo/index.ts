import { LinearRegression } from "./linear.js";

// Step 1: our training data
const x = [1, 2, 3, 4, 5, 6, 7];
const y = [40, 50, 60, 70, 80, 100, 115];

// Step 2: model parameters (start from zero)
let w = 0; // slope
let b = 0; // intercept

// Step 3: learning rate (how big our steps are)
const learningRate = 0.01;

// Step 4: define a function to predict y from x
function predict(xValue: number): number {
  return w * xValue + b;
}

// Step 5: define a function to compute the gradients (dw, db)
function computeGradients(): { dw: number; db: number } {
  let dwSum = 0;
  let dbSum = 0;
  const n = x.length;

  for (let i = 0; i < n; i++) {
    const yPred = predict(x[i] ?? 0); // our current guess
    const error = yPred - (y[i] ?? 0); // how far we are from true y

    dwSum += error * (x[i] ?? 0); // slope part
    dbSum += error; // intercept part
  }

  // take average to make update stable
  return {
    dw: (2 / n) * dwSum,
    db: (2 / n) * dbSum,
  };
}

// Step 6: define a function to update w and b
function updateParameters(dw: number, db: number): void {
  w = w - learningRate * dw;
  b = b - learningRate * db;
}

// Step 7: define training loop
function train(iterations: number): void {
  for (let i = 0; i < iterations; i++) {
    const { dw, db } = computeGradients();
    updateParameters(dw, db);

    if (i % 100 === 0) {
      console.log(`Iteration ${i}: w=${w.toFixed(2)}, b=${b.toFixed(2)}`);
    }
  }
}

// Step 8: train model
train(10000);

// Step 9: test prediction
console.log("Prediction for x=6:", predict(6).toFixed(2));
console.log("Prediction for x=7:", predict(7).toFixed(2));
console.log("Prediction for x=10:", predict(10).toFixed(2));
console.log("Prediction for x=15:", predict(15).toFixed(2));
console.log("Prediction for x=50:", predict(50).toFixed(2));
console.log("Prediction for x=1000:", predict(1000).toFixed(2));

const model = new LinearRegression(x, y);
console.log("Model Prediction for x=6:", model.predict(6).toFixed(2));
console.log("Model Prediction for x=7:", model.predict(7).toFixed(2));
console.log("Model Prediction for x=10:", model.predict(10).toFixed(2));
console.log("Model Prediction for x=15:", model.predict(15).toFixed(2));
console.log("Model Prediction for x=50:", model.predict(50).toFixed(2));
console.log("Model Prediction for x=1000:", model.predict(1000).toFixed(2));
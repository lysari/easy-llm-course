// Loss function: a single number measuring how wrong the model is
// Lower = better. Training = minimizing this number.

function mse(predictions: number[], targets: number[]): number {
  const n = predictions.length;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += (predictions[i] - targets[i]) ** 2;
  }
  return sum / n;
}

function mae(predictions: number[], targets: number[]): number {
  const n = predictions.length;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += Math.abs(predictions[i] - targets[i]);
  }
  return sum / n;
}

const targets = [40, 50, 60, 70, 80];

// Perfect predictions → loss = 0
console.log("Perfect MSE:", mse([40, 50, 60, 70, 80], targets));

// All off by 10 → MSE = 100
console.log("Off by 10  MSE:", mse([50, 60, 70, 80, 90], targets));

// One huge error → MSE explodes (squared penalty)
console.log("One big error MSE:", mse([140, 50, 60, 70, 80], targets));
console.log("One big error MAE:", mae([140, 50, 60, 70, 80], targets));

// Notice: MSE punishes big errors much more than MAE
// That's why MSE pushes the model to avoid large mistakes

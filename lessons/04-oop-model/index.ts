// Wrap gradient descent into a reusable class
// Multiple independent models can coexist

class LinearRegression {
  private w = 0;
  private b = 0;

  constructor(
    private x: number[],
    private y: number[],
    private lr = 0.01,
    iterations = 10000
  ) {
    this.train(iterations);
  }

  predict(xv: number): number {
    return this.w * xv + this.b;
  }

  loss(): number {
    const n = this.x.length;
    return this.x.reduce((sum, xi, i) => sum + (this.predict(xi) - (this.y[i] ?? 0)) ** 2, 0) / n;
  }

  private train(iterations: number): void {
    const n = this.x.length;
    for (let i = 0; i < iterations; i++) {
      let dw = 0, db = 0;
      for (let j = 0; j < n; j++) {
        const error = this.predict(this.x[j] ?? 0) - (this.y[j] ?? 0);
        dw += error * (this.x[j] ?? 0);
        db += error;
      }
      this.w -= this.lr * (2 / n) * dw;
      this.b -= this.lr * (2 / n) * db;
    }
  }

  summary(): void {
    console.log(`w=${this.w.toFixed(3)}, b=${this.b.toFixed(3)}, loss=${this.loss().toFixed(3)}`);
  }
}

// Two independent models on different datasets
const model1 = new LinearRegression([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]);
const model2 = new LinearRegression([1, 2, 3, 4, 5, 6, 7], [40, 50, 60, 70, 80, 100, 115]);

console.log("Model 1 (y = 10x):"); model1.summary();
console.log("Model 2 (y ≈ 13x + 25):"); model2.summary();

console.log("\nModel 1 predict x=6:", model1.predict(6).toFixed(1));
console.log("Model 2 predict x=8:", model2.predict(8).toFixed(1));

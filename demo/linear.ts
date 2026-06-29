import { drawGraph, drawGraphD3 } from "./preview.js";

/**
 * A simple implementation of Linear Regression using Gradient Descent in TypeScript.
 * This code defines a LinearRegression class that can be used to fit a line to a set of data points (x, y).
 * The model parameters (slope and intercept) are updated iteratively using the computed gradients.
 * The learning rate controls the size of the steps taken during the parameter updates.
 * The class includes methods for predicting new values and training the model.
 * Usage:
 * const model = new LinearRegression(x, y);
 * model.predict(xValue);
 * model.train(iterations);
 */
export class LinearRegression {
  private x: number[];
  private y: number[];
  private w: number; // slope
  private b: number; // intercept
  private learningRate: number;
  private iterations: number = 10000;

  constructor(x: number[], y: number[], learningRate = 0.01) {
    this.x = x;
    this.y = y;
    this.w = 0;
    this.b = 0;
    this.learningRate = learningRate;
    this.train(this.iterations);
  }

  // Step 4: define a function to predict y from x
  // y = wx + b
  /**
   *
   * @param xValue {number} - The input value for which to predict the output.
   * @returns {number} - The predicted output value.
   */
  predict(xValue: number): number {
    return this.w * xValue + this.b;
  }

  // Step 5: define a function to compute the gradients (dw, db)
  // dw = dL/dw, db = dL/db
  // L = (1/n) * Σ(yPred - y)^2
  // dL/dw = (2/n) * Σ(yPred - y) * x
  // dL/db = (2/n) * Σ(yPred - y)
  // Returns the gradients for slope (dw) and intercept (db)
  private computeGradients(): { dw: number; db: number } {
    let dwSum = 0;
    let dbSum = 0;
    const n = this.x.length;

    for (let i = 0; i < n; i++) {
      const yPred = this.predict(this.x[i] ?? 0); // our current guess
      const error = yPred - (this.y[i] ?? 0); // how far we are from true y

      dwSum += error * (this.x[i] ?? 0); // slope part
      dbSum += error; // intercept part
    }

    // take average to make update stable
    return {
      dw: (2 / n) * dwSum,
      db: (2 / n) * dbSum,
    };
  }

  // Step 6: define a function to update w and b
  // w = w - learningRate * dw
  // b = b - learningRate * db
  private updateParameters(dw: number, db: number): void {
    this.w = this.w - this.learningRate * dw;
    this.b = this.b - this.learningRate * db;
  }

  // Step 7: define training loop
  train(iterations: number): void {
    for (let i = 0; i < iterations; i++) {
      const { dw, db } = this.computeGradients();
      this.updateParameters(dw, db);
      if (i === iterations - 1) {
        drawGraphD3(this.x, this.y, this.w, this.b);
        console.log(
          `Iteration ${i}: w=${this.w.toFixed(2)}, b=${this.b.toFixed(2)}`
        );
      }
    }
  }
}

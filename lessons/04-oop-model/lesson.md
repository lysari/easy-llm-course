# Lesson 04 — Wrapping Into a Class (OOP)

---

## The problem with the code so far

In Lesson 03, `w` and `b` are **global variables**:

```ts
let w = 0;
let b = 0;
```

This means there can only be **one model at a time**. If you want two models on different datasets, they'd share the same `w` and `b` and overwrite each other.

Also, the functions `predict()`, `computeGradients()`, `updateParameters()` are all floating loose in the file. They're not clearly connected to the data they work with.

---

## What is a class?

A **class** is like a **blueprint** for creating objects.

Real-world analogy: a cookie cutter.
- The cookie cutter = the class (the blueprint)
- Each cookie = an object (created from the blueprint)
- All cookies have the same shape, but different fillings (data)

```ts
class Dog {
  name: string;
  breed: string;

  constructor(name: string, breed: string) {
    this.name = name;
    this.breed = breed;
  }

  bark(): void {
    console.log(`${this.name} says: woof!`);
  }
}

const dog1 = new Dog("Rex", "Labrador");
const dog2 = new Dog("Buddy", "Poodle");

dog1.bark();  // Rex says: woof!
dog2.bark();  // Buddy says: woof!
```

`dog1` and `dog2` are independent — they don't share memory.

---

## Applying this to our model

```ts
class LinearRegression {
  private w = 0;   // ← w belongs to THIS model instance
  private b = 0;   // ← b belongs to THIS model instance

  constructor(x: number[], y: number[], lr = 0.01) {
    // constructor runs automatically when you do: new LinearRegression(...)
    this.train(x, y, lr, 10000);
  }

  predict(xValue: number): number {
    return this.w * xValue + this.b;
  }
}
```

**Key words:**
- `class` — declares the blueprint
- `constructor` — runs once when the object is created (via `new`)
- `private` — only code inside the class can access this variable
- `this.w` — the `w` that belongs to this specific object
- `new LinearRegression(x, y)` — creates one model

---

## `private` vs no keyword

```ts
class Example {
  private secret = 42;   // only accessible inside the class
  public name = "hi";    // accessible from anywhere (this is the default)
}

const e = new Example();
console.log(e.name);     // ✓ works
console.log(e.secret);   // ✗ TypeScript error: property is private
```

Making `w` and `b` private prevents accidents like:
```ts
model.w = 999;  // ✗ would corrupt the model
```

---

## Creating multiple independent models

```ts
const model1 = new LinearRegression([1,2,3], [10,20,30]);
const model2 = new LinearRegression([1,2,3], [40,50,60]);

model1.predict(5);  // uses model1's w and b
model2.predict(5);  // uses model2's w and b — completely independent!
```

`model1.w` and `model2.w` are separate variables. Changing one doesn't affect the other.

---

## The constructor calls train() automatically

```ts
constructor(x: number[], y: number[], lr = 0.01, iterations = 10000) {
  this.x = x;
  this.y = y;
  this.lr = lr;
  this.train(iterations);  // ← training happens here, at creation time
}
```

This means: by the time you get the model back, it's already trained:
```ts
// This one line: creates the model AND trains it for 10,000 iterations
const model = new LinearRegression([1,2,3], [40,50,60]);

// Already trained! Ready to predict:
console.log(model.predict(5));
```

---

## What is `this`?

`this` refers to **the current object**.

```ts
class LinearRegression {
  private w = 0;

  predict(xValue: number): number {
    return this.w * xValue + this.b;
    //     ^^^^^ means: the w that belongs to THIS specific model
  }
}
```

Without `this`, JavaScript wouldn't know which model's `w` to use when multiple models exist.

---

## Full class structure

```
LinearRegression
├── Properties (data the model stores)
│   ├── w (slope)
│   ├── b (intercept)
│   ├── x (training inputs)
│   ├── y (training targets)
│   └── learningRate
│
├── constructor(x, y, lr)  — called once at creation
│
└── Methods (things the model can do)
    ├── predict(xValue)     — make a prediction
    ├── computeGradients()  — (private) calculate dw, db
    ├── updateParameters()  — (private) update w and b
    ├── train(iterations)   — run the training loop
    └── summary()           — print w, b, and loss
```

---

## Before vs After

**Before (procedural — messy):**
```ts
let w = 0;
let b = 0;
function predict(x) { return w * x + b; }
function train() { ... }
train(10000);
console.log(predict(5));
```

**After (OOP — clean):**
```ts
const model = new LinearRegression(x, y);
console.log(model.predict(5));
```

One line to create a fully trained model. Clean and reusable.

---

## Code for this lesson

See [index.ts](index.ts) — creates two independent models on different datasets.

## What's next
[Lesson 05 → Activation Functions](../05-activation-functions/lesson.md)

# Exercise 21 — The Research-Debugging Drill

You debug best what you have broken yourself. In this drill you will take a healthy
training loop, plant three classic bugs **one at a time**, and record the *symptoms*
each one produces. After this, when a real experiment misbehaves, you'll recognize the
fingerprints.

---

## Part 1 — the checklist (keep this forever)

Copy this into your experiment log. Run it top to bottom on ANY failing experiment
**before** touching hyperparameters and **long before** concluding "the idea doesn't work":

```
RESEARCH-DEBUGGING CHECKLIST
────────────────────────────
DATA
 [ ] Printed 5 raw (input, label) pairs and checked them by eye
 [ ] Input ranges sane (no accidental 0–255 vs 0–1, no NaN)
 [ ] Class balance known (majority-class accuracy computed — Lesson 18)
 [ ] Training order shuffled every epoch

SHAPES
 [ ] Printed shapes at every pipeline stage
 [ ] Labels and inputs verifiably aligned (same index → same example)

LOSS WIRING
 [ ] Loss at step 0 ≈ theoretical chance value (2 classes: ln2 ≈ 0.693)
 [ ] No double softmax / double normalization

LEARNING RATE
 [ ] Swept lr in decades (1e-4 … 1) before blaming anything else
 [ ] Loss neither NaN/exploding (lr too big) nor flat (lr too small)

INIT
 [ ] Weights small random values, not zeros, not huge

EVAL
 [ ] Eval data never seen in training
 [ ] Metric function tested on tiny hand-made cases

THE VERDICT GATE
 [ ] Overfit-one-batch test PASSED (loss → ~0 on 4 examples)
     → only now am I allowed to say "the idea didn't work"
```

---

## Part 2 — set up the healthy patient

The network is a copy of [../../lessons/06-neural-network/index.ts](../../lessons/06-neural-network/index.ts)
(2 → 4 → 1 MLP). That lesson stopped at the forward pass, so we add the missing pieces —
a dataset, cross-entropy loss, and the backprop update from
[../../lessons/07-backpropagation/index.ts](../../lessons/07-backpropagation/index.ts).

Create `AI-researcher-lessons/21-negative-results/bugs-drill/index.ts` with this
**healthy** version, run it, and save the output — it is your reference for "what
healthy looks like":

```ts
// Healthy training loop: 2 → 4 → 1 MLP on two noisy blobs.
// Copy of lesson 06's network + a training loop. We will break it on purpose.

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

// Seeded RNG so every run is identical (Lesson 20: control your randomness!)
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rand: () => number): number {
  const u = Math.max(rand(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

const rand = mulberry32(21);

// ── Data: 40 points, two overlapping blobs. labels[i] belongs to inputs[i]. ──
const inputs: number[][] = [];
const labels: number[] = [];
for (let i = 0; i < 40; i++) {
  const y = i % 2;
  const c = y === 0 ? -1 : 1; // blob centers (-1,-1) and (1,1)
  inputs.push([c + gauss(rand) * 1.1, c + gauss(rand) * 1.1]);
  labels.push(y);
}
// Shuffle the dataset ONCE so storage order is random (like real data files).
for (let i = inputs.length - 1; i > 0; i--) {
  const j = Math.floor(rand() * (i + 1));
  [inputs[i], inputs[j]] = [inputs[j]!, inputs[i]!];
  [labels[i], labels[j]] = [labels[j]!, labels[i]!];
}

// BUG 3 will be planted here (label pairing).
const labs = labels.slice();

// BUG 2 will be planted here (training order).
const order = inputs.map((_, i) => i);

// ── Network: 2 → 4 → 1, small random init ──
const W1 = Array.from({ length: 4 }, () => [(rand() - 0.5) * 0.5, (rand() - 0.5) * 0.5]);
const b1 = [0, 0, 0, 0];
const W2 = Array.from({ length: 4 }, () => (rand() - 0.5) * 0.5);
let b2 = 0;

// BUG 1 will be planted here.
const lr = 0.5;
const SHUFFLE_EACH_EPOCH = true; // BUG 2 sets this to false (+ sorts by label)

for (let epoch = 0; epoch <= 40; epoch++) {
  if (SHUFFLE_EACH_EPOCH) {
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [order[i], order[j]] = [order[j]!, order[i]!];
    }
  }

  // one epoch of online SGD
  let loss = 0;
  for (const idx of order) {
    const x = inputs[idx] ?? [0, 0];
    const y = labs[idx] ?? 0;
    const h = W1.map((w, j) => sigmoid((w[0] ?? 0) * (x[0] ?? 0) + (w[1] ?? 0) * (x[1] ?? 0) + (b1[j] ?? 0)));
    const out = sigmoid(h.reduce((s, hj, j) => s + hj * (W2[j] ?? 0), b2));
    loss += -(y * Math.log(out + 1e-12) + (1 - y) * Math.log(1 - out + 1e-12));

    const dOut = out - y; // cross-entropy + sigmoid gradient
    for (let j = 0; j < 4; j++) {
      const hj = h[j] ?? 0;
      const dH = dOut * (W2[j] ?? 0) * hj * (1 - hj);
      W2[j] = (W2[j] ?? 0) - lr * dOut * hj;
      if (W1[j]) {
        W1[j]![0] = (W1[j]?.[0] ?? 0) - lr * dH * (x[0] ?? 0);
        W1[j]![1] = (W1[j]?.[1] ?? 0) - lr * dH * (x[1] ?? 0);
      }
      b1[j] = (b1[j] ?? 0) - lr * dH;
    }
    b2 -= lr * dOut;
  }
  loss /= order.length;

  // diagnostics: accuracy vs TRUE labels + how often the model predicts class 1
  if (epoch <= 3 || epoch % 10 === 0) {
    let correct = 0, ones = 0;
    for (let i = 0; i < inputs.length; i++) {
      const x = inputs[i] ?? [0, 0];
      const h = W1.map((w, j) => sigmoid((w[0] ?? 0) * (x[0] ?? 0) + (w[1] ?? 0) * (x[1] ?? 0) + (b1[j] ?? 0)));
      const out = sigmoid(h.reduce((s, hj, j) => s + hj * (W2[j] ?? 0), b2));
      const pred = out > 0.5 ? 1 : 0;
      if (pred === 1) ones++;
      if (pred === labels[i]) correct++;
    }
    console.log(
      `epoch ${String(epoch).padStart(3)}  loss ${isNaN(loss) ? "NaN " : loss.toFixed(4)}  ` +
      `acc ${((100 * correct) / inputs.length).toFixed(0)}%  predicts-1 ${((100 * ones) / inputs.length).toFixed(0)}%`
    );
  }
}
```

Run it: `npx ts-node index.ts`

Healthy reference (seed 21): loss falls from ~0.68 to below 0.1, accuracy settles
around 95–98%, and `predicts-1` hovers near 50% (the true class balance).

---

## Part 3 — plant the three bugs (ONE at a time!)

For each bug: edit → run → **write down the symptom** (loss curve shape, final accuracy,
the `predicts-1` column) → revert to healthy before planting the next. You are running
controlled experiments *on bugs* — one independent variable at a time (Lesson 17).

### Bug 1 — wrong learning-rate scale (two flavors)

```ts
const lr = 50;       // flavor A: 100× too big
const lr = 0.0005;   // flavor B: 1000× too small
```

Watch for: flavor A — loss explodes into the 5–9 range and *oscillates* forever;
accuracy jumps around between chance and okay-ish. Flavor B — loss glued near
ln 2 ≈ 0.693 and creeping down invisibly slowly; accuracy stuck near chance.
Note how B looks *exactly like* "my idea doesn't work". It isn't the idea. It's one
constant.

### Bug 2 — no shuffle (data sorted by class)

```ts
const SHUFFLE_EACH_EPOCH = false;
// and sort the training order by label:
order.sort((a, b) => (labs[a] ?? 0) - (labs[b] ?? 0));
```

Watch for: the `predicts-1` column in the first epochs — after each epoch the model is
biased toward the class it saw *last* (all class-0 first, then all class-1 → it leaves
the epoch predicting 1 for everything: `predicts-1 100%`, accuracy 50%). On this easy
task it eventually recovers; on hard tasks and bigger models this "catastrophic
forgetting within the epoch" costs real accuracy forever. The healthy run never shows
that early 100% bias.

### Bug 3 — label off-by-one (misaligned pairing)

```ts
const labs = labels.map((_, i) => labels[(i + 1) % labels.length] ?? 0);
```

Every input is now paired with its *neighbor's* label — the classic zip/indexing bug.
Watch for: loss wanders around 0.6–0.8 and never truly falls; accuracy vs the true
labels bounces around or below chance. The one-batch test would fail too: the labels
are (approximately) noise, and 4 hidden units can't memorize noise on 40 points.
The model is fine. The *pairing* is broken.

---

## Part 4 — write your symptom → cause table

Fill this in from YOUR observed runs (your numbers, not mine), then keep it next to the
checklist:

| Symptom observed | Prime suspect | First check |
|---|---|---|
| Loss explodes / NaN, wild oscillation | lr too big | divide lr by 10, rerun |
| Loss pinned near ln 2 ≈ 0.693, barely moves | lr too small — or no gradient flow | multiply lr by 10; check init |
| Early epochs: predicts one class 100%, then recovers | unshuffled, class-sorted data | print first 10 training labels |
| Loss ~0.6–0.8 forever, acc ≈ chance, one-batch test FAILS | labels misaligned with inputs | print 5 (input, label) pairs |
| Loss falls nicely, eval accuracy ≈ chance | eval bug / train-test mismatch | run eval fn on hand-made cases |
| Results too good to be true | test data leaked into training | check data split code |

(The last two rows aren't in the drill — plant them yourself if you want extra credit:
evaluate on `labs` instead of `labels`, or evaluate on the training subset.)

---

## Part 5 — the verdict gate, applied

For each of your three buggy runs, answer in one sentence:

1. Would this run pass the **overfit-one-batch test**? (Try it: slice the dataset to 4
   examples and train 500 epochs. Bug 3 fails it; bugs 1 and 2 pass once you fix the
   constant/order — what does that tell you about what KIND of problem each bug is?)
2. If you saw this symptom in a real project, what would you check FIRST, given the
   table you just built?
3. Which of the three bugs would be most dangerous in a paper — which one produces a
   plausible-looking *wrong* number rather than an obviously broken one?

Done? You now debug from symptoms, not from vibes. Save the checklist, the table, and
your notes in your experiment log.

---

## What's next

Phase D complete — you can design experiments, compare honestly, quantify uncertainty,
and debug failures. On to the frontiers: what is actually happening *inside* the
networks you train?
[Lesson 22 → Interpretability](../22-interpretability/lesson.md)

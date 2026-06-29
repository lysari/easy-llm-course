# Reverse Lesson 05 — Weights, Not Wisdom

---

## Layer to peel: learning → gradient descent

At every layer we've looked at, there were numbers that determined the model's behavior:

- Embedding table: the vectors for each token ID
- W_Q, W_K, W_V: the matrices that compute queries, keys, values
- Feed-forward weights: the matrices inside each FFN layer
- Output matrix: converts final vector to logits

Where do all these numbers come from?

**Training.** And training is not wisdom. It is optimization.

---

## What training is

Training has one job: **minimize a single number called the loss.**

The loss measures how wrong the model is, on average, across the training data.

```
loss = average of -log(probability assigned to the correct next token)
                              across all training examples
```

If the model assigns high probability to the correct next token, loss is low. If it assigns low probability, loss is high.

The process:

```
1. Feed a sequence of tokens into the model
2. Look at what token comes next in the training data (the "label")
3. See what probability the model assigned to that label
4. Compute loss: -log(probability of label)
5. Compute gradient: how should each weight change to reduce loss?
6. Update all weights slightly in that direction
7. Repeat. Billions of times.
```

That's it. No comprehension. No understanding. Just: make the numbers that minimize prediction error.

---

## What the model learned (and didn't learn)

After training on a trillion tokens of text, the model has learned:

**What it learned:** statistical regularities in token sequences
- "Paris" follows "capital of France is" with high frequency
- "not" often precedes negations
- Code comments are often followed by function definitions
- Sentences tend to end with punctuation

**What it did NOT learn:**
- What France is
- What capitals are
- What Paris looks like
- What it feels like to visit Paris
- Whether any statement it produces is true

The model has never experienced anything. It has only seen text.

---

## The comparison to a child

A child learns "hot" from one experience: touching a hot stove. After one burn, they understand:
- Fire can hurt you
- Hot things damage skin
- Pain is a signal to avoid something

The model was trained on 50,000 instances of the word "hot" in text. It learned:
- "hot" often appears near "temperature," "fire," "summer," "coffee"
- "hot" rarely appears near "cold," "ice," "winter"
- "hot and cold" is a common phrase

The model can describe heat, recommend you be careful near fire, and explain Celsius vs Fahrenheit. It has never felt warmth. It has no concept of pain. The words it produces about heat are the most statistically likely continuations of prompts about heat — not descriptions of an experience it had.

---

## Gradient descent has no intent

When gradient descent adjusts a weight from `0.312` to `0.314`, it does not decide:

> "I should understand geography better."

It adjusts the weight because that specific 0.002 change reduced the loss by a tiny amount on the current batch of training examples.

The model does not decide to become smarter. The optimizer does not represent goals or intentions. There is no learning in the human sense — no "now I understand." There is only: this set of numbers produces lower prediction error than the previous set.

---

## GPT-4 has 1.8 trillion parameters (estimated)

Every one of those parameters is a floating-point number, adjusted by gradient descent to minimize average prediction error on text.

The model that writes poetry, explains quantum physics, and debugs your code is:

```
1,800,000,000,000 numbers, each a float.
All chosen to minimize: -log(P(correct_next_token)).
```

There is no poetry module. There is no physics module. There is no code-understanding module.

There is one objective: predict text well. And predicting text well, at the scale of a trillion parameters trained on a trillion tokens, produces outputs that look like poetry, physics, and code understanding.

---

## The state so far

```
WHAT YOU SEE                    WHAT'S ACTUALLY THERE
────────────────────────────    ──────────────────────────────────
"The model learned"             Numbers adjusted to minimize loss
"The model understood"          Gradient descended to lower error
"The model knows things"        Weights that produce likely token sequences
"AI wisdom"                     1.8T floats chosen by backpropagation
```

---

## Run the demo

See [demo.ts](demo.ts) — runs a miniature gradient descent loop on a tiny "learning" task, showing exactly what "learning" is: adjusting numbers to reduce a loss, with no understanding involved.

---

## Next

[RL-06 → Ground Zero](../RL-06-ground-zero/lesson.md)

We've stripped every layer. What's left? One equation, applied billions of times. This is the final lesson.

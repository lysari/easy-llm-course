# LLM for Kids — The Word-Guessing Machine

Some computers can chat with you, write stories, and answer questions.

The thing inside them is called an **LLM**. That stands for Large Language Model.

It sounds hard. But its main job is a game you already know how to play.

---

## Part 1: Play the game

Finish these:

```
Twinkle, twinkle, little ___
Happy birthday to ___
One, two, three, four, ___
```

Easy, right? How did you know?

You have heard those words many times, so you could guess what comes next.

**That is what an LLM does.** It guesses the next word. It is a very, very big word-guessing machine.

---

## Part 2: Build a tiny one on paper

Let's build a small guessing machine. Its name is **Guessy**.

Guessy has only ever read this tiny book:

```
the cat ate the fish
the cat sat on the mat
the dog ate the bone
```

**You need:** 6 cups, 13 small pieces of paper, and a pen.

Look at each word in the book. Find the word right after it. Write that next word on a slip of paper, and put the slip in a cup with the first word's name.

For example, "the cat" is in the book. So write `cat` on a slip, and put it in the `the` cup.

When you finish, your cups look like this:

| Cup | Slips inside |
|-----|--------------|
| the | cat, fish, cat, mat, dog, bone |
| cat | ate, sat |
| dog | ate |
| ate | the, the |
| sat | on |
| on | the |

Look at the `the` cup. It has **two** `cat` slips, because the book says "the cat" two times. More slips means that word gets picked more often.

Now Guessy has learned from its book.

---

## Part 3: Let Guessy write

1. Write the word `the`.
2. Close your eyes and pull one slip from the `the` cup.
3. Write down that word and put the slip back.
4. Go to the cup with the name of your new word and pull again.
5. No cup for that word? The sentence is finished.

Here is one thing that can happen:

```
the → dog → ate → the → mat
```

**"the dog ate the mat"**

That is silly! And that sentence is not in the book. Guessy made it up.

Guessy does not know what a dog or a mat is. It only knows which words came after which. Remember this. It is important in Part 7.

Now play again. You will probably get a different sentence, because you pull the slips without looking. This is why an LLM can give different answers to the same question.

**Want the computer to play?** Ask a grown-up to run this:

```bash
bun llm-explained/index.ts
```

Then open [index.ts](index.ts), change the sentences in `book`, and run it again.

---

## Part 4: Upgrade 1 — Word Town

Guessy has three big problems. A real LLM fixes all of them.

**Problem:** Computers only work with numbers, and Guessy does not know that a cat and a dog are alike.

**Fix:** Give every word a home in Word Town. A home's address is made of numbers. Words that get used in the same way live close together.

```
  cat   dog                  rice   bread
     horse                      mango


                 bus   car
```

Now, if the machine reads "the dog is hungry", it can guess that "the cat is hungry" makes sense too. Cat lives next door!

Nobody draws this map. The machine finds the addresses by itself while it reads.

---

## Part 5: Upgrade 2 — The highlighter

**Problem:** Guessy only looks at one word, the last one. Try these:

```
My dog is hungry, so I give it a ___
My cat is hungry, so I give it a ___
```

You might say "bone" and then "fish". But the last word is `a` both times, so Guessy cannot tell them apart. The clue is further back.

**Fix:** A real LLM looks at **all** the words so far, and highlights the ones that matter most:

```
My [dog] is [hungry], so I give it a ___  →  bone
My [cat] is [hungry], so I give it a ___  →  fish
```

---

## Part 6: Upgrade 3 — The knobs

**Problem:** To remember everything people have ever written, you would need more cups than could ever fit in the world.

**Fix:** Throw away the cups. Use a giant board of tiny **knobs**, like the volume knob on a radio. A big LLM has billions of them. Words go in, the knobs change the numbers, and a guess comes out.

At first the knobs are set at random, so the guesses are terrible. Then the machine practises:

1. Take a real sentence and cover the next word.
2. Guess the covered word.
3. Look at the real word.
4. Turn the knobs a tiny bit, so the real word gets a better chance next time.

Have you played **Hot and Cold**? One person hides a toy, and the other person looks for it while hearing "warmer" or "colder". Nobody says where the toy is, but small steps get you there. The knobs learn the same way.

The machine does this again and again, with more text than you could read in thousands of lifetimes. That is why it needs huge computers.

---

## Part 7: How it answers you, and why it can be wrong

You type:

```
What colour is the sky?
```

To the machine, your question is just the beginning of some text. So it plays the game:

```
What colour is the sky?  →  The
What colour is the sky? The  →  sky
What colour is the sky? The sky  →  is
What colour is the sky? The sky is  →  blue
```

One guess at a time. It knows to answer because it has read many questions followed by answers, and because people practised with it and marked which answers were good.

Now remember **"the dog ate the mat"**.

A real LLM is much, much better than Guessy, but it is still guessing. On its own, it picks words that *sound* right. It does not stop to check that they *are* right. So it can sound very sure and still be wrong.

**Your rule:** when something is important, check it with a person you trust or a good book.

---

## Two big questions

**Is it alive?**
No. It is a huge pile of numbers inside computers. It does not eat, sleep, or get hungry, even when it writes about being hungry.

**Does it understand me?**
Grown-up scientists still argue about this one! What we know for sure is how it was built: to guess the next word. What do *you* think?

---

## Say it in one breath

> An LLM reads a huge amount of text, learns which words tend to come next, and then writes one guess at a time. Sounding right is not the same as being right.

If you can explain that to a friend, you understand LLMs better than most grown-ups.

---

## Check yourself

1. What is the main job of an LLM?
2. The book never says "the dog ate the mat". How did Guessy write it?
3. Why can't Guessy tell what comes next in "My dog is hungry, so I give it a ___"?

**Answers:**

1. Guessing the next word.
2. Each pair of words was in the book ("the dog", "dog ate", "ate the", "the mat"). Guessy joined them without knowing what they mean.
3. It only looks at the last word, `a`. The clue, `dog`, is further back.

---

## For the grown-up

**How to teach it:** Parts 1–3 take about 15 minutes with the cups, and they are the heart of the lesson. If that is all you get through, that is enough. Avoid the words "probability" and "algorithm" unless the child asks.

**To show the real process:** run the app in [app/](app/) with `cd llm-explained/app && bun install && bun run dev`. It shows how a small language model is built, trained and used, as a live flow of connected cards.

**What is simplified on purpose:**

- Guessy is a real but very old kind of model (a bigram model). Modern LLMs do not store cups. Part 6 tells the child this.
- Real training does not test each knob one by one like Hot and Cold. Backpropagation works out which way to turn every knob in one pass.
- Many chat products can also search the web or use other tools. "It does not stop to check" describes the model itself.

**When they want the real words:**

| In this lesson | Real name | Learn more |
|----------------|-----------|------------|
| Guessing the next word | next-token prediction | [Lesson 11](../lessons/11-softmax/lesson.md) |
| A word | token (a word or a piece of one) | [Lesson 09](../lessons/09-tokenization/lesson.md) |
| More slips, picked more often | probability distribution | [Lesson 11](../lessons/11-softmax/lesson.md) |
| Pulling a slip without looking | sampling | [Lesson 23](../lessons/23-sampling-strategies/lesson.md) |
| A home in Word Town | embedding | [Lesson 10](../lessons/10-embeddings/lesson.md) |
| The highlighter | attention | [Lesson 12](../lessons/12-attention/lesson.md) |
| The knobs | weights (parameters) | [Lesson 00](../lessons/00-what-is-a-model/lesson.md) |
| How far off the guess was | loss | [Lesson 02](../lessons/02-loss-function/lesson.md) |
| Turning the knobs a tiny bit | gradient descent, backpropagation | [Lesson 03](../lessons/03-gradient-descent/lesson.md), [Lesson 07](../lessons/07-backpropagation/lesson.md) |
| People marking good answers | RLHF | [Lesson 28](../lessons/28-rlhf-ppo/lesson.md) |
| Sounding sure and being wrong | hallucination | [RL-04](../reverse-lessons/RL-04-prediction-not-knowing/lesson.md) |

## What's next

Ready for the math? Go to [Lesson 00 → What Is a Model?](../lessons/00-what-is-a-model/lesson.md)

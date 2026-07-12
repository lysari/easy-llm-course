// Lesson 15 — Reproducing a Paper: Kaplan Scaling Laws in Miniature
//
// ── REPRODUCTION REPORT ────────────────────────────────────────────────────
// PAPER:  "Scaling Laws for Neural Language Models", Kaplan et al., 2020
//         (arXiv:2001.08361)
// CLAIM (one sentence, with a number): language-model loss falls as a power
//         law in non-embedding parameter count, L(N) ≈ a · N^(−b), with
//         b ≈ 0.076 over ~10^3..10^9 params — a straight line on log-log axes.
// WHAT WE REPRODUCE: the qualitative claim (tier 1, see lesson.md) — that a
//         power law fits loss-vs-params — at toy scale: 5 char-level models,
//         ~240..4,900 params, ~11KB of text, from scratch, <30s runtime,
//         each size averaged over 2 seeds (noise ≈ effect size down here).
// HYPERPARAMETER INVENTORY (STATED = from paper / GUESSED = our choice):
//         objective: next-token cross-entropy ......... STATED
//         model family: small dense nets .............. STATED (theirs:
//                        transformers; ours: MLP — a deliberate deviation,
//                        the claim is about capacity, not architecture)
//         vary N only; same data, steps, optimizer ..... STATED (their
//                        "no bottleneck from data/compute" regime)
//         optimizer Adam, lr 5e-3, 3000 steps, batch 32  GUESSED
//         context length 4 chars ....................... GUESSED
// VERDICT: printed at the bottom of the run.
// ───────────────────────────────────────────────────────────────────────────
//
// No imports. All math from scratch (see companion lessons 06–11 for every
// building block used here).

// ---------------------------------------------------------------------------
// Seeded pseudo-random for reproducibility
// ---------------------------------------------------------------------------

let _seed = 42;
function resetSeed(s: number): void {
  _seed = s;
}
function seededRandom(): number {
  // Mulberry32 — fast, good-quality 32-bit PRNG
  _seed |= 0;
  _seed = (_seed + 0x6d2b79f5) | 0;
  let t = Math.imul(_seed ^ (_seed >>> 15), 1 | _seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
function randn(): number {
  const u1 = seededRandom() + 1e-10;
  const u2 = seededRandom();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ---------------------------------------------------------------------------
// The fixed corpus (a few KB, embedded so the "dataset" is fully specified —
// no preprocessing ambiguity, the #1 cause of reproduction gaps)
// ---------------------------------------------------------------------------

const CORPUS = `
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream, ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country from whose bourn
No traveller returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all.

Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date;
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course untrimm'd;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st;
Nor shall death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe or eyes can see,
So long lives this, and this gives life to thee.

Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day,
To the last syllable of recorded time;
And all our yesterdays have lighted fools
The way to dusty death. Out, out, brief candle!
Life's but a walking shadow, a poor player,
That struts and frets his hour upon the stage,
And then is heard no more. It is a tale
Told by an idiot, full of sound and fury,
Signifying nothing.

Now is the winter of our discontent
Made glorious summer by this sun of York;
And all the clouds that lour'd upon our house
In the deep bosom of the ocean buried.
Now are our brows bound with victorious wreaths;
Our bruised arms hung up for monuments;
Our stern alarums changed to merry meetings,
Our dreadful marches to delightful measures.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious:
If it were so, it was a grievous fault,
And grievously hath Caesar answer'd it.
Here, under leave of Brutus and the rest -
For Brutus is an honourable man;
So are they all, all honourable men -
Come I to speak in Caesar's funeral.
He was my friend, faithful and just to me:
But Brutus says he was ambitious;
And Brutus is an honourable man.

All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts,
His acts being seven ages. At first, the infant,
Mewling and puking in the nurse's arms.
Then the whining schoolboy, with his satchel
And shining morning face, creeping like snail
Unwillingly to school. And then the lover,
Sighing like furnace, with a woeful ballad
Made to his mistress' eyebrow. Then a soldier,
Full of strange oaths and bearded like the pard,
Jealous in honour, sudden and quick in quarrel,
Seeking the bubble reputation
Even in the cannon's mouth.

When, in disgrace with fortune and men's eyes,
I all alone beweep my outcast state,
And trouble deaf heaven with my bootless cries,
And look upon myself and curse my fate,
Wishing me like to one more rich in hope,
Featured like him, like him with friends possess'd,
Desiring this man's art and that man's scope,
With what I most enjoy contented least;
Yet in these thoughts myself almost despising,
Haply I think on thee, and then my state,
Like to the lark at break of day arising
From sullen earth, sings hymns at heaven's gate;
For thy sweet love remember'd such wealth brings
That then I scorn to change my state with kings.

Let me not to the marriage of true minds
Admit impediments. Love is not love
Which alters when it alteration finds,
Or bends with the remover to remove:
O no! it is an ever-fixed mark
That looks on tempests and is never shaken;
It is the star to every wandering bark,
Whose worth's unknown, although his height be taken.
Love's not Time's fool, though rosy lips and cheeks
Within his bending sickle's compass come:
Love alters not with his brief hours and weeks,
But bears it out even to the edge of doom.
If this be error and upon me proved,
I never writ, nor no man ever loved.

My mistress' eyes are nothing like the sun;
Coral is far more red than her lips' red;
If snow be white, why then her breasts are dun;
If hairs be wires, black wires grow on her head.
I have seen roses damask'd, red and white,
But no such roses see I in her cheeks;
And in some perfumes is there more delight
Than in the breath that from my mistress reeks.
I love to hear her speak, yet well I know
That music hath a far more pleasing sound;
I grant I never saw a goddess go;
My mistress, when she walks, treads on the ground:
And yet, by heaven, I think my love as rare
As any she belied with false compare.

This day is called the feast of Crispian:
He that outlives this day, and comes safe home,
Will stand a tip-toe when this day is named,
And rouse him at the name of Crispian.
He that shall live this day, and see old age,
Will yearly on the vigil feast his neighbours,
And say 'To-morrow is Saint Crispian:'
Then will he strip his sleeve and show his scars,
And say 'These wounds I had on Crispin's day.'
Old men forget: yet all shall be forgot,
But he'll remember with advantages
What feats he did that day: then shall our names,
Familiar in his mouth as household words,
Harry the king, Bedford and Exeter,
Warwick and Talbot, Salisbury and Gloucester,
Be in their flowing cups freshly remember'd.
This story shall the good man teach his son;
And Crispin Crispian shall ne'er go by,
From this day to the ending of the world,
But we in it shall be remember'd;
We few, we happy few, we band of brothers;
For he to-day that sheds his blood with me
Shall be my brother; be he ne'er so vile,
This day shall gentle his condition:
And gentlemen in England now a-bed
Shall think themselves accursed they were not here,
And hold their manhoods cheap whiles any speaks
That fought with us upon Saint Crispin's day.

The quality of mercy is not strain'd,
It droppeth as the gentle rain from heaven
Upon the place beneath: it is twice blest;
It blesseth him that gives and him that takes:
'Tis mightiest in the mightiest: it becomes
The throned monarch better than his crown;
His sceptre shows the force of temporal power,
The attribute to awe and majesty,
Wherein doth sit the dread and fear of kings;
But mercy is above this sceptred sway;
It is enthroned in the hearts of kings,
It is an attribute to God himself;
And earthly power doth then show likest God's
When mercy seasons justice. Therefore, Jew,
Though justice be thy plea, consider this,
That, in the course of justice, none of us
Should see salvation: we do pray for mercy;
And that same prayer doth teach us all to render
The deeds of mercy.

Our revels now are ended. These our actors,
As I foretold you, were all spirits and
Are melted into air, into thin air:
And, like the baseless fabric of this vision,
The cloud-capp'd towers, the gorgeous palaces,
The solemn temples, the great globe itself,
Yea, all which it inherit, shall dissolve
And, like this insubstantial pageant faded,
Leave not a rack behind. We are such stuff
As dreams are made on, and our little life
Is rounded with a sleep.

Two households, both alike in dignity,
In fair Verona, where we lay our scene,
From ancient grudge break to new mutiny,
Where civil blood makes civil hands unclean.
From forth the fatal loins of these two foes
A pair of star-cross'd lovers take their life;
Whose misadventured piteous overthrows
Do with their death bury their parents' strife.
The fearful passage of their death-mark'd love,
And the continuance of their parents' rage,
Which, but their children's end, nought could remove,
Is now the two hours' traffic of our stage;
The which if you with patient ears attend,
What here shall miss, our toil shall strive to mend.

But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief,
That thou her maid art far more fair than she.
O, that she knew she were!
She speaks yet she says nothing: what of that?
Her eye discourses; I will answer it.
See, how she leans her cheek upon her hand!
O, that I were a glove upon that hand,
That I might touch that cheek!

If music be the food of love, play on;
Give me excess of it, that, surfeiting,
The appetite may sicken, and so die.
That strain again! it had a dying fall:
O, it came o'er my ear like the sweet sound,
That breathes upon a bank of violets,
Stealing and giving odour!

Why, man, he doth bestride the narrow world
Like a Colossus, and we petty men
Walk under his huge legs and peep about
To find ourselves dishonourable graves.
Men at some time are masters of their fates:
The fault, dear Brutus, is not in our stars,
But in ourselves, that we are underlings.

Is this a dagger which I see before me,
The handle toward my hand? Come, let me clutch thee.
I have thee not, and yet I see thee still.
Art thou not, fatal vision, sensible
To feeling as to sight? or art thou but
A dagger of the mind, a false creation,
Proceeding from the heat-oppressed brain?
I see thee yet, in form as palpable
As this which now I draw.
Thou marshall'st me the way that I was going;
And such an instrument I was to use.
Mine eyes are made the fools o' the other senses,
Or else worth all the rest; I see thee still,
And on thy blade and dudgeon gouts of blood,
Which was not so before. There's no such thing:
It is the bloody business which informs
Thus to mine eyes.

What a piece of work is a man! how noble in reason!
how infinite in faculty! in form and moving how
express and admirable! in action how like an angel!
in apprehension how like a god! the beauty of the
world! the paragon of animals! And yet, to me,
what is this quintessence of dust? man delights not
me: no, nor woman neither.
`.trim();

// ---------------------------------------------------------------------------
// Character-level tokenizer (companion lesson 09)
// ---------------------------------------------------------------------------

const vocab = Array.from(new Set(CORPUS.split(''))).sort();
const V = vocab.length;
const charToId = new Map(vocab.map((c, i) => [c, i]));
const data = CORPUS.split('').map(c => charToId.get(c)!);

const CONTEXT = 4; // predict char t from chars t-4..t-1
// (Two earlier attempts of this reproduction failed, and both failures were
// instructive — workflow step 5 happening for real:
//  1. context 4 on a 3.8KB corpus: the largest model MEMORIZED — most
//     5-grams were unique, so its val loss went UP. Data was the
//     bottleneck, violating the regime Kaplan et al. train in.
//  2. context 3 on the same corpus: the task was too EASY — every model
//     above ~800 params hit the corpus's conditional-entropy floor and the
//     curve went flat. Capacity wasn't the bottleneck either.
// Fix: a ~12KB corpus + context 4 keeps capacity the binding constraint
// across the whole size range — which is exactly the condition the paper
// states for the power law to hold.)

// Train/val split: every (context, target) position in the corpus is one
// example; we shuffle positions deterministically and hold out 10%.
// The split rule is part of the recipe — "evaluation details" is usual-
// suspect #2 in lesson.md. (First attempt of this reproduction used a
// contiguous tail split and FAILED: the tail is a different speech, so
// distribution shift swamped the capacity effect. Kaplan's L is in-
// distribution test loss, so the faithful split is a shuffled one.
// That failed run was step 5 of the workflow happening for real.)
const positions: number[] = [];
for (let t = 0; t + CONTEXT < data.length; t++) positions.push(t);
resetSeed(1234);
for (let i = positions.length - 1; i > 0; i--) {
  const j = Math.floor(seededRandom() * (i + 1));
  [positions[i], positions[j]] = [positions[j], positions[i]];
}
const valCount = Math.floor(positions.length * 0.1);
const valPos = positions.slice(0, valCount);
const trainPos = positions.slice(valCount);

// ---------------------------------------------------------------------------
// The model: context-window MLP  (embed → concat → tanh hidden → softmax)
// Only the WIDTH varies between runs; everything else is held fixed.
// ---------------------------------------------------------------------------

interface Model {
  d: number; // embedding dim
  h: number; // hidden units
  E: number[][]; // V × d
  W1: number[][]; // (CONTEXT*d) × h
  b1: number[]; // h
  W2: number[][]; // h × V
  b2: number[]; // V
}

function paramCount(m: Model): number {
  return (
    V * m.d + CONTEXT * m.d * m.h + m.h + m.h * V + V
  );
}

function initModel(d: number, h: number): Model {
  const scale1 = 1 / Math.sqrt(CONTEXT * d);
  const scale2 = 1 / Math.sqrt(h);
  const mk = (r: number, c: number, s: number) =>
    Array.from({ length: r }, () => Array.from({ length: c }, () => randn() * s));
  return {
    d,
    h,
    E: mk(V, d, 0.1),
    W1: mk(CONTEXT * d, h, scale1),
    b1: new Array(h).fill(0),
    W2: mk(h, V, scale2),
    b2: new Array(V).fill(0),
  };
}

// Forward pass for one example. Returns everything backward needs.
function forward(m: Model, ctx: number[]) {
  const e: number[] = [];
  for (const id of ctx) for (let j = 0; j < m.d; j++) e.push(m.E[id][j]);

  const hPre = new Array(m.h).fill(0);
  for (let j = 0; j < m.h; j++) {
    let s = m.b1[j];
    for (let i = 0; i < e.length; i++) s += e[i] * m.W1[i][j];
    hPre[j] = s;
  }
  const hAct = hPre.map(Math.tanh);

  const logits = new Array(V).fill(0);
  for (let k = 0; k < V; k++) {
    let s = m.b2[k];
    for (let j = 0; j < m.h; j++) s += hAct[j] * m.W2[j][k];
    logits[k] = s;
  }

  // stable softmax (companion lesson 11)
  const mx = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - mx));
  const Z = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(x => x / Z);

  return { e, hAct, probs };
}

function lossOf(probs: number[], target: number): number {
  return -Math.log(probs[target] + 1e-12);
}

// ---------------------------------------------------------------------------
// Gradients (hand-derived backprop, companion lessons 07 & 16) + Adam
// (companion lesson 17)
// ---------------------------------------------------------------------------

interface Grads {
  E: number[][];
  W1: number[][];
  b1: number[];
  W2: number[][];
  b2: number[];
}

function zerosLike(m: Model): Grads {
  const zr = (r: number, c: number) =>
    Array.from({ length: r }, () => new Array(c).fill(0));
  return {
    E: zr(V, m.d),
    W1: zr(CONTEXT * m.d, m.h),
    b1: new Array(m.h).fill(0),
    W2: zr(m.h, V),
    b2: new Array(V).fill(0),
  };
}

function backward(
  m: Model,
  g: Grads,
  ctx: number[],
  target: number,
  fwd: { e: number[]; hAct: number[]; probs: number[] }
): void {
  const { e, hAct, probs } = fwd;

  // d(loss)/d(logits) = probs - onehot(target)
  const dLogits = probs.slice();
  dLogits[target] -= 1;

  // W2, b2, and dh
  const dH = new Array(m.h).fill(0);
  for (let j = 0; j < m.h; j++) {
    for (let k = 0; k < V; k++) {
      g.W2[j][k] += hAct[j] * dLogits[k];
      dH[j] += m.W2[j][k] * dLogits[k];
    }
  }
  for (let k = 0; k < V; k++) g.b2[k] += dLogits[k];

  // through tanh: dPre = dH * (1 - h^2)
  const dPre = new Array(m.h).fill(0);
  for (let j = 0; j < m.h; j++) dPre[j] = dH[j] * (1 - hAct[j] * hAct[j]);

  // W1, b1, and dE
  for (let j = 0; j < m.h; j++) {
    if (dPre[j] === 0) continue;
    for (let i = 0; i < e.length; i++) g.W1[i][j] += e[i] * dPre[j];
    g.b1[j] += dPre[j];
  }
  for (let i = 0; i < e.length; i++) {
    let dEi = 0;
    for (let j = 0; j < m.h; j++) dEi += m.W1[i][j] * dPre[j];
    const pos = Math.floor(i / m.d); // which context slot
    const dim = i % m.d;
    g.E[ctx[pos]][dim] += dEi;
  }
}

// Adam state mirrors the gradient structure.
function trainModel(d: number, h: number, steps: number, batch: number, lr: number): Model {
  const m = initModel(d, h);
  const mom = zerosLike(m); // first moment
  const vel = zerosLike(m); // second moment
  const b1a = 0.9, b2a = 0.999, eps = 1e-8;

  // one flat updater applied to every (param, grad, m, v) matrix trio
  function adamUpdate(P: number[][] | number[], G: number[][] | number[], M: number[][] | number[], Vv: number[][] | number[], t: number, scale: number) {
    const corr1 = 1 - Math.pow(b1a, t);
    const corr2 = 1 - Math.pow(b2a, t);
    const upd = (p: number[], gr: number[], mo: number[], ve: number[]) => {
      for (let i = 0; i < p.length; i++) {
        const gi = gr[i] * scale;
        mo[i] = b1a * mo[i] + (1 - b1a) * gi;
        ve[i] = b2a * ve[i] + (1 - b2a) * gi * gi;
        p[i] -= (lr * (mo[i] / corr1)) / (Math.sqrt(ve[i] / corr2) + eps);
        gr[i] = 0;
      }
    };
    if (typeof (P as number[])[0] === 'number') {
      upd(P as number[], G as number[], M as number[], Vv as number[]);
    } else {
      const Pm = P as number[][], Gm = G as number[][], Mm = M as number[][], Vm = Vv as number[][];
      for (let r = 0; r < Pm.length; r++) upd(Pm[r], Gm[r], Mm[r], Vm[r]);
    }
  }

  const g = zerosLike(m);

  for (let step = 1; step <= steps; step++) {
    for (let b = 0; b < batch; b++) {
      const t0 = trainPos[Math.floor(seededRandom() * trainPos.length)];
      const ctx = data.slice(t0, t0 + CONTEXT);
      const target = data[t0 + CONTEXT];
      const fwd = forward(m, ctx);
      backward(m, g, ctx, target, fwd);
    }
    const scale = 1 / batch;
    adamUpdate(m.E, g.E, mom.E, vel.E, step, scale);
    adamUpdate(m.W1, g.W1, mom.W1, vel.W1, step, scale);
    adamUpdate(m.b1, g.b1, mom.b1, vel.b1, step, scale);
    adamUpdate(m.W2, g.W2, mom.W2, vel.W2, step, scale);
    adamUpdate(m.b2, g.b2, mom.b2, vel.b2, step, scale);
  }
  return m;
}

function evalLoss(m: Model, posList: number[]): number {
  let total = 0;
  for (const t of posList) {
    const fwd = forward(m, data.slice(t, t + CONTEXT));
    total += lossOf(fwd.probs, data[t + CONTEXT]);
  }
  return total / posList.length;
}

// ---------------------------------------------------------------------------
// Step 5 of the fit: ordinary least squares on (log N, log L)
// log L = log a − b · log N   →   slope = −b  (companion lesson 01)
// ---------------------------------------------------------------------------

function olsFit(xs: number[], ys: number[]) {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) * (xs[i] - mx);
    syy += (ys[i] - my) * (ys[i] - my);
  }
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  const r2 = syy === 0 ? 1 : (sxy * sxy) / (sxx * syy);
  return { slope, intercept, r2 };
}

// ---------------------------------------------------------------------------
// The experiment: 4 sizes, everything else held fixed
// ---------------------------------------------------------------------------

console.log('Reproduction target: Kaplan et al. 2020 — L(N) = a * N^(-b)');
console.log(`Corpus: ${CORPUS.length} chars, vocab ${V}, ` +
  `train ${trainPos.length} / val ${valPos.length} positions, context ${CONTEXT}\n`);

const SIZES = [
  { name: 'micro', d: 1, h: 2 },
  { name: 'tiny', d: 2, h: 4 },
  { name: 'small', d: 4, h: 8 },
  { name: 'medium', d: 8, h: 16 },
  { name: 'large', d: 16, h: 32 },
];

const STEPS = 3000;
const BATCH = 32;
const LR = 5e-3;
const SEEDS = [42, 1337]; // average over seeds: single-run noise at this
                          // scale is ~the size of the effect (Lesson 20!)

const results: { name: string; N: number; loss: number }[] = [];

for (const s of SIZES) {
  const t0 = Date.now();
  const seedLosses: number[] = [];
  let N = 0;
  for (const seed of SEEDS) {
    resetSeed(seed); // same init noise + data order across sizes per seed
    const m = trainModel(s.d, s.h, STEPS, BATCH, LR);
    seedLosses.push(evalLoss(m, valPos));
    N = paramCount(m);
  }
  const loss = seedLosses.reduce((x, y) => x + y, 0) / seedLosses.length;
  results.push({ name: s.name, N, loss });
  console.log(
    `trained ${s.name.padEnd(6)}  d=${String(s.d).padStart(2)} h=${String(s.h).padStart(2)}  ` +
    `params=${String(N).padStart(5)}  val loss=${loss.toFixed(4)} ` +
    `(seeds: ${seedLosses.map(l => l.toFixed(3)).join(', ')})  ` +
    `(${((Date.now() - t0) / 1000).toFixed(1)}s)`
  );
}

// ---------------------------------------------------------------------------
// Fit the power law and print the comparison table
// ---------------------------------------------------------------------------

const logN = results.map(r => Math.log(r.N));
const logL = results.map(r => Math.log(r.loss));
const { slope, intercept, r2 } = olsFit(logN, logL);
const b = -slope;
const a = Math.exp(intercept);

console.log('\nPower-law fit on (log N, log L):');
console.log(`  L(N) ≈ ${a.toFixed(3)} · N^(-${b.toFixed(4)})`);
console.log(`  fitted exponent b = ${b.toFixed(4)}   R² = ${r2.toFixed(4)}\n`);

console.log('model    params    measured L    power-law fit    diff');
console.log('------   ------    ----------    -------------    ------');
for (const r of results) {
  const pred = a * Math.pow(r.N, -b);
  console.log(
    `${r.name.padEnd(6)}   ${String(r.N).padStart(6)}    ` +
    `${r.loss.toFixed(4).padStart(10)}    ${pred.toFixed(4).padStart(13)}    ` +
    `${(r.loss - pred >= 0 ? '+' : '') + (r.loss - pred).toFixed(4)}`
  );
}

// A crude log-log "plot" so you can SEE the straight line
console.log('\nlog-log view (each column ≈ 0.02 in log L):');
const maxLogL = Math.max(...logL);
for (let i = 0; i < results.length; i++) {
  const bar = Math.max(1, Math.round((maxLogL - logL[i]) / 0.02));
  console.log(
    `  logN=${logN[i].toFixed(2)}  ${' '.repeat(bar)}* logL=${logL[i].toFixed(3)} (${results[i].name})`
  );
}

// ---------------------------------------------------------------------------
// The verdict (step 4–5 of the workflow: compare, then judge)
// ---------------------------------------------------------------------------

const monotone = results.every((r, i) => i === 0 || r.loss < results[i - 1].loss);
console.log('\n── VERDICT ────────────────────────────────────────────────');
console.log(`loss decreases monotonically with N: ${monotone ? 'YES' : 'NO'}`);
console.log(`power law fits well (R² > 0.9):      ${r2 > 0.9 ? 'YES' : 'NO'} (R²=${r2.toFixed(3)})`);
console.log(`fitted exponent b = ${b.toFixed(4)}  (Kaplan's, at real scale: ≈ 0.076)`);
if (monotone && r2 > 0.9 && b > 0) {
  console.log('\nTier-1 (qualitative) reproduction: SUCCESS.');
  console.log('The claim "loss follows a power law in parameter count"');
  console.log(`survives even at ${results[0].N}–${results[results.length - 1].N} parameters, in TypeScript,`);
  console.log('on a few KB of Shakespeare. Straight lines all the way down.');
} else {
  console.log('\nTier-1 reproduction: NOT CLEAN on this run.');
  console.log('Investigate (lesson.md step 5): more steps? different lr? new seed?');
  console.log('A reproduction that fails cleanly is a finding too — write it down.');
}

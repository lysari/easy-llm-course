// Capstone: one complete research cycle, end to end, in a single file.
//
// QUESTION    Does weight tying (sharing the embedding matrix with the output
//             projection) hurt or help a tiny character-level language model?
// HYPOTHESIS  Tying will NOT hurt: final validation loss (tied vs untied) will
//             differ by less than the seed-to-seed noise band.
// DESIGN      2 conditions × 3 seeds, everything else identical (data, split,
//             architecture, init order, steps, batch size, learning rate).
// ANALYSIS    mean ± std per condition, Welch's t-test (lesson 20's method).
//
// Run:  npx ts-node AI-researcher-lessons/30-capstone-research-project/index.ts

// ────────────────────────────────────────────────────────────────────────────
// 0. Seedable randomness — no reproducibility, no research (lesson 11)
// ────────────────────────────────────────────────────────────────────────────

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

// Gaussian samples via Box–Muller, driven by the seeded PRNG
function gaussian(rng: () => number): number {
  const u = Math.max(rng(), 1e-12);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ────────────────────────────────────────────────────────────────────────────
// 1. Data — a fixed public-domain corpus, char-level (same for every run)
// ────────────────────────────────────────────────────────────────────────────

const CORPUS = (
  "First Citizen: Before we proceed any further, hear me speak. " +
  "All: Speak, speak. " +
  "First Citizen: You are all resolved rather to die than to famish? " +
  "All: Resolved, resolved. " +
  "First Citizen: First, you know Caius Marcius is chief enemy to the people. " +
  "All: We know it, we know it. " +
  "First Citizen: Let us kill him, and we will have corn at our own price. " +
  "Is it a verdict? " +
  "All: No more talking on it; let it be done: away, away! " +
  "Second Citizen: One word, good citizens. " +
  "First Citizen: We are accounted poor citizens, the patricians good. " +
  "What authority surfeits on would relieve us: if they would yield us " +
  "but the superfluity, while it were wholesome, we might guess they " +
  "relieved us humanely; but they think we are too dear: the leanness " +
  "that afflicts us, the object of our misery, is as an inventory to " +
  "particularise their abundance; our sufferance is a gain to them. " +
  "Let us revenge this with our pikes, ere we become rakes: for the gods " +
  "know I speak this in hunger for bread, not in thirst for revenge. " +
  "Second Citizen: Would you proceed especially against Caius Marcius? " +
  "All: Against him first: he is a very dog to the commonalty. " +
  "Second Citizen: Consider you what services he has done for his country? " +
  "First Citizen: Very well; and could be content to give him good report " +
  "for it, but that he pays himself with being proud. " +
  "Second Citizen: Nay, but speak not maliciously. " +
  "First Citizen: I say unto you, what he hath done famously, he did it to " +
  "that end: though soft-conscienced men can be content to say it was for " +
  "his country he did it to please his mother and to be partly proud; which " +
  "he is, even till the altitude of his virtue. " +
  "Second Citizen: What he cannot help in his nature, you account a vice in " +
  "him. You must in no way say he is covetous. " +
  "First Citizen: If I must not, I need not be barren of accusations; he " +
  "hath faults, with surplus, to tire in repetition. What shouts are these? " +
  "The other side of the city is risen: why stay we prating here? to the " +
  "Capitol! " +
  "All: Come, come. " +
  "First Citizen: Soft! who comes here? " +
  "Second Citizen: Worthy Menenius Agrippa; one that hath always loved the " +
  "people. " +
  "First Citizen: He is one honest enough: would all the rest were so! " +
  "Menenius: What work is done, my countrymen, in hand? where go you with " +
  "bats and clubs? The matter speak, I pray you. " +
  "First Citizen: Our business is not unknown to the senate; they have had " +
  "inkling this fortnight what we intend to do, which now we will show them " +
  "in deeds. They say poor suitors have strong breaths: they shall know we " +
  "have strong arms too. " +
  "Menenius: Why, masters, my good friends, mine honest neighbours, will " +
  "you undo yourselves? " +
  "First Citizen: We cannot, sir, we are undone already. " +
  "Menenius: I tell you, friends, most charitable care have the patricians " +
  "of you. For your wants, your suffering in this dearth, you may as well " +
  "strike at the heaven with your staves as lift them against the state; " +
  "whose course will on the way it takes, cracking ten thousand curbs of " +
  "more strong link asunder than can ever appear in your impediment. For " +
  "the dearth, the gods, not the patricians, make it, and your knees to " +
  "them, not arms, must help. Alack, you are transported by calamity " +
  "thither where more attends you, and you slander the helms of the state, " +
  "who care for you like fathers, when you curse them as enemies. " +
  "First Citizen: Care for us! True, indeed! They never cared for us yet: " +
  "suffer us to famish, and their storehouses crammed with grain; make " +
  "edicts for usury, to support usurers; repeal daily any wholesome act " +
  "established against the rich, and provide more piercing statutes daily " +
  "to chain up and restrain the poor. If the wars eat us not up, they " +
  "will; and there is all the love they bear us. " +
  "Menenius: Either you must confess yourselves wondrous malicious, or be " +
  "accused of folly. I shall tell you a pretty tale: it may be you have " +
  "heard it; but, since it serves my purpose, I will venture to stale it a " +
  "little more. " +
  "First Citizen: Well, I will hear it, sir: yet you must not think to fob " +
  "off our disgrace with a tale: but, if it please you, deliver. "
).toLowerCase();

const chars = Array.from(new Set(CORPUS.split(""))).sort();
const V = chars.length; // vocab size
const stoi = new Map(chars.map((c, i) => [c, i]));
const data = CORPUS.split("").map(c => stoi.get(c)!);

// 90/10 train/validation split — the val slice is NEVER trained on
const splitAt = Math.floor(data.length * 0.9);
const train = data.slice(0, splitAt);
const val = data.slice(splitAt);

// ────────────────────────────────────────────────────────────────────────────
// 2. Model — tiny char-LM: context of T chars → MLP → next-char logits
//
//    x  = concat of T char embeddings            (T·d)
//    h  = tanh(W1ᵀx + b1)                        (H)
//    z  = W2ᵀh + b2                              (d)   ← back to embedding dim
//    u  = z · Wout                               (V)   ← logits
//
//    UNTIED: Wout is its own (d × V) matrix.
//    TIED:   Wout IS Eᵀ — the same numbers as the embedding table.
//    That storage-sharing is the ONLY difference between conditions.
// ────────────────────────────────────────────────────────────────────────────

const T = 4;   // context length (chars)
const D = 16;  // embedding dim
const H = 48;  // hidden units
const STEPS = 3000;
const BATCH = 16;
const LR = 0.25;

type Matrix = number[][];

interface Model {
  tied: boolean;
  E: Matrix;      // (V × d) embedding table
  W1: Matrix;     // (T·d × H)
  b1: number[];
  W2: Matrix;     // (H × d)
  b2: number[];
  Wout?: Matrix;  // (d × V) — only exists when untied
}

function zeros(r: number, c: number): Matrix {
  return Array.from({ length: r }, () => Array<number>(c).fill(0));
}

function initMatrix(r: number, c: number, scale: number, rng: () => number): Matrix {
  return Array.from({ length: r }, () =>
    Array.from({ length: c }, () => gaussian(rng) * scale)
  );
}

function initModel(tied: boolean, rng: () => number): Model {
  // Identical draw order for shared params → same seed = same starting point;
  // the untied model just draws one extra matrix at the end.
  const m: Model = {
    tied,
    E: initMatrix(V, D, 0.1, rng),
    W1: initMatrix(T * D, H, 1 / Math.sqrt(T * D), rng),
    b1: Array(H).fill(0),
    W2: initMatrix(H, D, 1 / Math.sqrt(H), rng),
    b2: Array(D).fill(0),
  };
  if (!tied) m.Wout = initMatrix(D, V, 0.1, rng);
  return m;
}

function paramCount(m: Model): number {
  return V * D + T * D * H + H + H * D + D + (m.tied ? 0 : D * V);
}

// Forward pass for one example; returns loss and everything backprop needs
function forward(m: Model, ctx: number[], target: number) {
  const x: number[] = [];
  for (const id of ctx) for (let j = 0; j < D; j++) x.push(m.E[id]![j]!);

  const h = Array(H).fill(0);
  for (let k = 0; k < H; k++) {
    let s = m.b1[k]!;
    for (let i = 0; i < T * D; i++) s += x[i]! * m.W1[i]![k]!;
    h[k] = Math.tanh(s);
  }

  const z = Array(D).fill(0);
  for (let j = 0; j < D; j++) {
    let s = m.b2[j]!;
    for (let k = 0; k < H; k++) s += h[k]! * m.W2[k]![j]!;
    z[j] = s;
  }

  // logits: tied reads E as the output matrix, untied reads Wout
  const u = Array(V).fill(0);
  for (let v = 0; v < V; v++) {
    let s = 0;
    for (let j = 0; j < D; j++) s += z[j]! * (m.tied ? m.E[v]![j]! : m.Wout![j]![v]!);
    u[v] = s;
  }

  const maxU = Math.max(...u);
  const exps = u.map(a => Math.exp(a - maxU));
  const sum = exps.reduce((a, b) => a + b, 0);
  const p = exps.map(e => e / sum);
  const loss = -Math.log(Math.max(p[target]!, 1e-12));
  return { x, h, z, p, loss };
}

interface Grads { E: Matrix; W1: Matrix; b1: number[]; W2: Matrix; b2: number[]; Wout: Matrix; }

function newGrads(): Grads {
  return { E: zeros(V, D), W1: zeros(T * D, H), b1: Array(H).fill(0),
           W2: zeros(H, D), b2: Array(D).fill(0), Wout: zeros(D, V) };
}

// Backward pass (manual backprop — you built each rule in lessons 07 & 16)
function backward(m: Model, g: Grads, ctx: number[], target: number,
                  f: ReturnType<typeof forward>) {
  const du = f.p.slice();
  du[target]! -= 1; // d(loss)/d(logits) = p − one_hot(target)

  // Output layer. TIED: gradient flows into E from the output side too —
  // the embedding table gets TWO gradient contributions per example.
  const dz = Array(D).fill(0);
  for (let j = 0; j < D; j++) {
    for (let v = 0; v < V; v++) {
      const w = m.tied ? m.E[v]![j]! : m.Wout![j]![v]!;
      dz[j]! += w * du[v]!;
      if (m.tied) g.E[v]![j]! += f.z[j]! * du[v]!;
      else g.Wout[j]![v]! += f.z[j]! * du[v]!;
    }
  }

  const dh = Array(H).fill(0);
  for (let j = 0; j < D; j++) {
    g.b2[j]! += dz[j]!;
    for (let k = 0; k < H; k++) {
      g.W2[k]![j]! += f.h[k]! * dz[j]!;
      dh[k]! += m.W2[k]![j]! * dz[j]!;
    }
  }

  const dx = Array(T * D).fill(0);
  for (let k = 0; k < H; k++) {
    const dpre = dh[k]! * (1 - f.h[k]! * f.h[k]!); // tanh'
    g.b1[k]! += dpre;
    for (let i = 0; i < T * D; i++) {
      g.W1[i]![k]! += f.x[i]! * dpre;
      dx[i]! += m.W1[i]![k]! * dpre;
    }
  }

  // Embedding gradient from the input side
  for (let t = 0; t < T; t++)
    for (let j = 0; j < D; j++)
      g.E[ctx[t]!]![j]! += dx[t * D + j]!;
}

function sgdStep(m: Model, g: Grads, lr: number, batch: number) {
  const s = lr / batch;
  const upd = (M: Matrix, G: Matrix) => {
    for (let i = 0; i < M.length; i++)
      for (let j = 0; j < M[0]!.length; j++) M[i]![j]! -= s * G[i]![j]!;
  };
  upd(m.E, g.E); upd(m.W1, g.W1); upd(m.W2, g.W2);
  if (!m.tied) upd(m.Wout!, g.Wout);
  for (let k = 0; k < H; k++) m.b1[k]! -= s * g.b1[k]!;
  for (let j = 0; j < D; j++) m.b2[j]! -= s * g.b2[j]!;
}

// ────────────────────────────────────────────────────────────────────────────
// 3. The run — one training run = one (condition, seed) cell
// ────────────────────────────────────────────────────────────────────────────

function valLoss(m: Model): number {
  let total = 0, n = 0;
  for (let i = T; i < val.length; i++) {
    total += forward(m, val.slice(i - T, i), val[i]!).loss;
    n++;
  }
  return total / n;
}

function trainRun(tied: boolean, seed: number): number {
  const rng = mulberry32(seed);
  const m = initModel(tied, rng);
  for (let step = 0; step < STEPS; step++) {
    const g = newGrads();
    for (let b = 0; b < BATCH; b++) {
      const i = T + Math.floor(rng() * (train.length - T));
      const ctx = train.slice(i - T, i);
      backward(m, g, ctx, train[i]!, forward(m, ctx, train[i]!));
    }
    sgdStep(m, g, LR, BATCH);
  }
  return valLoss(m);
}

// ────────────────────────────────────────────────────────────────────────────
// 4. Statistics — mean ± std and Welch's t-test (lesson 20's method)
// ────────────────────────────────────────────────────────────────────────────

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
const std = (xs: number[]) => {
  const mu = mean(xs);
  return Math.sqrt(xs.reduce((a, x) => a + (x - mu) ** 2, 0) / (xs.length - 1));
};

function welch(a: number[], b: number[]): { t: number; df: number; p: number } {
  const va = std(a) ** 2 / a.length, vb = std(b) ** 2 / b.length;
  const t = (mean(a) - mean(b)) / Math.sqrt(va + vb);
  const df = (va + vb) ** 2 /
    (va ** 2 / (a.length - 1) + vb ** 2 / (b.length - 1)); // Welch–Satterthwaite
  return { t, df, p: tTwoSidedP(Math.abs(t), df) };
}

// Two-sided p-value for Student's t via the regularized incomplete beta:
// p = I_{df/(df+t²)}(df/2, 1/2)
function tTwoSidedP(t: number, df: number): number {
  return regIncBeta(df / (df + t * t), df / 2, 0.5);
}

function logGamma(x: number): number { // Lanczos approximation
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
             -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x, tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) ser += c[j]! / ++y;
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}

function regIncBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const ln = logGamma(a + b) - logGamma(a) - logGamma(b) +
             a * Math.log(x) + b * Math.log(1 - x);
  const front = Math.exp(ln);
  // continued fraction (Numerical Recipes betacf)
  const cf = (x0: number, a0: number, b0: number): number => {
    const qab = a0 + b0, qap = a0 + 1, qam = a0 - 1;
    let c1 = 1, d1 = 1 - (qab * x0) / qap;
    if (Math.abs(d1) < 1e-30) d1 = 1e-30;
    d1 = 1 / d1;
    let h = d1;
    for (let m1 = 1; m1 <= 200; m1++) {
      const m2 = 2 * m1;
      let aa = (m1 * (b0 - m1) * x0) / ((qam + m2) * (a0 + m2));
      d1 = 1 + aa * d1; if (Math.abs(d1) < 1e-30) d1 = 1e-30;
      c1 = 1 + aa / c1; if (Math.abs(c1) < 1e-30) c1 = 1e-30;
      d1 = 1 / d1; h *= d1 * c1;
      aa = (-(a0 + m1) * (qab + m1) * x0) / ((a0 + m2) * (qap + m2));
      d1 = 1 + aa * d1; if (Math.abs(d1) < 1e-30) d1 = 1e-30;
      c1 = 1 + aa / c1; if (Math.abs(c1) < 1e-30) c1 = 1e-30;
      d1 = 1 / d1;
      const del = d1 * c1;
      h *= del;
      if (Math.abs(del - 1) < 3e-9) break;
    }
    return h;
  };
  return x < (a + 1) / (a + b + 2)
    ? (front * cf(x, a, b)) / a
    : 1 - (front * cf(1 - x, b, a)) / b;
}

// ────────────────────────────────────────────────────────────────────────────
// 5. The experiment — 2 conditions × 3 seeds, then the honest verdict
// ────────────────────────────────────────────────────────────────────────────

console.log("=== Capstone experiment: weight tying in a tiny char-LM ===");
console.log(`corpus: ${data.length} chars, vocab ${V} | train ${train.length} / val ${val.length}`);
console.log(`model: ctx=${T}, d=${D}, hidden=${H} | ${STEPS} steps, batch ${BATCH}, lr ${LR}`);

const SEEDS = [1, 2, 3];
const results: Record<string, number[]> = { tied: [], untied: [] };

const t0 = Date.now();
for (const tied of [true, false]) {
  const name = tied ? "tied" : "untied";
  const params = paramCount(initModel(tied, mulberry32(0)));
  console.log(`\n── condition: ${name} (${params} parameters) ──`);
  for (const seed of SEEDS) {
    const loss = trainRun(tied, seed);
    results[name]!.push(loss);
    console.log(`  seed ${seed}: final val loss = ${loss.toFixed(4)}`);
  }
}
console.log(`\n(total runtime: ${((Date.now() - t0) / 1000).toFixed(1)}s)`);

// Results table: mean ± std — never a bare mean (lesson 20)
console.log("\n=== Results ===");
console.log("condition | val loss (mean ± std, 3 seeds)");
for (const name of ["tied", "untied"]) {
  const r = results[name]!;
  console.log(`${name.padEnd(9)} | ${mean(r).toFixed(4)} ± ${std(r).toFixed(4)}`);
}

const { t, df, p } = welch(results.tied!, results.untied!);
const gap = mean(results.tied!) - mean(results.untied!);
console.log(`\nWelch's t-test: t = ${t.toFixed(3)}, df = ${df.toFixed(2)}, two-sided p = ${p.toFixed(3)}`);

// The automated, pre-registered verdict — decided BEFORE seeing the numbers
console.log("\n=== Verdict ===");
if (p < 0.05) {
  const winner = gap < 0 ? "TIED" : "UNTIED";
  console.log(`SIGNIFICANT (p < 0.05): ${winner} wins by ${Math.abs(gap).toFixed(4)} nats.`);
  console.log(gap < 0
    ? "Tying helped here — the shared matrix may act as a regularizer at this scale."
    : "Tying hurt here — the extra output parameters earned their keep at this scale.");
} else {
  console.log(`NOT significant (p = ${p.toFixed(3)} ≥ 0.05): the gap of ${Math.abs(gap).toFixed(4)} nats`);
  console.log(`is within seed noise. At this scale, weight tying appears to be free —`);
  console.log(`it saves ${D * V} parameters (${(100 * D * V / paramCount(initModel(false, mulberry32(0)))).toFixed(1)}% of the untied model) at no measured cost.`);
}
console.log("\nLimitations: one corpus, one architecture, ~5k params, loss-only metric.");
console.log("This answers the question at THIS scale — nothing more. That's research.");

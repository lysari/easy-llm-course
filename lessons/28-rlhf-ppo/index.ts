// Lesson 28 — RLHF with PPO: Making the Model Helpful
//
// A self-contained simulation of RLHF training dynamics.
// No imports. Runs with: npx ts-node index.ts
//
// What this demonstrates:
//   - generate_with_logprobs: getting token log-probs from the policy
//   - KL divergence penalty between RL policy and SFT reference
//   - PPO-style update (simplified: REINFORCE with clipped ratio)
//   - Total reward = reward_model_score - β * KL
//   - Training dynamics: reward hacking when β is too small

// ─── Pseudo-random number generator (seeded, no Math.random dependency) ──────

function makePRNG(seed: number) {
  let s = seed;
  return {
    next(): number {
      s = (s * 1664525 + 1013904223) & 0xffffffff;
      return (s >>> 0) / 0xffffffff;
    },
    randn(): number {
      // Box-Muller transform for Gaussian noise
      const u = this.next() + 1e-10;
      const v = this.next() + 1e-10;
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    },
    choice<T>(arr: T[]): T {
      return arr[Math.floor(this.next() * arr.length)];
    },
  };
}

const rng = makePRNG(42);

// ─── Math utilities ───────────────────────────────────────────────────────────

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

function logSoftmax(logits: number[]): number[] {
  const probs = softmax(logits);
  return probs.map((p) => Math.log(p + 1e-10));
}

function dot(a: number[], b: number[]): number {
  return a.reduce((sum, ai, i) => sum + ai * b[i], 0);
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function mean(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function clip(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

// ─── Vocabulary ───────────────────────────────────────────────────────────────
//
// We work with a tiny vocabulary of 8 "quality tokens":
//   0 = <PAD>
//   1 = <START>
//   2 = "helpful" (high quality)
//   3 = "detailed" (high quality)
//   4 = "accurate" (high quality)
//   5 = "vague" (low quality)
//   6 = "wrong" (low quality)
//   7 = "off-topic" (low quality)
//
// A "good" completion (rewarded highly) is one with more tokens 2-4.
// A "bad" completion has more tokens 5-7.
// This lets us simulate a reward model that actually distinguishes quality.

const VOCAB_SIZE = 8;
const TOKEN_START = 1;
const GOOD_TOKENS = [2, 3, 4]; // helpful, detailed, accurate
const BAD_TOKENS = [5, 6, 7]; // vague, wrong, off-topic
const COMPLETION_LEN = 5; // number of tokens per completion

// Prompts are just indices 0-3 (4 prompt types)
const PROMPTS = [0, 1, 2, 3];

// ─── Tiny LLM: A Linear Policy ───────────────────────────────────────────────
//
// To keep this runnable without a real transformer, we model the LLM as:
//   logits = W[context_embedding] + bias
//
// The "context" embedding is a simple average of the prompt token and
// the last generated token. The weights are a VOCAB_SIZE x VOCAB_SIZE matrix.
//
// This is obviously not a real transformer, but it captures the key structure:
//   - It has learnable parameters
//   - It produces a probability distribution over the next token
//   - We can compute log-probs for any token it might choose

interface TinyLLM {
  W: number[][]; // VOCAB_SIZE x VOCAB_SIZE weight matrix
  bias: number[]; // VOCAB_SIZE bias
}

function makeTinyLLM(initScale: number): TinyLLM {
  const W: number[][] = [];
  for (let i = 0; i < VOCAB_SIZE; i++) {
    W.push(Array.from({ length: VOCAB_SIZE }, () => rng.randn() * initScale));
  }
  const bias = Array.from({ length: VOCAB_SIZE }, () => rng.randn() * initScale);
  return { W, bias };
}

function cloneLLM(model: TinyLLM): TinyLLM {
  return {
    W: model.W.map((row) => [...row]),
    bias: [...model.bias],
  };
}

// Given a context token (integer 0..VOCAB_SIZE-1), compute logits over next token
function getLogits(model: TinyLLM, contextToken: number): number[] {
  // context_embedding = one-hot(contextToken)
  // logits = W * one_hot + bias = W[contextToken] + bias
  return model.W[contextToken].map((w, i) => w + model.bias[i]);
}

// ─── SFT Model ───────────────────────────────────────────────────────────────
//
// Simulates a supervised fine-tuned baseline: slightly biased toward good tokens
// but not strongly aligned.

function makeSFTModel(): TinyLLM {
  const model = makeTinyLLM(0.5);
  // Slightly bias toward good tokens in the weights
  for (let ctx = 0; ctx < VOCAB_SIZE; ctx++) {
    for (const goodTok of GOOD_TOKENS) {
      model.W[ctx][goodTok] += 0.3;
    }
    for (const badTok of BAD_TOKENS) {
      model.W[ctx][badTok] -= 0.1;
    }
  }
  return model;
}

// ─── Reward Model ─────────────────────────────────────────────────────────────
//
// Simulates lesson 27's reward model. Scores a completion between -1 and +1.
// Good tokens contribute positive reward, bad tokens contribute negative.
// We add a small random component to simulate imperfect reward modeling.

function scoreCompletion(tokens: number[]): number {
  let score = 0;
  for (const t of tokens) {
    if (GOOD_TOKENS.includes(t)) score += 0.4;
    if (BAD_TOKENS.includes(t)) score -= 0.4;
  }
  // Small noise (simulates reward model imperfection — enables reward hacking)
  score += rng.randn() * 0.1;
  // Clip to [-1, 1] and shift to [0, 1] for cleaner display
  return clip(score / (COMPLETION_LEN * 0.4), -1, 1);
}

// ─── generate_with_logprobs ───────────────────────────────────────────────────
//
// Generates a completion autoregressively and records the log probability
// of each chosen token. This is essential for computing the PPO ratio.

interface GenerationResult {
  tokens: number[];
  logprobs: number[]; // log π(token_t | context_t) for each step
}

function generateWithLogprobs(model: TinyLLM, promptToken: number): GenerationResult {
  const tokens: number[] = [];
  const logprobs: number[] = [];

  let contextToken = promptToken;

  for (let t = 0; t < COMPLETION_LEN; t++) {
    const logits = getLogits(model, contextToken);
    const logProbs = logSoftmax(logits);
    const probs = softmax(logits);

    // Sample from the distribution
    let chosen = 0;
    let cumulative = 0;
    const r = rng.next();
    for (let i = 0; i < VOCAB_SIZE; i++) {
      cumulative += probs[i];
      if (r <= cumulative) {
        chosen = i;
        break;
      }
    }
    // Ensure we don't stay on padding/start tokens for the whole completion
    if (chosen <= 1 && rng.next() > 0.3) {
      chosen = rng.next() > 0.5 ? GOOD_TOKENS[Math.floor(rng.next() * 3)] : BAD_TOKENS[Math.floor(rng.next() * 3)];
    }

    tokens.push(chosen);
    logprobs.push(logProbs[chosen]);
    contextToken = chosen;
  }

  return { tokens, logprobs };
}

// Get log probs of a GIVEN sequence of tokens (for the SFT reference model)
function getLogprobsForTokens(model: TinyLLM, promptToken: number, tokens: number[]): number[] {
  const logprobs: number[] = [];
  let contextToken = promptToken;
  for (const token of tokens) {
    const logits = getLogits(model, contextToken);
    const logProbs = logSoftmax(logits);
    logprobs.push(logProbs[token]);
    contextToken = token;
  }
  return logprobs;
}

// ─── computeKL ────────────────────────────────────────────────────────────────
//
// Simplified KL divergence between RL policy and SFT policy.
// KL(π_rl || π_sft) ≈ mean_t(log π_rl(token_t) - log π_sft(token_t))
//
// This is the per-sample KL: how much more (or less) likely the RL policy
// makes the same sequence compared to the SFT baseline.
// Positive → RL moved away from SFT (penalty increases)
// Negative → RL moved closer to SFT (penalty decreases, rare)

function computeKL(logprobsRL: number[], logprobsSFT: number[]): number {
  const diffs = logprobsRL.map((lp, i) => lp - logprobsSFT[i]);
  return mean(diffs);
}

// ─── Value Function ───────────────────────────────────────────────────────────
//
// Estimates the baseline (expected reward from a given prompt).
// We use a running exponential average per prompt — simple but effective
// for a simulation. In real RLHF this is a separate neural network.

class ValueFunction {
  private baselines: Map<number, number> = new Map();
  private alpha = 0.1; // EMA coefficient

  getBaseline(prompt: number): number {
    return this.baselines.get(prompt) ?? 0;
  }

  update(prompt: number, reward: number): void {
    const current = this.getBaseline(prompt);
    this.baselines.set(prompt, current + this.alpha * (reward - current));
  }
}

// ─── PPO Policy Update (simplified) ──────────────────────────────────────────
//
// In full PPO:
//   1. Collect a batch of (state, action, logprob_old, advantage) tuples
//   2. Run multiple epochs of gradient updates with clipped ratio
//
// Here we simplify to online updates (one sample at a time) but keep:
//   - The clipped probability ratio (the defining feature of PPO)
//   - The advantage (reward minus baseline)
//   - KL penalty in the reward signal

const PPO_EPSILON = 0.2; // clipping range: [1-ε, 1+ε]
const LEARNING_RATE = 0.01;

function ppoPolicyUpdate(
  rlModel: TinyLLM,
  sftModel: TinyLLM,
  promptToken: number,
  tokens: number[],
  logprobsRL: number[],
  totalReward: number,
  baseline: number
): { gradNorm: number } {
  const advantage = totalReward - baseline;

  // Recompute current log probs (may have changed since last update)
  const currentLogprobs = getLogprobsForTokens(rlModel, promptToken, tokens);
  const oldLogprobs = logprobsRL; // from when we generated

  let gradNormSq = 0;

  // Apply gradient to each token step
  let contextToken = promptToken;
  for (let t = 0; t < tokens.length; t++) {
    const token = tokens[t];

    // Probability ratio: how much has the policy changed for this token?
    const ratio = Math.exp(currentLogprobs[t] - oldLogprobs[t]);
    const clippedRatio = clip(ratio, 1 - PPO_EPSILON, 1 + PPO_EPSILON);

    // PPO objective: min(r·A, clip(r)·A)
    // We use the min to compute which gradient direction to take
    const useClipped = clippedRatio * advantage < ratio * advantage;
    const effectiveRatio = useClipped ? clippedRatio : ratio;

    // Gradient of log π(token | context) w.r.t. W[context][token]
    // For a linear softmax model: dL/dW[ctx][tok] = -advantage * effectiveRatio * (1 - π(tok))
    const logits = getLogits(rlModel, contextToken);
    const probs = softmax(logits);

    // Policy gradient update for each output logit
    for (let v = 0; v < VOCAB_SIZE; v++) {
      // d log π(token) / d logit_v = (1[v==token] - π(v))
      const dlogpi = (v === token ? 1 : 0) - probs[v];
      // PPO gradient: advantage * effectiveRatio * dlogpi
      const grad = advantage * effectiveRatio * dlogpi;
      // Update W[contextToken][v]
      const update = LEARNING_RATE * grad;
      rlModel.W[contextToken][v] += update;
      rlModel.bias[v] += LEARNING_RATE * 0.01 * grad; // smaller bias update
      gradNormSq += update * update;
    }

    contextToken = token;
  }

  return { gradNorm: Math.sqrt(gradNormSq) };
}

// ─── RLHF Training Loop ───────────────────────────────────────────────────────

interface EpisodeStats {
  episode: number;
  prompt: number;
  rewardModelScore: number;
  klDivergence: number;
  klPenalty: number;
  totalReward: number;
  advantage: number;
  gradNorm: number;
  goodTokenFraction: number;
}

function runRLHFTraining(beta: number, numEpisodes: number, label: string): void {
  console.log(`\n${"=".repeat(70)}`);
  console.log(`RLHF Training — β=${beta} (${label})`);
  console.log(`${"=".repeat(70)}`);
  console.log(`β controls KL penalty: small β → aggressive optimization → reward hacking`);
  console.log();

  // Initialize models
  const sftModel = makeSFTModel(); // frozen reference
  const rlModel = cloneLLM(sftModel); // starts from SFT, gets updated
  const valueFunction = new ValueFunction();

  const history: EpisodeStats[] = [];

  // Table header
  const header = [
    "Ep".padStart(4),
    "Prompt".padStart(6),
    "RM Score".padStart(9),
    "KL Div".padStart(8),
    "KL Pen".padStart(8),
    "Total R".padStart(8),
    "Advantage".padStart(10),
    "GoodTok%".padStart(9),
    "GradNorm".padStart(9),
  ].join(" | ");
  const divider = "-".repeat(header.length);

  for (let episode = 1; episode <= numEpisodes; episode++) {
    // 1. Sample a prompt
    const prompt = rng.choice(PROMPTS);
    const promptToken = prompt; // prompt index doubles as context token

    // 2. Generate completion with current RL policy
    const { tokens, logprobs: logprobsRL } = generateWithLogprobs(rlModel, promptToken);

    // 3. Get SFT log probs for the same sequence (for KL)
    const logprobsSFT = getLogprobsForTokens(sftModel, promptToken, tokens);

    // 4. Score with reward model
    const rewardModelScore = scoreCompletion(tokens);

    // 5. Compute KL divergence: KL(π_rl || π_sft)
    const klDivergence = computeKL(logprobsRL, logprobsSFT);

    // 6. KL penalty and total reward
    //    total_reward = reward_model_score - β * KL(π_rl || π_sft)
    const klPenalty = beta * klDivergence;
    const totalReward = rewardModelScore - klPenalty;

    // 7. Get baseline and compute advantage
    const baseline = valueFunction.getBaseline(prompt);
    const advantage = totalReward - baseline;

    // 8. PPO update
    const { gradNorm } = ppoPolicyUpdate(
      rlModel,
      sftModel,
      promptToken,
      tokens,
      logprobsRL,
      totalReward,
      baseline
    );

    // 9. Update value function (learns the baseline)
    valueFunction.update(prompt, totalReward);

    // Track good token fraction
    const goodTokenFraction = tokens.filter((t) => GOOD_TOKENS.includes(t)).length / tokens.length;

    const stats: EpisodeStats = {
      episode,
      prompt,
      rewardModelScore,
      klDivergence,
      klPenalty,
      totalReward,
      advantage,
      gradNorm,
      goodTokenFraction,
    };
    history.push(stats);

    // Print table every 10 episodes (and first 3)
    if (episode <= 3 || episode % 10 === 0) {
      if (episode === 1 || episode === 11) {
        console.log(header);
        console.log(divider);
      }
      console.log(
        [
          String(episode).padStart(4),
          String(prompt).padStart(6),
          rewardModelScore.toFixed(4).padStart(9),
          klDivergence.toFixed(4).padStart(8),
          klPenalty.toFixed(4).padStart(8),
          totalReward.toFixed(4).padStart(8),
          advantage.toFixed(4).padStart(10),
          (goodTokenFraction * 100).toFixed(1).padStart(8) + "%",
          gradNorm.toFixed(5).padStart(9),
        ].join(" | ")
      );
    }
  }

  // ─── Summary statistics ───────────────────────────────────────────────────

  const earlyEpisodes = history.slice(0, 10);
  const lateEpisodes = history.slice(-10);

  const earlyRM = mean(earlyEpisodes.map((s) => s.rewardModelScore));
  const lateRM = mean(lateEpisodes.map((s) => s.rewardModelScore));
  const earlyKL = mean(earlyEpisodes.map((s) => s.klDivergence));
  const lateKL = mean(lateEpisodes.map((s) => s.klDivergence));
  const earlyGood = mean(earlyEpisodes.map((s) => s.goodTokenFraction));
  const lateGood = mean(lateEpisodes.map((s) => s.goodTokenFraction));

  console.log();
  console.log("Summary (first 10 vs last 10 episodes):");
  console.log(`  Reward Model Score:  ${earlyRM.toFixed(3)} → ${lateRM.toFixed(3)}  (${lateRM > earlyRM ? "+" : ""}${((lateRM - earlyRM) * 100).toFixed(1)}%)`);
  console.log(`  KL Divergence:       ${earlyKL.toFixed(3)} → ${lateKL.toFixed(3)}  (${lateKL > earlyKL ? "grew" : "shrank"} by ${Math.abs(lateKL - earlyKL).toFixed(3)})`);
  console.log(`  Good Token Fraction: ${(earlyGood * 100).toFixed(1)}% → ${(lateGood * 100).toFixed(1)}%`);

  if (beta < 0.1) {
    console.log();
    console.log("[!] WARNING: Low β detected. Watch the KL divergence — reward hacking may occur.");
    console.log("    The reward model score may rise while the completions drift from SFT baseline.");
  }
  if (beta >= 0.3) {
    console.log();
    console.log("[i] High β: strong KL penalty keeps model close to SFT. Slower improvement expected.");
  }
}

// ─── Demonstrate: What Happens at Different β Values ─────────────────────────

function demonstrateKLPenaltyEffect(): void {
  console.log("\n" + "=".repeat(70));
  console.log("DEMONSTRATION: Effect of β (KL penalty coefficient)");
  console.log("=".repeat(70));
  console.log();
  console.log("β is the key hyperparameter in RLHF:");
  console.log("  β = 0 (no penalty) → pure reward optimization → reward hacking");
  console.log("  β = 0.1 (typical)  → good balance: improve reward, stay grounded");
  console.log("  β = 0.5 (high)     → conservative: safe but slow improvement");
  console.log();

  // Show a single episode with different β values to illustrate the math
  const sftModel = makeSFTModel();
  const prompt = 0;

  console.log("Single episode example (same completion, different β values):");
  console.log();

  // Generate one completion
  const { tokens, logprobs: logprobsRL } = generateWithLogprobs(sftModel, prompt);
  const logprobsSFT = getLogprobsForTokens(sftModel, prompt, tokens);
  const rmScore = scoreCompletion(tokens);
  const kl = computeKL(logprobsRL, logprobsSFT);

  const tokenNames: { [k: number]: string } = {
    0: "<PAD>", 1: "<START>", 2: "helpful", 3: "detailed",
    4: "accurate", 5: "vague", 6: "wrong", 7: "off-topic",
  };
  console.log(`  Completion tokens: [${tokens.map((t) => tokenNames[t] ?? t).join(", ")}]`);
  console.log(`  Reward Model Score: ${rmScore.toFixed(4)}`);
  console.log(`  KL Divergence: ${kl.toFixed(4)}`);
  console.log();
  console.log("  β       | KL Penalty | Total Reward | Interpretation");
  console.log("  " + "-".repeat(65));
  for (const beta of [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]) {
    const pen = beta * kl;
    const total = rmScore - pen;
    let interp = "";
    if (beta === 0.0) interp = "pure RM (reward hacking risk)";
    else if (beta <= 0.1) interp = "typical InstructGPT range";
    else if (beta <= 0.2) interp = "moderate constraint";
    else interp = "conservative, safe";
    console.log(`  β=${beta.toFixed(2)}  | ${pen.toFixed(4).padStart(10)} | ${total.toFixed(4).padStart(12)} | ${interp}`);
  }
}

// ─── DPO Loss Illustration ────────────────────────────────────────────────────
//
// Show the DPO loss calculation for a preference pair.
// This illustrates how DPO replaces the RL loop with a supervised objective.

function demonstrateDPO(): void {
  console.log("\n" + "=".repeat(70));
  console.log("BONUS: DPO (Direct Preference Optimization) vs PPO");
  console.log("=".repeat(70));
  console.log();
  console.log("DPO loss: -log σ(β · (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))");
  console.log();

  const sftModel = makeSFTModel();
  const rlModel = cloneLLM(sftModel);
  const beta = 0.1;
  const prompt = 0;

  // Preferred completion: mostly good tokens
  const preferredTokens = [2, 3, 4, 2, 3]; // helpful, detailed, accurate, helpful, detailed
  // Rejected completion: mostly bad tokens
  const rejectedTokens = [5, 6, 7, 5, 6]; // vague, wrong, off-topic, vague, wrong

  const logprobWinRL = mean(getLogprobsForTokens(rlModel, prompt, preferredTokens));
  const logprobLoseRL = mean(getLogprobsForTokens(rlModel, prompt, rejectedTokens));
  const logprobWinRef = mean(getLogprobsForTokens(sftModel, prompt, preferredTokens));
  const logprobLoseRef = mean(getLogprobsForTokens(sftModel, prompt, rejectedTokens));

  const dpoMargin = beta * (logprobWinRL - logprobLoseRL - logprobWinRef + logprobLoseRef);
  const dpoLoss = -Math.log(sigmoid(dpoMargin) + 1e-10);

  console.log(`  Preferred completion: [helpful, detailed, accurate, helpful, detailed]`);
  console.log(`  Rejected completion:  [vague, wrong, off-topic, vague, wrong]`);
  console.log();
  console.log(`  log π_rl(y_win):  ${logprobWinRL.toFixed(4)}`);
  console.log(`  log π_rl(y_lose): ${logprobLoseRL.toFixed(4)}`);
  console.log(`  log π_ref(y_win): ${logprobWinRef.toFixed(4)}`);
  console.log(`  log π_ref(y_lose):${logprobLoseRef.toFixed(4)}`);
  console.log();
  console.log(`  DPO margin (β=0.1): ${dpoMargin.toFixed(4)}`);
  console.log(`  σ(margin):          ${sigmoid(dpoMargin).toFixed(4)}`);
  console.log(`  DPO loss:           ${dpoLoss.toFixed(4)}`);
  console.log();
  console.log("  DPO interpretation:");
  console.log("    - margin > 0 → RL policy already prefers the good completion (good!)");
  console.log("    - margin < 0 → RL policy still prefers the bad completion (loss drives update)");
  console.log("    - loss → 0 as the model learns to strongly prefer y_win over y_lose");
  console.log();
  console.log("  vs PPO: DPO needs no reward model, no value function, no RL loop.");
  console.log("  It's just a classification loss on the LLM directly from preference pairs.");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

console.log("╔══════════════════════════════════════════════════════════════════════╗");
console.log("║        Lesson 28: RLHF with PPO — Simplified Training Loop          ║");
console.log("╠══════════════════════════════════════════════════════════════════════╣");
console.log("║  Models: TinyLLM (linear softmax policy) + simulated reward model   ║");
console.log("║  Vocab: 8 tokens (2-4=good quality, 5-7=bad quality)                ║");
console.log("║  Key formula: total_reward = RM_score - β · KL(π_rl || π_sft)      ║");
console.log("╚══════════════════════════════════════════════════════════════════════╝");

// Run 1: Small β — aggressive optimization, reward hacking risk
runRLHFTraining(0.05, 50, "LOW β — aggressive, reward hacking risk");

// Run 2: Typical β — balanced trade-off
runRLHFTraining(0.15, 50, "TYPICAL β — balanced alignment");

// Run 3: Large β — conservative, stays close to SFT
runRLHFTraining(0.4, 50, "HIGH β — conservative, slow improvement");

// Demonstrate the math behind the KL penalty
demonstrateKLPenaltyEffect();

// Bonus: DPO illustration
demonstrateDPO();

console.log("\n" + "=".repeat(70));
console.log("KEY TAKEAWAYS");
console.log("=".repeat(70));
console.log();
console.log("1. PPO's clipped ratio prevents catastrophic updates by limiting how");
console.log("   much the policy can change in a single step (ε=0.2 → max 20% change).");
console.log();
console.log("2. KL divergence penalty is essential: without it (β≈0), the RL policy");
console.log("   will exploit the reward model's flaws (reward hacking).");
console.log();
console.log("3. total_reward = RM_score - β·KL is the core equation of RLHF.");
console.log("   The β hyperparameter controls the alignment/capability trade-off.");
console.log();
console.log("4. DPO bypasses RL entirely: it directly optimizes preference pairs");
console.log("   into the LLM as a supervised loss. Simpler, often comparable quality.");
console.log();
console.log("5. InstructGPT showed 1.3B RLHF model > 175B base GPT-3 by human raters.");
console.log("   Alignment quality >> raw scale for instruction-following tasks.");

# Lesson 21 — KV Cache: Making Inference Fast

## The Problem: Recomputing Everything on Every Step

You have a trained GPT. Now you want to generate text. Generation is
autoregressive: you feed the model one token at a time and ask it to predict the
next one.

Here is what naive inference looks like under the hood:

```
Step 1: input = [t1]
  compute Q, K, V for [t1]
  run attention over sequence of length 1
  predict next token → t2

Step 2: input = [t1, t2]
  compute Q, K, V for [t1, t2]   ← K1, V1 were already computed in step 1!
  run attention over sequence of length 2
  predict next token → t3

Step 3: input = [t1, t2, t3]
  compute Q, K, V for [t1, t2, t3]   ← K1,V1,K2,V2 already computed!
  run attention over sequence of length 3
  predict next token → t4

Step N: input = [t1 ... tN]
  compute Q, K, V for ALL N tokens   ← almost all of this is wasted work
  run attention over sequence of length N
  predict next token → t(N+1)
```

Total computation for generating a sequence of length N:

- Step 1 does work proportional to 1
- Step 2 does work proportional to 2
- Step 3 does work proportional to 3
- Step N does work proportional to N
- **Total: 1+2+3+...+N = N(N+1)/2 = O(N²)**

For a 1000-token response, that is roughly 500,000 units of attention work instead
of 1000. For a 10,000-token response: 50,000,000 vs 10,000. The waste grows
quadratically. This is the problem KV cache solves.

---

## Why K and V — But Not Q?

Recall what attention computes:

```
Attention(Q, K, V) = softmax(Q * Kᵀ / sqrt(d_k)) * V
```

- **Q (Query):** "What am I looking for?" — depends on the current token only.
- **K (Key):** "What do I contain?" — depends on each past token only.
- **V (Value):** "What do I contribute?" — depends on each past token only.

When you are at step N predicting token N+1:

- The new token's Q is fresh — you must always compute it.
- The new token's K and V are also fresh — compute once, then cache.
- Every past token's K and V are identical to what they were in previous steps
  — they depend only on those tokens' embeddings, which never change.

K and V are the things that get recomputed wastefully. Q is always new.

---

## The Solution: Cache K and V

Instead of recomputing K and V for every token at every step, compute them once
and store them.

```
Prefill phase — process the prompt all at once:
  input = [t1, t2, t3]   (your seed/prompt)
  compute K and V for all tokens
  store in KV cache:
    cache.K = [K1, K2, K3]   shape: (3, d_k)
    cache.V = [V1, V2, V3]   shape: (3, d_v)

Decode phase — generate one token at a time:
  Step 1:
    input = t4 only (single new token)
    compute new_Q, new_K, new_V from t4
    append to cache:
      cache.K = [K1, K2, K3, K4]   ← only K4 is new
      cache.V = [V1, V2, V3, V4]
    attention: new_Q (shape 1×d) attends over cache.K, cache.V (shape 4×d)
    output: attended vector of shape (1×d)
    no causal mask needed — the cache IS the past; no future tokens exist

  Step 2:
    input = t5 only
    compute new_Q, new_K, new_V from t5
    append → cache is now length 5
    attend: new_Q over all 5 cached K, V
    output: (1×d)

  Step N:
    O(1) new computation per step (just the new token)
    O(N) attention (attend over N cached keys) — unavoidable, but done once
    Total over all steps: O(N) instead of O(N²)
```

Each step:

1. Embed the single new token.
2. Compute Q, K, V via linear projection — each is a single row vector.
3. Append K and V to the cache (cache grows by 1 row).
4. Compute attention: Q (1×d) dot all cached K (T×d) → scores (1×T) → softmax → weighted sum over V → output (1×d).
5. Pass output through the rest of the transformer block (FFN, layer norm).
6. Project to vocabulary logits and sample.

---

## Memory Cost of the KV Cache

Speed is not free. Caching K and V requires storing them.

Memory per layer = 2 × (seq_len × head_dim × num_heads) × bytes_per_param

Full formula across all layers:

```
KV cache memory = 2 × num_layers × num_heads × seq_len × head_dim × bytes_per_param
```

Example: GPT-4 scale model with 128k context window

```
2 × 96 layers × 96 heads × 128,000 tokens × 128 head_dim × 2 bytes (fp16)
= 2 × 96 × 96 × 128,000 × 128 × 2
= 2 × 96 × 96 × 32,768,000
= 2 × 301,989,888,000
≈ 600 GB per batch item
```

That is 600 GB of memory for a single user's conversation at 128k context.
Serving 10 concurrent users: 6 TB of memory just for KV caches.

This is why:
- Long context inference is expensive (linearly more memory per token of context).
- GPU memory, not compute, is often the bottleneck for LLM inference.
- Cloud providers charge more for long-context API calls.
- Systems like vLLM were built specifically to manage KV cache memory.

---

## Prefill vs Decode: Two Distinct Phases

Modern inference systems treat these as separate workloads because they have
completely different performance characteristics.

**Prefill phase** (processing the prompt):
- All prompt tokens are known upfront.
- Run a full forward pass over the entire prompt in parallel.
- Highly parallelizable — GPU utilization is high.
- Compute-bound: the bottleneck is matrix multiplications.
- Fast per token when prompt is long (batch parallelism).

**Decode phase** (generating new tokens):
- One token at a time, each step depends on the previous.
- Cannot be parallelized across steps.
- Memory-bound: the bottleneck is reading the KV cache from GPU memory.
- Slow per token in absolute terms but much faster than naive recomputation.
- Throughput is measured in tokens/second per user.

Production systems (like vLLM, TensorRT-LLM) often run prefill and decode on
different hardware or schedule them separately to maximize GPU utilization.

---

## PagedAttention: Managing the KV Cache Like Virtual Memory

The KV cache has a fragmentation problem. Different requests have different
lengths. If you pre-allocate the maximum context length for every request:

- Short requests waste most of their allocation.
- You cannot predict which requests will grow long.
- Fragmentation means you can serve fewer requests than memory would allow.

**PagedAttention** (introduced in the vLLM paper, 2023) solves this by borrowing
ideas from operating system virtual memory:

```
Physical KV cache memory is divided into fixed-size blocks ("pages").
Each page holds K and V for a fixed number of tokens (e.g., 16 tokens).

A page table maps (request_id, logical_block_number) → physical_block_address.

When a request needs more tokens:
  Allocate a new physical page (anywhere in memory).
  Add the mapping to that request's page table.
  The attention kernel follows the page table to read K and V.

When a request finishes:
  Mark its pages as free — immediately available for other requests.
  No copying, no defragmentation needed.
```

Benefits:
- Near-zero memory waste (only the last page of each sequence is partially filled).
- More requests fit in memory simultaneously → higher throughput.
- Enables copy-on-write sharing for parallel sampling (beam search, multiple
  completions from the same prompt share the same physical pages until they diverge).

This is the core insight behind vLLM's 2-3x throughput improvement over naive
serving.

---

## Summary

| Approach | Computation per step | Total for N tokens | Memory |
|---|---|---|---|
| Naive (no cache) | O(N) attention | O(N²) total | O(1) extra |
| KV Cache | O(1) new work + O(N) attend | O(N) total | O(N × layers) |

Key facts to remember:

1. Cache K and V — not Q. Q is always fresh for the new token.
2. Prefill is parallel and compute-bound. Decode is sequential and memory-bound.
3. KV cache memory grows linearly with context length, number of layers, and
   head dimension.
4. For large models with long contexts, KV cache memory dominates — often larger
   than the model weights themselves.
5. PagedAttention manages KV cache fragmentation using virtual-memory-style paging.

In the next lesson you will implement quantization — reducing KV cache memory
(and model size) by using fewer bits per value.

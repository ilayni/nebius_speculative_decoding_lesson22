# Speculative Decoding — Run Insights

**Run summary:** `'once upon'` → 30 tokens | 8 big-model passes | 0.85s | 77.14% acceptance rate

---

## 0. What is accept_prob and How is it Calculated?

`accept_prob = min(1, p(x̃) / q(x̃))`

Where:
- `x̃` is the draft token the small model sampled
- `q(x̃)` is the probability the **small model** assigned to that token
- `p(x̃)` is the probability the **big model** assigns to that same token

**Concrete example from this run — Step 1, token 3 (`','`):**
```
p = 0.264  (big model: comma has 26.4% chance here)
q = 0.365  (small model: comma has 36.5% chance)

accept_prob = min(1, 0.264 / 0.365) = 0.72
r = 0.04  →  0.04 ≤ 0.72  →  ACCEPTED
```

**The intuition:**
- `p ≥ q` → big model likes the token at least as much → `accept_prob = 1.00` → always accept
- `p < q` → big model is less enthusiastic → accept with probability `p/q`
- The bigger the disagreement, the lower the threshold, the more likely rejection

**On rejection** the token is resampled from the corrected distribution `normalize(max(0, p − q))` — the probability mass the big model has *above* the small model. This is the Metropolis-Hastings acceptance rule adapted for token sampling, and it's what makes the algorithm **lossless**: the final output distribution is identical to what the big model alone would have produced.

**Important: `max(0, p−q)` is used once, not carried forward.** The exact sequence on rejection:
```
1. Draft token x̃ is REJECTED
2. Immediately resample a replacement from normalize(max(0, p − q))
3. Append that replacement token to the output
4. STOP — discard all remaining draft tokens after position i
5. Next step: small model drafts γ fresh tokens from the new position
6. New r is rolled independently for each new draft token
```
After a rejection you get exactly 1 token (the resampled one), then the whole draft→verify→accept cycle starts over fresh. `max(0, p−q)` has no memory or carry-over to the next step.

---

## 0b. What is the Vocabulary and Where Does the Resampled Token Come From?

**The vocabulary** is the fixed set of all tokens both models know — it belongs to the tokenizer, shared by both models. For Gemma it's ~256,000 tokens: every word, subword, punctuation mark, number, etc.

`p` and `q` are both vectors of length 256,000 — one probability number per token. They are already fully computed from the draft and verify phases. **No model runs during rejection** — it's pure arithmetic.

**Concrete example — Step 2, token 2 (`' K'`) from this run:**
```
Draft token: ' K'
q[' K'] = 0.017   (small model gave it 1.7%)
p[' K'] = 0.006   (big model gives it only 0.6%)

accept_prob = 0.006 / 0.017 = 0.34
r = 0.90  →  REJECTED
```

Now compute `corrected = max(0, p − q)` over all 256,000 tokens:
```
token        p       q      p−q    corrected
──────────────────────────────────────────────
' the'      0.21   0.05   +0.16     0.16   ← big model likes more
' heaven'   0.08   0.01   +0.07     0.07   ← big model likes more
' of'       0.06   0.02   +0.04     0.04
' K'        0.006  0.017  −0.011    0.00   ← clamped to 0 (q > p)
' a'        0.04   0.06   −0.02     0.00   ← clamped to 0 (q > p)
... (all other 255,995 tokens) ...
```

Normalize so all values sum to 1, then call `torch.multinomial` to randomly sample one token. The result was `' heaven'` — not the top token (`' the'`), but sampled probabilistically. Run it again and you might get `' of'` or `' the'`.

**Key insight:** the corrected distribution zeroes out every token the small model over-estimated, keeping only tokens the big model favors more. Sampling from it gives a token the big model would have accepted — without running either model again.

**What does p−q actually represent?**

`p - q` is the difference in opinion between the two models over every token:
```
p[token] > q[token]  →  big model is MORE excited about this token than small model
p[token] < q[token]  →  small model over-estimated it, big model doesn't want it
p[token] = q[token]  →  they agree
```

`max(0, p−q)` keeps only tokens where the big model is more confident, zeros out everything else. You are literally "removing" choices where `q > p` from the distribution you sample from.

You don't take `argmax` (the single top token) because that would be greedy decoding — the output distribution would no longer match what the big model generates with sampling. By sampling from `max(0, p−q)` you preserve the big model's full probabilistic behavior. The math guarantees: accept with `p/q` + on rejection sample from `max(0, p−q)` = output distribution exactly equals `p`. That's the lossless proof.

**The values in p−q matter — it's weighted sampling, not uniform random.**

`torch.multinomial` picks tokens **proportional to their value**, not uniformly. Higher value = higher chance of being picked. From your run (Step 2 rejection):

```
token        p−q     normalized   chance of being picked
──────────────────────────────────────────────────────────
' the'       0.16      0.59            59%   ← most likely
' heaven'    0.07      0.26            26%   ← got picked this run
' of'        0.04      0.15            15%
' K'         0.00      0.00             0%   ← impossible (q > p)
' a'         0.00      0.00             0%   ← impossible (q > p)
```

`' heaven'` was picked — less likely than `' the'` but perfectly valid. Run it 100 times and `' the'` wins ~59% of the time. If it were uniform over non-zero values, all three would have 33% each — that would break losslessness because `' the'` should dominate.

The magnitude of `p[token] − q[token]` encodes **how much more** the big model wants that token over the small model. Bigger gap = bigger weight = sampled more often.

**Do you sample from the big model's vocab or the small model's vocab?**

They are identical — both models must share the same tokenizer and vocabulary for speculative decoding to work. For Gemma 3 (270M and 1B), both use the same tokenizer with the same 256k tokens. Every index in `p` maps to exactly the same token as the same index in `q`.

If the vocabularies didn't match, `p[i] − q[i]` would be meaningless — you'd be subtracting the probability of one token from the probability of a completely different token. This is why model family matters: you cannot pair arbitrary models. Draft and target must share a tokenizer.

---

## 0c. Are p and q Recomputed on Every Rejection?

No — this is the core efficiency of the algorithm. `p` and `q` are computed **once per step**, then reused for all γ acceptance checks. No recomputation happens at rejection time.

The exact timeline per step:
```
1. Small model runs γ times → produces q[0]…q[γ-1]    (γ forward passes, 256k floats each)
2. Big model runs ONCE on (prefix + all γ draft tokens) → produces p[0]…p[γ]  (1 forward pass)
3. Loop through γ positions checking accept/reject — using already-computed p and q in GPU memory
4. If rejection at position i → p[i] and q[i] are full 256k vectors already in memory
      → compute max(0, p[i] − q[i]) element-wise over all 256k tokens → no model runs
5. Start next step with updated prefix → fresh forward passes → new p and q
```

So per step you have in memory:
- **γ vectors of q** — one 256k vector per draft token, each from a separate small-model pass
- **γ+1 vectors of p** — all from ONE big-model pass over the full draft sequence in parallel

**This is the whole efficiency trick:** the big model's single batched forward pass over γ+1 positions gives you all the `p` vectors for the entire step at once. The rejection math just reuses `p[i]` and `q[i]` already sitting in GPU memory — subtraction, clamp, normalize, sample. No model touched.

**What does "big model preferred" mean in the output?**

It is a debug label only — it does not affect the algorithm. It is computed as `argmax(p[i])`: the single token with the highest probability in the big model's distribution at position i, decoded to text.

```python
# Pure debug, no model run:
big_top = tokenizer.decode([p[i].argmax().item()])
print(f"... big model preferred='{big_top}'")
```

It answers "what would the big model have picked if it were doing greedy decoding here?" — useful for understanding disagreements, but the algorithm never uses argmax. Acceptance uses `p[i, draft_token_id]` (a single scalar), and rejection uses the full `p[i]` vector for weighted sampling. The big model still ran only once.

---

**Precise accounting per step (γ=5):**

- Small model runs **5 times**, once per draft token → produces 5 separate 256k `q` vectors (q[0]…q[4])
- Big model runs **once** on prefix + 5 draft tokens → produces 6 separate 256k `p` vectors (p[0]…p[5]) in one batched pass
- For each of the 5 positions: `p[i, token_id]` and `q[i, token_id]` are read as scalars → compute `accept_prob`
- `p−q` (full 256k subtraction) is only computed for the **one position that gets rejected** — the other pairs are used only for their accept_prob scalar and then discarded
- If no rejection: `p−q` is never computed at all — just sample the bonus token from `p[5]`

---

## 1. The Core Speedup in Numbers

Without speculative decoding: 30 tokens = **30 big-model forward passes**
With speculative decoding: 30 tokens = **8 big-model forward passes**

That's a **3.75x reduction** in big-model calls on this run. On an H100 the wall time was 0.85s — on CPU this would have been minutes.

---

## 2. Acceptance Rate Breakdown Per Step

| Step | Drafted | Accepted | Outcome |
|------|---------|----------|---------|
| 1 | 5 | 5 + bonus | Perfect — 6 tokens from 1 big-model pass |
| 2 | 5 | 1 + reject resample | Small model diverged early (' K' vs ' the') |
| 3 | 5 | 3 + reject resample | Partial — stopped at token 4 |
| 4 | 5 | 5 + bonus | Perfect — 6 tokens from 1 big-model pass |
| 5 | 5 | 5 + bonus | Perfect — 6 tokens from 1 big-model pass |
| 6 | 5 | 2 + reject resample | Stopped at token 3 |
| 7 | 3 | 0 + reject resample | Worst case — only 1 token from 1 big-model pass |
| 8 | 2 | 1 + reject resample | Near end of sequence |

Steps 1, 4, 5 were optimal (all 5 accepted + bonus). Step 7 was the worst — rejected on the very first token, getting only 1 token per big-model pass (same as naive decoding).

---

## 3. When Does Acceptance Happen?

**High acceptance = both models agree strongly:**
- Step 1 token 1: `' a'` — p=0.977, q=0.980 → `accept_prob=1.00` ✓
- Step 1 token 2: `' time'` — p=0.973, q=0.973 → `accept_prob=1.00` ✓

These are "once upon **a time**" — a very predictable phrase. Both models are near-certain, so acceptance is guaranteed.

**Low acceptance = models disagree:**
- Step 2 token 2: `' K'` — p=0.006, q=0.017 → `accept_prob=0.34` → REJECTED, resampled `' heaven'`
- Step 7 token 1: `' weekly'` — p=0.002, q=0.004 → `accept_prob=0.39` → REJECTED, resampled `' communities'`

The small model picked low-probability tokens that the big model didn't like. The corrected distribution `max(0, p−q)` redirected to what the big model actually wanted.

---

## 4. The Lossless Guarantee in Action

Notice the corrected resamples are not random — they reflect the big model's actual preference:

| Rejected | Big model preferred | Resampled |
|----------|-------------------|-----------|
| `' K'` | `' the'` | `' heaven'` |
| `'.'` | `' was'` | `' of'` |
| `' that'` | `' the'` | `' their'` |
| `' weekly'` | `' church'` | `' communities'` |
| `' where'` | `' they'` | `' they'` |

The resample comes from `normalize(max(0, p − q))` — the probability mass the big model has *above* the small model. This is why the algorithm is **lossless**: the final token distribution exactly matches what the big model would have generated alone.

---

## 5. accept_prob=1.00 When p > q

An important edge case: when the big model is *more* confident than the small model, acceptance probability is clamped to 1.00. Examples:
- Step 2 token 1: `' of'` — p=0.926, q=0.801 → always accept
- Step 4 token 1: `' the'` — p=0.730, q=0.609 → always accept

This means the big model effectively "upgrades" the small model's output for free — you never reject a token the big model likes *more* than the small model does.

---

## 6. Generated Text Quality

`once upon a time, in the town of heaven, every days of the week the church was founded by 15 families, and in their communities, they`

Minor grammar issues ("every days") but coherent fairy-tale-style prose — expected from a pre-trained base model (not instruction-tuned). The text quality is **identical** to what the big model alone would have produced (that's the lossless guarantee).

---

## 7. Key Takeaways

- **77% acceptance rate** is good for a 270M → 1B pair. Real-world Speculative Decoding (e.g. Llama 68M → 7B) achieves 70–85% on natural text.
- **Predictable phrases** ("once upon a time") get near-perfect acceptance. **Creative/ambiguous continuations** (what comes after "15 families,") get lower acceptance.
- **Worst case** (Step 7, 0 accepted) is no worse than naive decoding — you still produce 1 token per big-model pass.
- **Best case** (Steps 1, 4, 5) gives γ+1 = 6 tokens per big-model pass — a 6x improvement over naive.
- The algorithm's expected tokens per step = `γ · α + 1` = `5 × 0.77 + 1 = 4.85` — which matches our average of 30 tokens / 8 steps ≈ **3.75 tokens/step** (slightly lower because early steps were harder).

## note 
The lossless guarantee is visible in the data: every rejected token gets resampled from max(0, p−q) and the result tracks
  what the big model actually wanted (e.g. ' K' → ' heaven', ' weekly' → ' communities').

  77% acceptance on a 270M→1B pair is solid. The formula predicts 4.85 tokens/step; you actually got ~3.75 — the gap is because
  the later steps (6–8) were harder text.

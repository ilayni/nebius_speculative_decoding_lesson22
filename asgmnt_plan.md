# Workshop Assignment Plan: Vanilla Speculative Decoding
**Paper:** [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192) (Leviathan et al., 2022)

---

## Environment Setup

**Preferred: CPU** (feasible for 270M + 1B models at seqlen=30, ~1–3 min total)
- Change `device_type = 'cuda'` → `'cpu'`
- Change `torch_dtype = torch.bfloat16` → `torch.float32` (bfloat16 CPU support is limited)
- If too slow on CPU: fall back to Google Colab (T4 GPU) or Nebius GPU

**HuggingFace login** (replace Colab-specific cell):
```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")   # needs read access to google/gemma-3-* (gated)
```
Alternatively use open models (GPT-2 / DistilGPT-2) to skip the Gemma gate:
```python
big_model_name   = "gpt2-medium"  # ~345M
small_model_name = "distilgpt2"   # ~82M
```

---

## Algorithm Reference (from the paper)

```
Given: draft model Mq (small), target model Mp (big), prefix tokens, γ, seqlen

while output < seqlen:
    # --- DRAFT PHASE (γ forward passes of Mq) ---
    autoregressively sample γ tokens x̃₁…x̃ᵧ from Mq
    save q₁…qᵧ  ← the full vocab distributions at each draft step

    # --- VERIFY PHASE (1 forward pass of Mp) ---
    run Mp on (prefix + x̃₁…x̃ᵧ) → p₁…pᵧ₊₁

    # --- ACCEPTANCE / REJECTION ---
    for i = 1 … γ:
        r ~ Uniform(0,1)
        if r ≤ min(1, pᵢ(x̃ᵢ) / qᵢ(x̃ᵢ)):   ← accept
            keep x̃ᵢ
        else:                                   ← reject
            sample xᵢ from  normalize(max(0, p(x) − q(x)))
            stop inner loop

    if all γ accepted:
        sample one bonus token from pᵧ₊₁

    append accepted tokens → advance prefix
```

Key insight: the big model runs **once** on a batch of γ+1 positions — this is the parallelism gain.

---

## Implementation Steps

### Step 1 — Draft + collect q(x) and p(x)

```python
def generate_draft_and_verify(
    big_model, small_model, tokenizer, input_ids, gamma, device
):
    """
    Returns:
        q: Tensor [gamma, vocab]   — draft model distributions
        p: Tensor [gamma+1, vocab] — target model distributions
        draft_tokens: Tensor [gamma] — sampled draft token ids
    """
    draft_tokens = []
    q_probs = []

    small_model.eval()
    big_model.eval()

    current_ids = input_ids.clone()

    with torch.no_grad():
        # --- Draft phase: γ autoregressive steps ---
        for _ in range(gamma):
            out = small_model(current_ids)
            logits = out.logits[:, -1, :]          # [1, vocab]
            probs  = torch.softmax(logits, dim=-1) # [1, vocab]

            token = torch.multinomial(probs, num_samples=1)  # [1, 1]
            q_probs.append(probs.squeeze(0))       # [vocab]
            draft_tokens.append(token.item())
            current_ids = torch.cat([current_ids, token], dim=-1)

        q = torch.stack(q_probs, dim=0)  # [gamma, vocab]
        draft_token_tensor = torch.tensor(draft_tokens, device=device)

        # --- Verify phase: 1 forward pass of big model ---
        # current_ids already contains prefix + gamma draft tokens
        out_big = big_model(current_ids)
        # We need positions [-gamma-1 : ] — i.e. p for each draft position + 1 bonus
        p_logits = out_big.logits[:, -(gamma + 1):, :]  # [1, gamma+1, vocab]
        p = torch.softmax(p_logits.squeeze(0), dim=-1)  # [gamma+1, vocab]

    return q, p, draft_token_tensor
```

### Step 2 — Speculative Sampling (acceptance rule)

```python
def speculative_sampling(q, p, draft_tokens, gamma, device):
    """
    Returns:
        accepted_tokens: list of accepted token ids
        bonus_token:     int | None  (bonus token if all gamma accepted)
    """
    accepted = []

    for i in range(gamma):
        token_id = draft_tokens[i].item()
        q_i = q[i, token_id]
        p_i = p[i, token_id]

        acceptance_prob = torch.clamp(p_i / (q_i + 1e-9), max=1.0)
        r = torch.rand(1, device=device).item()

        if r <= acceptance_prob:
            accepted.append(token_id)
        else:
            # Resample from corrected distribution
            corrected = torch.clamp(p[i] - q[i], min=0.0)
            if corrected.sum() < 1e-9:
                corrected = p[i]  # fallback to target
            corrected /= corrected.sum()
            resampled = torch.multinomial(corrected, num_samples=1).item()
            accepted.append(resampled)
            return accepted, None   # stop — rejection happened

    # All gamma accepted → bonus token from p[gamma]
    bonus = torch.multinomial(p[gamma], num_samples=1).item()
    return accepted, bonus
```

### Step 3 — Outer Loop with Colorama Coloring

```python
from colorama import Fore, Style, init as colorama_init
colorama_init()

def speculative_decode(big_model, small_model, tokenizer, input_text,
                       gamma=5, seqlen=30, device='cpu'):
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)
    generated_ids = input_ids.clone()
    output_text   = input_text

    total_accepted = 0
    total_draft    = 0
    import time
    t0 = time.time()

    while generated_ids.shape[-1] - input_ids.shape[-1] < seqlen:
        remaining = seqlen - (generated_ids.shape[-1] - input_ids.shape[-1])
        g = min(gamma, remaining)

        q, p, draft_tokens = generate_draft_and_verify(
            big_model, small_model, tokenizer, generated_ids, g, device
        )
        accepted, bonus = speculative_sampling(q, p, draft_tokens, g, device)

        # Append accepted tokens with GREEN coloring
        for tok in accepted:
            total_accepted += 1
            output_text += Fore.GREEN + tokenizer.decode([tok]) + Style.RESET_ALL
        total_draft += g

        # Bonus token (from big model) in YELLOW — or rejection correction in RED
        if bonus is not None:
            output_text += Fore.YELLOW + tokenizer.decode([bonus]) + Style.RESET_ALL
            accepted.append(bonus)
        # else: last element in accepted is the rejection resample (already appended)

        new_ids = torch.tensor([accepted], device=device)
        generated_ids = torch.cat([generated_ids, new_ids], dim=-1)

    walltime = time.time() - t0
    acceptance_rate = total_accepted / max(total_draft, 1)
    print(f"\n\n[Walltime: {walltime:.2f}s | Acceptance rate: {acceptance_rate:.2%}]")
    print(output_text)
    return generated_ids, acceptance_rate, walltime
```

### Step 4 — Benchmark on Fineweb Sample

```python
from datasets import load_dataset

dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                       split="train", streaming=True)
samples = list(itertools.islice(dataset, 5))  # 5 samples is enough for timing

results = []
for sample in samples:
    prefix = sample['text'][:50]  # first 50 chars as prefix
    _, ar, wt = speculative_decode(
        big_model, small_model, tokenizer,
        prefix, gamma=gamma, seqlen=seqlen, device=device_type
    )
    results.append({'prefix': prefix, 'acceptance_rate': ar, 'walltime': wt})

# Report
import pandas as pd
df = pd.DataFrame(results)
print(df)
print(f"\nMean acceptance rate: {df['acceptance_rate'].mean():.2%}")
print(f"Mean walltime:        {df['walltime'].mean():.2f}s")
```

---

## CPU vs GPU Decision Tree

| Condition | Action |
|---|---|
| CPU, seqlen ≤ 50, 1B + 270M | Should finish in 2–5 min — try it |
| CPU too slow (> 10 min) | Switch to `gpt2-medium` + `distilgpt2` |
| Still too slow | Use Google Colab (T4) or Nebius GPU |
| GPU | Set `device_type='cuda'`, `torch_dtype=torch.bfloat16` |

---

## Option B: Nebius Cloud GPU + Consulting Claude Code

### Setup on Nebius

1. Open your Nebius Cloud Jupyter notebook (GPU instance — L4 or A100 recommended)
2. The workshop notebook (`workshop.ipynb`) is already configured for `device_type = 'cuda'` and `torch.bfloat16` — **no changes needed**
3. Install dependencies in the first cell:
   ```python
   !pip install colorama datasets huggingface_hub transformers accelerate -q
   ```
4. HuggingFace login — store your token as a Nebius secret or paste directly:
   ```python
   from huggingface_hub import login
   login(token="YOUR_HF_TOKEN")
   ```
5. Load models with `device_map="auto"` so they land on GPU automatically:
   ```python
   big_model = AutoModelForCausalLM.from_pretrained(
       big_model_name, torch_dtype=torch_dtype, device_map="auto"
   )
   small_model = AutoModelForCausalLM.from_pretrained(
       small_model_name, torch_dtype=torch_dtype, device_map="auto"
   )
   tokenizer = AutoTokenizer.from_pretrained(big_model_name)
   ```

### GPU-specific code differences vs CPU

| Thing | CPU version | GPU version |
|---|---|---|
| `device_type` | `'cpu'` | `'cuda'` |
| `torch_dtype` | `torch.float32` | `torch.bfloat16` |
| Model loading | `AutoModelForCausalLM.from_pretrained(name)` | add `device_map="auto"`, `torch_dtype=torch_dtype` |
| `input_ids.to(device)` | `.to('cpu')` | `.to('cuda')` |
| `torch.tensor(..., device=device)` | no change — just pass `device` variable | no change |

Everything else (functions, acceptance logic, colorama, benchmark) is **identical**.

### How to consult Claude while working in Nebius

Since Nebius is a remote Jupyter environment, you can't run Claude Code inside it directly. Use this workflow:

**Option 1 — Copy-paste into this terminal session (fastest)**
- Write or copy code from your Nebius notebook into a message here
- Paste error tracebacks, unexpected outputs, or tensor shapes directly
- Ask: *"this cell gives me X, expected Y — what's wrong?"*

**Option 2 — Edit the local file, copy to Nebius**
- Edit code here in `asgmnt_plan.md` or a local `.py` scratch file
- Copy the corrected cell into your Nebius notebook manually

**What to share when asking for help:**
```
# Always include:
1. The cell code you're running
2. The full error traceback (or the wrong output)
3. Tensor shapes if a shape mismatch — e.g. print(q.shape, p.shape)
4. Which step you're on (Step 1 / 2 / 3 / 4)
```

**Common GPU gotchas to watch for:**
- `RuntimeError: Expected all tensors to be on the same device` — make sure `input_ids`, `new_ids`, and `draft_token_tensor` are all `.to(device)`
- `torch.multinomial` needs float32 on some CUDA versions — cast: `probs.float()` before sampling, then the token id is device-agnostic
- Memory: 1B + 270M in bfloat16 needs ~2.5 GB VRAM — any Nebius GPU is fine
- `colorama` ANSI codes render in terminal but not in Jupyter — use `IPython.display` or just `print` and ignore the color escape chars in notebook output

---

## Full Working Notebook Cell Order

```
[1] Login to HuggingFace (or skip if using GPT-2 variants)
[2] !pip install colorama datasets
[3] Imports + device + model names
[4] Load tokenizer + models (with device_map if GPU)
[5] gamma, seqlen, input_text params
[6] generate_draft_and_verify()
[7] speculative_sampling()
[8] speculative_decode() loop + colorama
[9] Single-prompt test: speculative_decode(big_model, small_model, tokenizer, 'once upon', gamma=5, seqlen=30, device=device_type)
[10] Fineweb benchmark loop
```

---

## Key Correctness Checks

- `q` shape must be `[gamma, vocab]` — probabilities (not logits), summing to 1
- `p` shape must be `[gamma+1, vocab]` — the +1 is the bonus position
- Acceptance prob is `min(1, p[i, x̃ᵢ] / q[i, x̃ᵢ])` — scalar, using the actual draft token's probability
- Corrected distribution: `normalize(max(0, p − q))` — element-wise, over full vocab
- Bonus token position: `p[gamma]` — the output position *after* all γ draft tokens

---

## Notes on the Paper

- The algorithm is **lossless**: the distribution of accepted tokens matches exactly what Mp would have generated directly.
- The expected number of tokens generated per step is `γ·α + 1` where α is the per-token acceptance rate.
- Speedup comes when the draft model is much faster than the target model and α is high (models generate similar text).

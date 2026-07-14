# Training Optimizations for Larger Models

**Date:** 2026-07-14
**Context:** Pre-training MIR-LM-Small (~125M params) on RX 9070 XT 16GB. Current baseline achieves GPU matmul with ~1.6 GB for model+optimizer states, leaving ~14 GB headroom for activations.

---

## Current Baseline

| Metric | Value |
|---|---|
| Model params | ~125M |
| Weight memory (BF16) | ~0.23 GB |
| Optimizer states (AdamW fp32) | ~1.40 GB |
| Fixed overhead total | ~1.63 GB |
| Micro batch size | 4 |
| Sequence length | 2048 |
| Gradient accumulation | 64 |
| Mixed precision | BF16 AMP |
| Attention | Manual `matmul` + `softmax` |
| Gradient checkpointing | Not enabled |

---

## Tier 1 — Drop-in Changes (No Architecture Changes)

### 1. Replace Manual Attention with `F.scaled_dot_product_attention`

`torch.backends.cuda.flash_sdp_enabled()` returns `True` on ROCm 7.2. Switching from the manual `matmul` + `softmax` pattern to `F.scaled_dot_product_attention` delegates to the Flash Attention backend when possible, reducing attention memory from O(n²) to O(n).

**Impact:** Lower peak memory, faster attention, negligible accuracy change.

**Implementation:** Replace the attention score computation block in `GroupedQueryAttention.forward` (model.py:162-169) with a single call to `F.scaled_dot_product_attention(xq, keys, values, mask, dropout_p=0.0, is_causal=False)`.

**Note:** The causal mask needs adjustment — SDPA expects `is_causal=True` or a custom attn_mask. The current `-1e9` upper-triangular mask works but should use a float mask with `is_causal=False`, or just pass `is_causal=True` with `attn_mask=None`.

### 2. Gradient / Activation Checkpointing

Wrap each `TransformerBlock.forward` call with `torch.utils.checkpoint.checkpoint()`. During the backward pass, intermediate activations are recomputed instead of stored, reducing activation memory by ~60% at ~15% compute overhead.

**Implementation (model.py):**
```python
from torch.utils.checkpoint import checkpoint

# In MIRLM.forward, replace:
h = layer(h, cos, sin, mask=mask, ...)
# with:
h = checkpoint(layer, h, cos, sin, mask, use_cache, start_pos, use_reentrant=False)
```

**Impact:** Essential for scaling beyond 250M params. Without it, activation memory dominates at larger widths.

### 3. Profile and Increase Micro Batch Size

The current micro batch of 4 is conservative. With the 14 GB headroom, likely 8–16 fits. Run a sweep:
```
for bs in 4 8 12 16 20; do
    python train.py --batch_size $bs --warmup_steps 50 --epochs 1
done
```
Find the largest batch that doesn't OOM, then adjust gradient accumulation to maintain the same global batch size.

---

## Tier 2 — Moderate Effort, Significant Memory Savings

### 4. 8-bit AdamW Optimizer

AdamW stores 3 fp32 values per parameter (master copy + 2 moments) = 4 bytes × 3 = 12 bytes/param. A block-wise 8-bit Adam reduces this to ~2 bytes/param (75% reduction).

**Options:**
- **`bitsandbytes`** (`pip install bitsandbytes`) — well-tested, but ROCm wheel availability is inconsistent. If installable, use `bnb.optim.AdamW8bit`.
- **Custom implementation** — write a simplified block-wise 8-bit Adam optimizer in ~100 lines. The core idea: store optimizer states as `torch.uint8` with block-level scaling factors, dequantize on-the-fly during the step. This avoids external dependencies.

**Memory savings for target models:**

| Model Size | AdamW (fp32) | 8-bit Adam | Savings |
|---|---|---|---|
| 125M | 1.40 GB | 0.35 GB | ~1.1 GB |
| 500M | 5.60 GB | 1.40 GB | ~4.2 GB |
| 1B | 11.2 GB | 2.80 GB | ~8.4 GB |

### 5. CPU-Offloaded Optimizer States

Move AdamW moments to host RAM during forward/backward passes. Only transfer to GPU during the optimizer step.

**Implementation:** Use `torch.Tensor.pin_memory()` for the state tensors and do the step with the tensors on CPU. Can combine with `torch.amp.autocast` — the master weights stay in fp32 on CPU, gradients arrive on GPU, then the step runs on CPU.

**Trade-off:** Adds ~50–200 ms per step for CPU-GPU synchronization. This is acceptable when gradient accumulation is large (64 steps × gradient computation >> 50 ms).

---

## Tier 3 — Architectural Changes

### 6. Parallel Attention + FFN (PaLM-Style)

Replace the sequential:
```python
h = x + attention(norm(x))
out = h + ffn(norm(h))
```
with the parallel formulation:
```python
h = x + attention(norm(x)) + ffn(norm(x))
```
Reduces the number of sequential operations per layer from 2 to 1, which increases throughput at a given batch size. No accuracy loss has been observed in published work (PaLM, GPT-J, Cerebras-GPT).

### 7. Scaled-Up Model Architectures

With Tiers 1–2 applied, estimate the maximum model sizes that fit 16 GB VRAM (micro batch = 1):

| Model | Params | d_model | n_layers | n_heads | d_ff | Feasible |
|---|---|---|---|---|---|---|
| Current | 125M | 768 | 12 | 12 | 2048 | ✅ (batch 16+) |
| Small XL | 350M | 1024 | 24 | 16 | 4096 | ✅ (batch 1-2) |
| Medium | 500M | 1280 | 24 | 20 | 5120 | ✅ (batch 1) |
| Large | 1B | 1536 | 32 | 24 | 6144 | ⚠️ needs CPU-offload |

**Recommendation:** 500M is the practical upper limit for 16 GB VRAM without CPU offloading. With CPU-offloaded optimizer states and gradient checkpointing, 700M–1B may be achievable.

---

## Implementation Priority

1. ✅ Replace attention with `F.scaled_dot_product_attention` (model.py, 15 min)
2. ✅ Add activation checkpointing (model.py, 5 min)
3. ✅ Profile max micro batch size (train.py CLI)
4. ⏳ 8-bit AdamW optimizer (train.py, depends on dep check)
5. ⏳ CPU-offloaded optimizer (train.py, 1-2 hrs)
6. ⏳ Parallel attention+FFN (model.py, 30 min)
7. ⏳ Scale to 350M–500M architecture

---

## Notes

- Flash Attention is confirmed available (`torch.backends.cuda.flash_sdp_enabled() = True`, ROCm 7.2, gfx1201)
- BF16 is supported by RDNA4 hardware
- Device capability reported as `(12, 0)` (gfx1200/gfx1201)
- Current HIP runtime: 7.2.53211

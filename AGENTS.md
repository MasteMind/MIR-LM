# AGENTS.md — MIR-LM

## Project
**MIR-LM** (Multilingual Indian Reasoning Language Model) — a from-scratch, decoder-only Transformer LLM (LLaMA/Mistral-style: RMSNorm + RoPE + Grouped-Query Attention + SwiGLU). Goal: small local model for code generation/reasoning with native Indian-language support (Hindi, Tamil, Telugu, Gujarati, Bengali, Malayalam, Kannada). Target hardware: AMD Radeon RX 9070 XT 16GB (RDNA4 / gfx1201). Dual-path strategy: **native Ubuntu dual-boot + ROCm for pre-training**, DirectML on Windows for dev/verification. Status: pipeline verified on DirectML, ROCm pre-training pending.

## Files
- `model.py` — the core architecture (`ModelArgs`, RMSNorm, real-valued RoPE, GQA, SwiGLU, `TransformerBlock`, `MIRLM`). Ends with a `__main__` diagnostics block.
- `train.py` — pre-training loop (AdamW + cosine LR + linear warmup, gradient accumulation, BF16/FP16 AMP, checkpointing, `--resume`). Backend-aware (CUDA/ROCm first, DirectML fallback, CPU last).
- `prepare_data.py` — small-corpus packer: tokenizes `data_raw/` + workspace `*.py`/`*.md` into `data_bin/{train,val}.npy` + `metadata.txt` (NumPy `.npy` loaded into RAM, not memory-mapped).
- `download_datasets.py` — real data: streams ~30M tokens from Hugging Face (Flytech python-codes-25k, cfilt/iitb-english-hindi, HuggingFaceFW/fineweb-edu) into the same `data_bin/*.npy` format.
- `generate.py` — autoregressive streaming generator with KV cache; temperature / top-k / top-p sampling; `--checkpoint` CLI (auto-picks latest if omitted).
- `train_tokenizer.py` — trains a 32k Byte-Level BPE tokenizer. Prefers a real HF-streamed corpus (code + Hindi + English); falls back to an inline placeholder corpus offline. Writes `corpus_temp/` and `tokenizer/`.
- `test_tokenizer.py` — tokenizer encode/decode round-trip checks (not a unit-test suite).
- `test_gpu.py` — GPU/ROCm environment probe for Linux/Ubuntu.
- `test_gpu_win.py` — GPU DirectML probe for native Windows.
- `setup_ubuntu.sh` — native Ubuntu dual-boot env bootstrap (ROCm prereqs documented, then conda + pip).
- `setup_wsl.sh` — WSL2 alternative path (sets `HSA_ENABLE_DXG_DETECTION`, the WSL2-only DXG layer).
- `docs/` — `(README|architecture|decisions|reasoning_design|walkthrough).md`. Read these before changing sensitive areas.

## Commands
Each script is run directly. All paths are relative to the repo root — `cd` into the repo before running.
- `python model.py` — instantiate model, print param count, run a dummy forward pass.
- `python train_tokenizer.py` — build corpus (HF if `datasets` available, else placeholder) and train + save BPE tokenizer to `./tokenizer`.
- `python prepare_data.py` — pack `data_raw/` + workspace files into `data_bin/`.
- `python download_datasets.py` — stream ~30M real tokens from HF into `data_bin/` (needs network + `datasets`).
- `python train.py` — pre-train. `--warmup_steps` defaults to 2000 (matches `architecture.md` §3; use ~50 for the 30M-token dev run, which yields only ~285 total steps). `--resume PATH` restores model/optimizer/step/epoch.
- `python generate.py` — interactive generation. `--checkpoint PATH` (default: latest `.pt` in `checkpoints/`).
- `python test_tokenizer.py` — verify tokenizer round-trips code + Indian scripts.
- `bash setup_ubuntu.sh` — native Ubuntu setup (after manual amdgpu/ROCm install; see script header).
- `bash setup_wsl.sh` — WSL2 alternative.
- Runtime deps (unpinned): `torch` (ROCm build), `tokenizers`, `transformers`, `numpy`, `datasets`, `tqdm`.

## Architecture & conventions
- **Config lives in the `ModelArgs` dataclass** (`model.py:8-19`) and is mirrored in `docs/architecture.md` §1. Keep these two in sync when changing model sizing.
- One architectural concept per `nn.Module` subclass; pre-norm residuals; `bias=False` on all Linears by convention.
- RoPE is intentionally **real-valued** (not complex) for cross-backend compatibility (DirectML/ROCm/CPU) — see `apply_rotary_emb`. Do not switch to complex-tensor RoPE.
- `MIRLM` stores `self.args`; other modules (`GroupedQueryAttention`, `SwiGLUFeedForward`, `TransformerBlock`) copy fields individually and do **not** keep a reference to `args`.
- On Linux, `torch.cuda.*` is the correct API (ROCm exposes the CUDA-compat shim); `torch.cuda.is_available()` returning True is the ROCm success signal. On Windows, `torch_directml.device()` is the fallback (guarded by `try/except ImportError`, never reached when CUDA is available).
- Output head is **untied** from embeddings (`self.output`, separate `nn.Linear`). Inference (`targets is None`) returns only last-token logits, shape `(bsz, 1, vocab)`.
- Causal mask uses `-1e9` (not `float("-inf")`) and is explicitly `torch.float32` — a DirectML NaN-safety choice that's harmless on ROCm. `generate.py`'s sampler matches.

## Known gotchas & design details
- **KV-cache `NameError` is FIXED.** `GroupedQueryAttention.__init__` now stores `self.max_seq_len = args.max_seq_len` (`model.py`) and `forward` uses `self.max_seq_len`. The cache is lazily allocated and **re-allocated whenever `start_pos == 0`** — that's the convention `generate.py` relies on to reset between runs. The broader hazard remains: `self.cache_k`/`self.cache_v` are mutable instance state pre-allocated to fixed `max_seq_len`/`bsz`, with no invalidation on batch-size change or concurrent/batched decoding — silent reuse hazard outside the single-stream `start_pos=0` prefill pattern.
- **Warmup-vs-data-size interaction.** `train.py`'s `--warmup_steps` defaults to 2000 (spec). On the 30M-token dev dataset (~285 total steps at global batch 256), 2000 warmup never completes — the model stays in linear warmup and never reaches peak LR or cosine decay. `train.py` prints a warning when `warmup_steps >= total_steps`. For dev runs pass `--warmup_steps 50`; 2000 is correct for the planned 5–10B token run.
- **ROCm wheel / RDNA4 verification.** The RX 9070 XT is RDNA4 (gfx1201). ROCm 6.0 wheels may predate full RDNA4 support. After `setup_ubuntu.sh`, run `test_gpu.py` — if `torch.cuda.is_available()` is False, use a newer ROCm wheel index (6.2+) or a system ROCm build.
- **AMP is CUDA/ROCm-only.** `train.py` enables autocast only when `torch.cuda.is_available()` and not DirectML; on DirectML it silently runs fp32. BF16 is used if `torch.cuda.is_bf16_supported()` is True, else FP16 with `GradScaler`.
- **`train.py` save-only by default; resume via `--resume`.** Checkpoints save model + optimizer + scaler + step + epoch + args. Resume uses `map_location="cpu"` (ROCm-safe) and moves optimizer state tensors onto the device.
- **DirectML-saved checkpoints** in `checkpoints/` carry `privateuseone` device tags; `map_location="cpu"` *should* remap them onto ROCm — **verify one loads before relying on it** rather than assuming.
- **`generate.py` checkpoint loading** uses `map_location="cpu"` then `.to(device)` (the documented DirectML workaround; correct and portable on ROCm).
- **Dataloader size clamp:** if dataset token count is smaller than `batch_size`, the custom `DataLoader` clamps the effective batch size to avoid `StopIteration` crashes. Empty `val.npy` can still crash the eval loop's uncaught second `next()` — bites on tiny data, fine on the 30M run.
- **Param count unreconciled**: docs/decisions say 135M; computing from `ModelArgs` gives ~125M. Either reconcile the doc or revisit `d_ff`.
- **Dataset mix discrepancy**: `download_datasets.py` targets 25% Indian text; `docs/reasoning_design.md` specifies ~30%. Left as a documented decision.

## Docs to read before sensitive edits
- `docs/architecture.md` — exact config, component math, training hyperparameters.
- `docs/decisions.md` — ADR log with rationale (Decision 4 supersedes Decision 2: native Ubuntu dual-boot over WSL2). Follow ADR style when adding decisions.
- `docs/reasoning_design.md` — "code-as-thought" reasoning strategy (uses the `0x1F` ASCII Unit Separator as a scratchpad delimiter). Relevant if touching tokenizer special tokens or generation.
- `docs/walkthrough.md` — end-to-end usage walkthrough.

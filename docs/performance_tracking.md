# Training Performance & Metrics Tracking Log

This log tracks model specifications, pre-training runs, VRAM footprint, throughput speeds, and validation metrics to compare baselines and optimizations.

---

## 1. Run History Table

| Run ID / Date | Model Size | Vocab Size | Optimizations Enabled | Batch config (micro / accum / global) | Final Train Loss | Final Val Perplexity | Step Time | Throughput (True / Logged)* | Peak VRAM |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run 1 (Baseline)** <br> 2026-07-14 | 108.74M | 21,631 | None (Baseline manual attention, no checkpointing) | 1 / 256 / 256 | 4.22 | 558.30 (Step 100) | 32.0s | **16,376 tokens/s** <br> (Logged: 3,275) | ~7.8 GB |
| **Run 2 (Optimized)** <br> 2026-07-14 | 108.74M | 21,631 | SDPA (FlashAttention-2), Activation Checkpointing | 4 / 64 / 256 | 4.18 | 536.26 (Step 100) | 12.6s | **41,590 tokens/s** <br> (Logged: 8,308) | ~6.4 GB |
| **Run 3 (Multilingual)** <br> 2026-07-14 | **116.23M** | **26,505** | SDPA (FlashAttention-2), Activation Checkpointing | 4 / 64 / 256 | **4.36** | **530.25** (Step 100) | **12.7s** | **41,260 tokens/s** <br> (Logged: 8,099) | **~6.6 GB** |

*\*Note on Throughput Reporting: The logging block in `train.py` has a reporting bug where it computes tokens/s by dividing a single step's tokens by the accumulated time of 5 steps (`dt`). This under-reports the training speed in logs by exactly 5× for all steps > 1. We have provided the true corrected tokens/s alongside the logged values.*

---

## 2. Run Details & Notes

### Run 1: Baseline Run
* **Date**: 2026-07-14
* **Hardware**: AMD Radeon RX 9070 XT 16GB
* **Software**: ROCm 7.2 + PyTorch 2.13.0
* **VRAM Footprint**: 
  * Micro-batch size 4 caused an Out Of Memory (OOM) error immediately on the first forward pass due to manual attention activations ($O(L^2)$ matrix size: 805 MB per layer).
  * Safely completed training at micro-batch size 1, retaining ~7.8 GB peak memory usage.
* **Throughput**: Average step duration: 32.0s. True throughput: **16,376 tokens/s**.
* **Loss Convergence**:
  * Train loss converged to 4.22.
  * Validation perplexity converged to 558.30 (step 100).
* **Generation Output**: Autoregressive completions showed basic Python indentation structures and English phrasing.

### Run 2: Optimized Run
* **Date**: 2026-07-14
* **Hardware**: AMD Radeon RX 9070 XT 16GB
* **Software**: ROCm 7.2 + PyTorch 2.13.0
* **Optimizations**: Added PyTorch SDPA (FlashAttention-2) and Activation Checkpointing (`use_reentrant=False`).
* **VRAM Footprint**: Micro-batch size 4 runs successfully with no OOM errors. Peak VRAM dropped to ~6.4 GB (a **18% VRAM reduction** despite processing 4× larger micro-batches).
* **Throughput**:
  * Average step duration dropped from **32.0s to 12.6s** (a **2.54× step time speedup**).
  * True throughput increased from **16,376 tokens/s to 41,590 tokens/s** (a **154% throughput speedup** due to better GPU core utilization).
* **Loss Convergence**: Final training loss: **4.1813**. Final validation perplexity: **536.26**.
* **Generation Output**: Produced more logical coding block comments (e.g. autocompleting Python array comments) and clean structured Javascript loops.

### Run 3: Multilingual & Multi-Script Run
* **Date**: 2026-07-14
* **Hardware**: AMD Radeon RX 9070 XT 16GB
* **Software**: ROCm 7.2 + PyTorch 2.13.0
* **Optimizations**: Added Tamil + Telugu scripts to tokenizer, integrated multi-language coding instructions (JS, SQL, C, HTML) from Magicoder, and implemented logit-based repetition penalty in generator.
* **VRAM Footprint**: Peak VRAM stabilized at **~6.6 GB** (only +200 MB increase despite the larger 116.2M parameter embedding size).
* **Throughput**:
  * Step duration: **12.7s**. True throughput: **41,260 tokens/s**. The extra 7.5 Million parameters did not degrade performance.
* **Loss Convergence**:
  * Final validation perplexity dropped to **530.25** at step 100, which represents the lowest validation perplexity achieved across all runs.
* **Generation Output**:
  * Repetition penalty of 1.2 successfully broke the infinite loops (e.g., repeating `return False` or numbers) that previously attracted greedy generation.
  * Outputs displayed high diversity and generated clean multi-line Python imports and functions.

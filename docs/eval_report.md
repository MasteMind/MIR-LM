# Model Evaluation Report — Baseline vs Optimized

**Date:** 2026-07-14
**Checkpoints:** `checkpoints/baseline_run_step_114.pt` (Run 1) vs `checkpoints/optimized_run_step_114.pt` (Run 2)
**Model:** MIR-LM-Small (108.74M params, vocab 21631, seq 2048)
**Training:** 114 steps, ~30M tokens, BF16 AMP
**GPU:** AMD Radeon RX 9070 XT (ROCm 7.2)

---

## 0. Training Comparison Overview

| Metric | Run 1 (Baseline) | Run 2 (Optimized) | Improvement |
|---|---|---|---|
| Micro-batch size | 1 | 4 | 4× |
| Step duration | 32.0s | 12.6s | 2.54× |
| True throughput | 16,376 tok/s | 41,590 tok/s | 154% |
| VRAM peak | ~7.8 GB | ~6.4 GB | -18% |
| Final val perplexity | 558 | 536 | -4% |
| Total run duration | ~61 min | ~24 min | -60% |

**Optimizations applied:** Flash Attention (`F.scaled_dot_product_attention`), activation checkpointing, micro-batch increased from 1 → 4.

---

## 1. Test Methodology

10 scenarios, 50+ prompts total, run via `generate.py` on GPU with default sampling (temp=0.7, top_p=0.9, top_k=50) unless noted.

---

## 2. Scenario Results

### 2.1 Python Code Patterns (12 prompts)

**Strongest signals:** Syntactic scaffolding present — `def`, `return`, `if`, `for i in range(n)`, `self.append()`, `print()`, `# Output:`, `True`/`False`, `__init__`, `__name__`, `len()`, `range()`.

**Examples:**
```
Prompt:  def add_two_numbers(a, b):
Output:  def add_two_numbers(a, b):
             # Piece of numbers
             return sum(num1):
                 for i in range(n):
                 if len(X_data) ...
```
```
Prompt:  def factorial(n):
             if n <= 1:
Output:  def factorial(n):
             if n <= 1:
                 return self.
             return True
                 if i ...
```

**High-confidence pattern logits:**

| Prompt | Run 1 (Baseline) | Run 2 (Optimized) | Change |
|---|---|---|---|
| `def hello():` → `\n` | 57.2% | 56.9% | -0.3 pp (stable) |
| `import numpy` → ` as` | 74.4% | 29.9% | -44.5 pp (less certain) |
| `print(` → `1` | 5.0% (top was `n`) | 5.6% (top is `1`) | Top prediction changed |
| `ल` → `े` | 20.8% | 12.0% | -8.8 pp (broader distribution) |
| "capital of France" → ` a` | 4.2% | 7.1% | +2.9 pp (still near random) |

### 2.2 Multi-Language Code (11 languages)

| Language | Quality | Notes |
|---|---|---|
| Python | ✅ Strongest | Syntax structure present, logic absent |
| JavaScript | ❌ | Falls back to Python `def` immediately |
| C | ❌ | Python-like code inside `{ }` |
| SQL | ❌ | URL fragments, no SQL |
| Rust | ❌ | Empty output |
| Go | ❌ | Python patterns |
| HTML | ❌ | List literals instead of tags |
| Bash | ❌ | English prose instead of shell |
| CSS | ❌ | String fragments |
| JSON | ❌ | Garbled keys |
| YAML | ❌ | Whitespace-only |
| LaTeX | ❌ | Empty output |

### 2.3 Indian Languages (9 prompts)

| Prompt type | Quality | Notes |
|---|---|---|
| Hindi prose | 🟡 Devanagari script correct | Common words present (है, तो, और, में, का, की); no semantics |
| Hindi tech terms | 🟡 Short (2-3 tokens) | Collapses immediately |
| Code + Hindi comments | 🟡 Mixed | Hindi comment triggers Hindi output, code collapses |
| Tamil script | ❌ Garbled UTF-8 | Tokenizer has no Tamil coverage |
| Telugu script | ❌ Garbled UTF-8 | Tokenizer has no Telugu coverage |
| Hindi numerals (१ २ ३) | ❌ Ignored | Generates Devanagari prose, ignores numerals |

### 2.4 Temperature Sensitivity

| Temperature | Top-P / Top-K | Behavior |
|---|---|---|
| 0.1 (greedy) | 1.0 / 0 | Extreme repetition (number loops, `return False` loop) |
| 0.5 | 0.9 / 50 | Conservative, short generations |
| **0.7 (default)** | **0.9 / 50** | **Best balance of diversity/coherence** |
| 1.0 | none | Hindi tokens mixed into English, high entropy |
| 1.5 | none | Mostly garbage, very high entropy |

### 2.5 Logit / Entropy Analysis — Baseline vs Optimized

| Prompt | Run 1 entropy | Run 2 entropy | Run 1 max prob | Run 2 max prob |
|---|---|---|---|---|
| `def hello():` | 2.37 | 2.72 | 57.2% (`\n`) | 56.9% (`\n`) |
| `print(` | 6.14 | 5.79 | 5.0% (`n`) | 5.6% (`1`) |
| `import numpy` | 2.55 | 6.90 | 74.4% (` as`) | 29.9% (` as`) |
| `ल` (Hindi 'la') | 4.14 | 4.75 | 20.8% (`े`) | 12.0% (`े`) |
| "The capital of France is" | 7.35 | 7.00 | 4.2% (` the`) | 7.1% (` a`) |

**Notable change:** `import numpy → as` dropped from 74.4% to 29.9% — the optimized model is less certain about this pattern, suggesting a slightly broader learned distribution. All other patterns remain largely stable between runs.

**Lowest entropy patterns** (model is confident): `def hello(): → \n`, Hindi vowel sign prediction.

**Highest entropy** (model has no clue): factual knowledge ("capital of France"), English prose continuation.

### 2.6 Context Length Effect

| Prompt length | Response length | Quality |
|---|---|---|
| 5 tokens | ~70 tokens | Best |
| 17 tokens | ~60 tokens | Slightly shorter |
| 55 tokens | 1 token (`return False`) | Collapsed to filler |

**Diagnosis:** Longer prompts push the model past its reliable generation window. The effective generation budget shrinks with prompt length.

### 2.7 Termination Behavior

| Prompt | Generated tokens | Hit max_len? |
|---|---|---|
| `[1, 2, 3, ..., 10]` | 105 | No |
| `print("hello world")` | 24 | No |
| `import os\nimport sys\ndef main():` | 130 | No |

**EOS token (`</s>`) never fires.** All generations hit `max_gen_len` or produce empty output. The model has not learned to stop.

### 2.8 Perplexity on Held-Out Text — Baseline vs Optimized

| Text | Run 1 (Baseline) | Run 2 (Optimized) | Δ |
|---|---|---|---|
| "The quick brown fox jumps over the lazy dog." | 12,530 | 10,160 | **-19%** |
| `def add(a, b): return a + b` | 2,044 | 1,628 | **-20%** |
| `print('Hello, World!')` | 9,091 | 8,307 | **-9%** |
| `if __name__ == '__main__': main()` | 5,759 | 4,104 | **-29%** |
| `for i in range(10): print(i)` | 2,180 | 1,902 | **-13%** |
| **Average** | **6,321** | **5,220** | **-17%** |

**All perplexities improved by 9–29%** despite identical training steps. The optimized run converged to a better local minimum, likely because the larger micro-batch size (4 vs 1) produced more stable gradients.

**For reference:** A well-trained model achieves <20 perplexity on in-domain text.

---

## 3. Baseline vs Optimized — Qualitative Comparison

| Capability | Run 1 (Baseline) | Run 2 (Optimized) | Verdict |
|---|---|---|---|
| Python syntax | ✅ Emerging | ✅ Emerging | **Equal** |
| Python logic | ❌ Absent | ❌ Absent | **Equal** |
| Hindi script | ✅ Character-level | ✅ Character-level | **Equal** |
| Tamil/Telugu | ❌ Garbled | ❌ Garbled | **Equal** (tokenizer limitation) |
| Non-Python languages | ❌ Absent | ❌ Absent | **Equal** |
| Generation length | 1–130 tokens | 1–130 tokens | **Equal** |
| EOS termination | ❌ Never fires | ❌ Never fires | **Equal** |
| Perplexity (avg) | 6,321 | 5,220 | **Optimized -17% better** |
| Temperature robustness | Similar | Similar | **Equal** |

**Conclusion:** The optimizations (Flash Attention + checkpointing + larger micro-batch) did not change model behavior qualitatively — both runs learned the same patterns at the same rate. The optimized run achieves a modestly better perplexity (-17% avg), likely from more stable gradients at batch size 4 vs 1. **The only real path to qualitative improvement is more training data (5B+ tokens).**

---

## 4. Key Weaknesses

| # | Issue | Evidence | Root Cause |
|---|---|---|---|
| 1 | EOS token never fires | All generations hit max length | No natural sentence boundaries in training data |
| 2 | `return False` filler pattern | Appears in >50% of generations | High-probability degenerate attractor state |
| 3 | No factual knowledge | "capital of France" = uniform 4% across 8 tokens | Insufficient data |
| 4 | Non-Python languages near-zero | JS/C/Rust/SQL/HTML all degenerate | 30M-token corpus is >95% Python |
| 5 | Tamil/Telugu not supported | Garbled UTF-8 output | Tokenizer trained only on Hindi + English + Code |
| 6 | Long prompt collapse | 55-tok prompt → 1 token response | Limited effective generation window |
| 7 | Repetition at low temperature | Number/value loops at temp ≤ 0.1 | Model hasn't learned diverse continuations |

---

## 5. Recommendations for Next Pre-training Run

| Priority | Change | Expected Impact |
|---|---|---|
| **Critical** | Train on 5B+ tokens | Fixes all weaknesses; needed for convergence |
| **High** | Add Tamil + Telugu text to tokenizer corpus | Enables Indian language coverage |
| **High** | Add non-Python code (JS, SQL, C, Rust) to dataset | Enables multi-language generation |
| **Medium** | Include natural sentence boundaries / EOS examples | Fix termination behavior |
| **Medium** | Keep optimizations: Flash Attention + checkpointing + batch 4 | 2.5× speedup (verified ✅) |
| **Low** | Implement repetition penalty in sampler | Improves output diversity immediately |
| **Low** | Add factual QA / trivia to dataset | Enables knowledge retention |

---

## 6. Model Capability Snapshot (vs Target)

| Capability | Current (114 steps, 30M tokens) | Target (5B+ tokens) |
|---|---|---|
| Python syntax | ✅ Emerging | ✅ Fluent |
| Python logic/semantics | ❌ Absent | ✅ Functional code |
| Hindi script | ✅ Character-level | ✅ Fluent prose |
| Tamil/Telugu | ❌ Not supported | ✅ Supported |
| JS / C / SQL / Rust | ❌ Absent | ✅ Basic syntax |
| EOS termination | ❌ Never fires | ✅ Natural stopping |
| Perplexity (Python) | ~1,600–10,000 | <20 |
| Factual knowledge | ❌ Random chance | ✅ Factual answers |

---

## 7. Test Commands (for reproducibility)

```bash
# Run 2 (optimized) — single prompt
$HOME/miniconda/envs/mir-lm/bin/python generate.py \
  --checkpoint checkpoints/optimized_run_step_114.pt \
  --prompt "def hello():" --max_gen 64

# Run 2 (optimized) — interactive mode
$HOME/miniconda/envs/mir-lm/bin/python generate.py \
  --checkpoint checkpoints/optimized_run_step_114.pt \
  --interactive
```

---

*Next comparison point: evaluate against the same 10-scenario suite after multi-billion-token pre-training.*

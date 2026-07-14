# Decision Log: MIR-LM

This document logs all key design, environment, and architectural decisions made during the development of MIR-LM.

---

## [2026-07-14] Decision 1: Model Sizing
* **Status**: Approved
* **Decision**: Target **MIR-LM-Small (135M parameters)** for the initial phase.
* **Rationale**: 
  - Since we are pre-training from scratch on a single GPU (AMD RX 9070 XT 16GB), training a >1B parameter model from scratch to high quality would take weeks/months.
  - A 135M parameter model serves as a perfect proof-of-concept. It can be trained on a reasonably sized dataset (5B to 10B tokens) in 1 to 3 days.
  - It allows us to verify that our tokenizer, dataset preprocessing, and PyTorch training pipeline are working optimally before scaling up to larger models (e.g., 360M or 1.5B).

---

## [2026-07-14] Decision 2: Development & Compute Environment
* **Status**: Approved (Dual-Path Strategy)
* **Decision**: Use **Windows Native (DirectML)** for pipeline development and local verification, and **Native Ubuntu OS (Dual-Boot)** for full pre-training.
* **Rationale**:
  - Setting up ROCm inside WSL2 encountered driver partition mapping issues (`/usr/lib/wsl/lib` not mounting AMD files), which would require complex Windows registry/virtualization debugging.
  - Using **Windows Native with PyTorch DirectML (`torch-directml`)** allows us to develop and verify all scripts (tokenizer, dataset preparation, model, training loop) inside this active session without any reboots or virtualization conflicts.
  - Since PyTorch code is cross-platform, once the pipeline is verified on Windows, the developer can boot into their native Ubuntu partition where native ROCm is fully supported and run the training script at maximum speed.


---

## [2026-07-14] Decision 3: Tokenizer and Multilingual Scope
* **Status**: Approved
* **Decision**: Train a **custom Byte-Pair Encoding (BPE) tokenizer** from scratch with a vocabulary size of **32,000**, with full support for Indian languages (multilingual) and code structure.
* **Rationale**:
  - **Code-optimization**: Standard English tokenizers fail to parse whitespace and tabs efficiently, leading to bloated token usage on indentation. A custom tokenizer will treat consecutive spaces/tabs as distinct single tokens.
  - **Indian Language Inclusion**: To make it the "first Local LLM of India," the vocabulary needs to support Devanagari and other Indian scripts without relying on character-by-character byte-fallback, which would slow down processing. A vocab of 32k is standard and provides the right trade-off between speed and representation power.

# MIR-LM Documentation

Welcome to the documentation folder for **MIR-LM**, India's first custom Local LLM built and trained from scratch.

This directory is designed to maintain a highly descriptive log of decisions, specifications, and progress from day one.

## Documentation Index

1. **[decisions.md](file:///c:/Users/mehra/OneDrive/Documents/MIR-LM/docs/decisions.md)**
   - Log of all architectural, environmental, and engineering decisions. Refer to this to understand *why* we chose certain model sizes, datasets, platforms, and hyperparameters.
2. **[architecture.md](file:///c:/Users/mehra/OneDrive/Documents/MIR-LM/docs/architecture.md)**
   - Exact model configurations, parameter shapes, mathematical explanations of the modern Transformer components (RMSNorm, RoPE, SwiGLU, GQA), and pre-training schedules.
3. **[reasoning_design.md](file:///c:/Users/mehra/OneDrive/Documents/MIR-LM/docs/reasoning_design.md)**
   - Detailed logic of the "Code-as-Thought" reasoning formulation, scratchpad token usage, and how we avoid token taxing on Indian languages.
4. **[walkthrough.md](file:///c:/Users/mehra/OneDrive/Documents/MIR-LM/docs/walkthrough.md)**
   - Verification logs of the pipeline, details of the DirectML specific bug resolutions, and instructions for native Ubuntu ROCm pre-training.

## Project Overview

* **Project Name**: MIR-LM (Multilingual Indian Reasoning Language Model)
* **Goal**: Build and train a small, high-quality, local language model from scratch, optimized for code generation/reasoning and supporting Indian languages.
* **Target Hardware**: 1x AMD Radeon RX 9070 XT (16GB VRAM)
* **Backend Platform**: **Dual-Path Strategy** (Windows Native DirectML for development/verification; Dedicated Ubuntu partition with native ROCm for full pre-training).
* **Phase-1 Model Size**: 135M Parameters (decoder-only)
* **Status**: Development pipeline complete and verified on native Windows. Ready for the pre-training run on native Ubuntu.

# Architecture Specification: MIR-LM-Small (135M)

This document details the configuration, layer definitions, and hyperparameter specifications for **MIR-LM-Small**.

---

## 1. Model Configuration

We employ a decoder-only Transformer model similar to LLaMA and Mistral. Below are the precise shape and configuration parameters:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| `vocab_size` | 32,000 | Vocabulary size of the custom trained BPE tokenizer |
| `max_seq_len` | 2,048 | Context window size (in tokens) |
| `n_layers` | 12 | Number of Transformer blocks (decoder layers) |
| `d_model` | 768 | Hidden dimension size |
| `n_heads` | 12 | Number of query attention heads |
| `n_kv_heads` | 4 | Number of key/value attention heads (Grouped-Query Attention) |
| `d_ff` | 2,048 | Intermediate dimension for SwiGLU FFN block |
| `norm_eps` | 1e-5 | Epsilon value for RMSNorm |
| `rope_theta` | 10,000.0 | Base frequency for Rotary Positional Embeddings |
| `bias` | False | Disables bias in all Attention and MLP Linear layers |

---

## 2. Key Components & Implementation Notes

### Pre-Normalization (RMSNorm)
We use Root Mean Square Normalization (RMSNorm) instead of standard LayerNorm, placed *before* the self-attention and MLP blocks. A final RMSNorm layer is applied before the unembedding projection to the vocabulary.
$$\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d} \sum_{j=1}^{d} x_j^2 + \epsilon}} \gamma_i$$
This is faster to compute and provides equal or better training stability.

### Rotary Position Embeddings (RoPE)
We apply RoPE to the query and key vectors inside each attention block. This encodes absolute positional information via a rotation matrix, which preserves relative distance relationships between tokens and enables length extrapolation during inference.

### Grouped-Query Attention (GQA)
To reduce memory bandwidth overhead during decoding (inference), we use GQA. 
- With `n_heads = 12` and `n_kv_heads = 4`, every group of 3 query heads shares a single key-value head.
- This results in a $3\times$ reduction in the size of the Key-Value (KV) cache compared to Multi-Head Attention (MHA), which is crucial for local deployment on memory-constrained devices.

### SwiGLU Feed-Forward Network
We replace the standard MLP layer (Linear -> GeLU -> Linear) with a SwiGLU activation block.
$$\text{SwiGLU}(x) = \left( \text{Silu}(x W_{gate}) \otimes (x W_{up}) \right) W_{down}$$
where $\otimes$ is element-wise multiplication.
- $W_{gate} \in \mathbb{R}^{d_{model} \times d_{ff}}$
- $W_{up} \in \mathbb{R}^{d_{model} \times d_{ff}}$
- $W_{down} \in \mathbb{R}^{d_{ff} \times d_{model}}$

This has been empirically proven to improve reasoning capabilities at equivalent parameter counts.

---

## 3. Training Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Optimizer** | AdamW | $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 1\text{e-}8$ |
| **Weight Decay** | 0.1 | Applied to all weight matrices (except bias/norms) |
| **Base LR** | 3e-4 | Peak learning rate |
| **Warmup Steps** | 2000 | Linear learning rate warmup |
| **LR Schedule** | Cosine Annealing | Decay down to 3e-5 (10% of peak LR) |
| **Global Batch Size**| 256 | Sequences per training step (approx. 524,288 tokens) |
| **Micro Batch Size** | 4 | Sequences per GPU step (adjusted for 16GB VRAM) |
| **Grad Accumulation**| 64 | Steps to accumulate gradients before updating weights |
| **Precision** | FP16/BF16 | Auto-mixed precision for speed and memory efficiency |

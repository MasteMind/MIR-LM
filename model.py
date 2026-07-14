import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

@dataclass
class ModelArgs:
    vocab_size: int = 32000
    max_seq_len: int = 2048
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 2048  # Intermediate dimension for SwiGLU FFN
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    bias: bool = False

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    More computationally efficient than standard LayerNorm while providing similar regularization.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cosine and sine frequency tables for Rotary Position Embeddings (RoPE).
    Using real-valued rotations for maximum compatibility across backends (DirectML/ROCm/CPU).
    """
    assert head_dim % 2 == 0
    # inv_freq = 1.0 / (theta ** (2 * [0, 1, 2, ..., d/2] / d))
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim // 2)
    
    # Duplicate frequencies for the rotation arithmetic: (cos, sin) applied to halves of the vector
    freqs = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, head_dim)
    return freqs.cos(), freqs.sin()

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply precomputed Rotary Position Embeddings to query/key tensors.
    x: (batch_size, seq_len, n_heads, head_dim)
    cos, sin: (seq_len, head_dim)
    """
    # Align shapes for broadcasting: (1, seq_len, 1, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    # Split x into first half and second half along the head_dim axis
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    
    # Rotate representation: [-x2, x1]
    rx = torch.cat((-x2, x1), dim=-1)
    
    # Return rotated tensor: x * cos(t) + rx * sin(t)
    return x * cos + rx * sin

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat Key/Value heads to match Query heads in Grouped-Query Attention (GQA).
    x: (batch_size, seq_len, n_kv_heads, head_dim)
    """
    if n_rep == 1:
        return x
    bs, seq_len, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bs, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(bs, seq_len, n_kv_heads * n_rep, head_dim)
    )

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) layer.
    Allows query heads to share key-value heads, reducing memory bandwidth during autoregressive decoding.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.d_model = args.d_model
        self.max_seq_len = args.max_seq_len
        
        assert self.n_heads % self.n_kv_heads == 0
        self.num_queries_per_kv = self.n_heads // self.n_kv_heads
        self.head_dim = args.d_model // args.n_heads
        
        # Projections
        self.wq = nn.Linear(args.d_model, args.n_heads * self.head_dim, bias=args.bias)
        self.wk = nn.Linear(args.d_model, args.n_kv_heads * self.head_dim, bias=args.bias)
        self.wv = nn.Linear(args.d_model, args.n_kv_heads * self.head_dim, bias=args.bias)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.d_model, bias=args.bias)
        
        # KV Cache containers (for inference)
        self.cache_k = None
        self.cache_v = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        start_pos: int = 0
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        
        # Project inputs to Query, Key, and Value
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        
        # Apply Rotary Position Embeddings
        # Slice cos/sin to the exact sequence length / position we need
        cos_sliced = cos[start_pos : start_pos + seqlen]
        sin_sliced = sin[start_pos : start_pos + seqlen]
        xq = apply_rotary_emb(xq, cos_sliced, sin_sliced)
        xk = apply_rotary_emb(xk, cos_sliced, sin_sliced)
        
        # Handle KV cache for decoding
        if use_cache:
            if self.cache_k is None or start_pos == 0:
                # Initialize caches
                self.cache_k = torch.zeros(bsz, self.max_seq_len, self.n_kv_heads, self.head_dim, device=x.device, dtype=x.dtype)
                self.cache_v = torch.zeros(bsz, self.max_seq_len, self.n_kv_heads, self.head_dim, device=x.device, dtype=x.dtype)
            
            # Store values
            self.cache_k[:, start_pos : start_pos + seqlen] = xk
            self.cache_v[:, start_pos : start_pos + seqlen] = xv
            
            # Retrieve complete history up to current length
            keys = self.cache_k[:, : start_pos + seqlen]
            values = self.cache_v[:, : start_pos + seqlen]
        else:
            keys = xk
            values = xv
            
        # Repeat KV heads to match query heads (GQA support)
        keys = repeat_kv(keys, self.num_queries_per_kv)  # (bsz, seqlen_kv, n_heads, head_dim)
        values = repeat_kv(values, self.num_queries_per_kv)  # (bsz, seqlen_kv, n_heads, head_dim)
        
        # Reshape to compute scaled dot product attention
        # (bsz, n_heads, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Compute scaled dot product attention using PyTorch SDPA (highly memory-efficient)
        if use_cache:
            output = F.scaled_dot_product_attention(
                xq, keys, values,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False
            )
        else:
            output = F.scaled_dot_product_attention(
                xq, keys, values,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True  # Triggers fast causal FlashAttention kernels
            )
        
        # Concatenate heads and project output back to d_model
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    Formulation: (SiLU(x * W_gate) * (x * W_up)) * W_down
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.d_model, args.d_ff, bias=args.bias)  # W_gate
        self.w2 = nn.Linear(args.d_ff, args.d_model, bias=args.bias)  # W_down
        self.w3 = nn.Linear(args.d_model, args.d_ff, bias=args.bias)  # W_up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish(x * W1) * (x * W3) -> W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    """
    Decoder-only Transformer layer incorporating pre-RMSNorm, Attention, and SwiGLU FFN.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.attention = GroupedQueryAttention(args)
        
        self.ffn_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.feed_forward = SwiGLUFeedForward(args)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        start_pos: int = 0
    ) -> torch.Tensor:
        # Residual self-attention
        h = x + self.attention(
            self.attention_norm(x),
            cos,
            sin,
            mask=mask,
            use_cache=use_cache,
            start_pos=start_pos
        )
        # Residual Feed-Forward
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class MIRLM(nn.Module):
    """
    MIR-LM core architecture.
    Decoder-only Transformer trained from scratch.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.d_model)
        
        # Transformer Layers
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        
        # Final Norm and Head
        self.norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.output = nn.Linear(args.d_model, args.vocab_size, bias=args.bias)
        
        # Precompute RoPE frequency tables
        cos, sin = precompute_rope_freqs(args.d_model // args.n_heads, args.max_seq_len, args.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        
        # Ensure we move RoPE cache tensors to correct device
        cos = self.rope_cos
        sin = self.rope_sin
        
        # Create causal mask if we have multiple tokens (pre-training or prompt encoding)
        mask = None
        if seqlen > 1:
            # Standard causal lower-triangular mask
            mask = torch.full((seqlen, seqlen), -1e9, device=tokens.device, dtype=torch.float32)
            mask = torch.triu(mask, diagonal=1)
            # Expand to (1, 1, seqlen, seqlen) for broadcasting
            mask = mask.unsqueeze(0).unsqueeze(1)
            
        # Pass through layers (using activation checkpointing during training to save VRAM)
        for layer in self.layers:
            if self.training:
                h = checkpoint(
                    layer,
                    h,
                    cos,
                    sin,
                    mask,
                    use_cache,
                    start_pos,
                    use_reentrant=False
                )
            else:
                h = layer(h, cos, sin, mask=mask, use_cache=use_cache, start_pos=start_pos)
            
        # Final normalization
        h = self.norm(h)
        
        # Compute loss if targets are provided (for training)
        loss = None
        if targets is not None:
            logits = self.output(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # Inference mode: only compute logits for the last token to save computation
            logits = self.output(h[:, -1:, :])
            
        return logits, loss

if __name__ == "__main__":
    # Diagnostic test to check architecture shape and parameters
    args = ModelArgs()
    model = MIRLM(args)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MIR-LM-Small Model initialized successfully.")
    print(f"Total parameters: {total_params / 1_000_000:.2f} Million ({total_params:,} parameters)")
    
    # Dummy forward pass check
    test_input = torch.randint(0, args.vocab_size, (2, 512))
    test_target = torch.randint(0, args.vocab_size, (2, 512))
    
    logits, loss = model(test_input, targets=test_target)
    print(f"Forward pass successful. Logits shape: {logits.shape}, Loss: {loss.item():.4f}")

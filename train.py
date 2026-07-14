import os
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from model import ModelArgs, MIRLM

# Print CPU/GPU status
def print_gpu_status(device=None):
    if torch.cuda.is_available():
        print(f"CUDA/ROCm active. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Initial allocated memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    else:
        try:
            import torch_directml
            if device and "privateuseone" in str(device):
                print("DirectML active. Using AMD GPU.")
            else:
                print("Using CPU for training (GPU not detected by PyTorch).")
        except ImportError:
            print("Using CPU for training (GPU not detected by PyTorch).")

class DataLoader:
    """
    A simple dataloader that yields batches of inputs and targets from pre-saved NumPy token chunks.
    """
    def __init__(self, data_path, batch_size, seq_len):
        self.data = np.load(data_path)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_samples = len(self.data)
        
        # Shuffle indices for epochs
        self.indices = np.arange(self.num_samples)
        self.shuffle()
        self.current_idx = 0
        
    def shuffle(self):
        np.random.shuffle(self.indices)
        
    def __iter__(self):
        return self
        
    def __next__(self):
        effective_batch_size = min(self.batch_size, self.num_samples)
        if effective_batch_size == 0:
            raise StopIteration("No data samples available.")
            
        if self.current_idx + effective_batch_size > self.num_samples:
            # Epoch finished
            self.shuffle()
            self.current_idx = 0
            raise StopIteration
            
        batch_indices = self.indices[self.current_idx : self.current_idx + effective_batch_size]
        self.current_idx += effective_batch_size
        
        # Retrieve chunks of tokens
        # Each chunk is of size seq_len
        chunks = self.data[batch_indices]  # Shape: (batch_size, seq_len)
        
        # Input (x) is tokens from 0 to seq_len-1
        # Target (y) is tokens shifted by 1 (from 1 to seq_len)
        # Note: To do this, our chunks must have been saved with size seq_len + 1,
        # or we slice x and y inside the seq_len window.
        # Since prepare_data.py saves chunks of exact size seq_len,
        # we will use the standard language modeling approach:
        # x is the sequence up to the second-to-last token, y is shifted by 1.
        # This reduces active sequence length to seq_len - 1.
        x = torch.tensor(chunks[:, :-1].astype(np.int64))
        y = torch.tensor(chunks[:, 1:].astype(np.int64))
        
        return x, y

def get_lr(step, total_steps, warmup_steps, max_lr, min_lr):
    """
    Cosine learning rate decay with linear warmup.
    """
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def main():
    parser = argparse.ArgumentParser(description="Pre-train MIR-LM-Small")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size (fit in VRAM)")
    parser.add_argument("--grad_accum", type=int, default=64, help="Gradient accumulation steps")
    parser.add_argument("--max_lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Minimum decayed learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="LR linear warmup steps (matches architecture.md §3; use ~50 for the 30M-token dev run)")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--eval_interval", type=int, default=50, help="Evaluate every N steps")
    parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Where to save models")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint .pt to resume from (restores model, optimizer, step, epoch)")
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device (ROCm uses CUDA, DirectML uses torch_directml device)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import torch_directml
            device = torch_directml.device()
        except ImportError:
            device = torch.device("cpu")
            
    print(f"Training device selected: {device}")
    print_gpu_status(device)
    
    # 1. Load dataset metadata
    metadata_path = "data_bin/metadata.txt"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Data files not found in 'data_bin/'. Please run 'prepare_data.py' first!")
        
    metadata = {}
    with open(metadata_path, "r") as f:
        for line in f:
            k, v = line.strip().split("=")
            metadata[k] = int(v)
            
    print(f"Dataset metadata loaded: {metadata}")
    
    # 2. Instantiate Model
    model_args = ModelArgs(
        vocab_size=metadata["vocab_size"],
        max_seq_len=metadata["seq_len"],
        n_layers=12,
        d_model=768,
        n_heads=12,
        n_kv_heads=4,
        d_ff=2048,
        bias=False
    )
    
    print("Initializing MIR-LM-Small architecture...")
    model = MIRLM(model_args).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params / 1_000_000:.2f} Million")
    
    # 3. Create Optimizer with Weight Decay logic
    # Separate parameters: apply weight decay to 2D matrices (weights of Linears),
    # but not to 1D vectors (biases, RMSNorm weights)
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.max_lr, betas=(0.9, 0.95), eps=1e-8)
    
    # 4. Mixed Precision Setup
    is_directml = "privateuseone" in str(device)
    use_amp = torch.cuda.is_available() and not is_directml
    # Use bfloat16 if supported (RDNA 4/ROCm supports BF16), otherwise float16
    mixed_precision_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    print(f"Using mixed-precision (AMP): {use_amp} (Dtype: {mixed_precision_dtype})")
    
    # Initialize Gradient Scaler for FP16 (only enabled when not on BF16)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and mixed_precision_dtype == torch.float16))
    
    # 5. Dataloaders
    train_loader = DataLoader("data_bin/train.npy", args.batch_size, model_args.max_seq_len)
    val_loader = DataLoader("data_bin/val.npy", args.batch_size, model_args.max_seq_len)
    
    # 6. Training Loop Configuration
    # We estimate total steps based on number of samples and global batch size
    global_batch_size = args.batch_size * args.grad_accum
    steps_per_epoch = len(train_loader.data) // global_batch_size
    total_steps = steps_per_epoch * args.epochs
    
    if total_steps == 0:
        total_steps = 500  # Fallback for small local tests
        print(f"Data is small. Overriding total training steps to {total_steps}.")

    # Guard against the warmup-vs-data-size footgun: if warmup consumes the whole run,
    # the model never reaches peak LR or cosine decay. The 2000-step default matches
    # architecture.md §3 (intended for the 5-10B token run); for the 30M-token dev
    # dataset (~285 total steps) pass --warmup_steps 50 or similar.
    if args.warmup_steps >= total_steps:
        print(
            f"WARNING: warmup_steps ({args.warmup_steps}) >= total_steps ({total_steps}). "
            f"The model will never reach peak LR ({args.max_lr}) or begin cosine decay. "
            f"Reduce --warmup_steps or increase the dataset."
        )
        
    print(f"Global Batch Size: {global_batch_size} sequences ({global_batch_size * (model_args.max_seq_len - 1):,} tokens per step)")
    print(f"Steps per epoch: {steps_per_epoch}, Total pre-training steps: {total_steps}")
    
    step = 0
    epoch = 0
    t0 = time.time()

    # Resume from checkpoint if requested (restores model, optimizer, scaler, step, epoch).
    # Uses map_location="cpu" (same portable pattern as generate.py) then moves optimizer
    # state tensors onto the active device — required because optimizer state dicts load onto CPU.
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")
        print(f"Resuming from checkpoint: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        # Move optimizer state tensors (exp_avg, exp_avg_sq, ...) to the active device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        if "scaler_state_dict" in resume_ckpt and scaler.is_enabled():
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
        step = resume_ckpt.get("step", 0)
        epoch = resume_ckpt.get("epoch", 0)
        print(f"Resumed at step {step}, epoch {epoch}. Continuing training.")
    
    while step < total_steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        # Accumulate gradients
        for micro_step in range(args.grad_accum):
            try:
                x, y = next(train_loader)
            except StopIteration:
                epoch += 1
                print(f"--- Completed Epoch {epoch} ---")
                train_loader.shuffle()
                x, y = next(train_loader)
                
            x, y = x.to(device), y.to(device)
            
            # Forward pass under autocast (mixed precision)
            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=mixed_precision_dtype, enabled=use_amp):
                logits, loss = model(x, targets=y)
                # Scale loss to account for gradient accumulation
                loss = loss / args.grad_accum
                
            accum_loss += loss.item()
            
            # Backward pass
            if use_amp and mixed_precision_dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
        # Gradient clipping to prevent exploding gradients
        if use_amp and mixed_precision_dtype == torch.float16:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Step optimizer via scaler
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        step += 1
        
        # Update learning rate
        lr = get_lr(step, total_steps, args.warmup_steps, args.max_lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
        # Logging metrics
        if step % 5 == 0 or step == 1:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tokens_per_sec = (global_batch_size * (model_args.max_seq_len - 1)) / dt
            print(f"Step {step}/{total_steps} | Loss: {accum_loss:.4f} | LR: {lr:.2e} | Time: {dt*1000:.0f}ms | Speed: {tokens_per_sec:.0f} tokens/s")
            
        # Validation Evaluation
        if step % args.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 10  # Evaluate on 10 steps to save time
            
            with torch.no_grad():
                for v_step in range(val_steps):
                    try:
                        vx, vy = next(val_loader)
                    except StopIteration:
                        val_loader.shuffle()
                        vx, vy = next(val_loader)
                        
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=mixed_precision_dtype, enabled=use_amp):
                        _, loss = model(vx, targets=vy)
                    val_loss += loss.item()
                    
            val_loss /= val_steps
            perplexity = math.exp(val_loss) if val_loss < 20 else float('inf')
            print(f"--- Evaluation at step {step}: Val Loss: {val_loss:.4f} | Val Perplexity: {perplexity:.2f} ---")
            
        # Save checkpoints
        if step % args.save_interval == 0 or step == total_steps:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"mirlm_step_{step}.pt")
            print(f"Saving checkpoint to {checkpoint_path}...")
            torch.save({
                "step": step,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": accum_loss,
                "args": model_args
            }, checkpoint_path)
            
    print("Training finished successfully!")

if __name__ == "__main__":
    main()

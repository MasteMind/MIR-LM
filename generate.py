import os
import sys
import argparse
import torch
from tokenizers import Tokenizer
from model import ModelArgs, MIRLM

def sample_top_p_k(logits, generated_tokens=None, repetition_penalty=1.2, top_p=0.9, top_k=50, temperature=1.0):
    """
    Apply temperature scaling, repetition penalty, Top-K, and Top-P (nucleus) filtering to logits.
    """
    # 1. Apply repetition penalty
    if generated_tokens is not None and repetition_penalty != 1.0:
        for token_id in set(generated_tokens):
            logit = logits[token_id].item()
            if logit < 0:
                logits[token_id] = logit * repetition_penalty
            else:
                logits[token_id] = logit / repetition_penalty

    # Scale by temperature
    logits = logits / max(temperature, 1e-5)
    
    # Sort logits descending
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # 2. Apply Top-K filtering
    if top_k > 0:
        # Keep only the top-k indices, set the rest to -1e9 (matches model.py's mask sentinel)
        sorted_logits[top_k:] = -1e9
        
    # 3. Apply Top-P (nucleus) filtering
    if top_p < 1.0:
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        # We shift the mask to the right to keep the first token that exceeds top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        sorted_logits[sorted_indices_to_remove] = -1e9
        
    # Reconstruct logits in original order
    filtered_logits = torch.full_like(logits, -1e9)
    filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    
    # Sample from the filtered distribution
    probabilities = torch.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probabilities, num_samples=1)
    return next_token

@torch.no_grad()
def generate(model, prompt, tokenizer, max_gen_len=256, temperature=0.7, repetition_penalty=1.2, top_p=0.9, top_k=50, device="cpu"):
    """
    Autoregressive generation using GQA KV cache inside the model.
    Prints generated tokens to stdout in real-time.
    """
    model.eval()
    
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt).ids
    prompt_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    print(f"\n--- Prompt ---\n{prompt}\n--------------")
    print("--- Generated Response ---")
    
    # Print the prompt back first so it looks like a continuous response (optional)
    # Standard: just print what's generated in real-time
    
    bsz, seqlen = prompt_tokens.shape
    # Bound check must cover prompt + generation: the KV cache is pre-allocated to max_seq_len,
    # so decoding past it would write out of range. See model.py GroupedQueryAttention.forward.
    assert seqlen + max_gen_len <= model.args.max_seq_len, (
        f"Prompt ({seqlen}) + max_gen_len ({max_gen_len}) exceeds max_seq_len ({model.args.max_seq_len}); "
        f"shorten the prompt or reduce max_gen_len."
    )
    
    # Phase 1: Prefill phase (evaluates prompt, populates KV cache)
    # We pass the full prompt through the model
    # logits shape: (1, seqlen, vocab_size). We only care about the last token's logits
    logits, _ = model(prompt_tokens, use_cache=True, start_pos=0)
    
    # Sample the first new token (no repetition penalty needed for first token)
    next_token = sample_top_p_k(logits[0, -1], generated_tokens=None, repetition_penalty=1.0, top_p=top_p, top_k=top_k, temperature=temperature)
    
    generated_tokens = [next_token.item()]
    
    # Stream the first token
    first_word = tokenizer.decode([next_token.item()])
    sys.stdout.write(first_word)
    sys.stdout.flush()
    
    # Phase 2: Generation phase (evaluates token-by-token using KV cache)
    cur_pos = seqlen
    
    for _ in range(max_gen_len - 1):
        if next_token.item() == tokenizer.token_to_id("</s>"):
            break  # End of text token generated
            
        # Pass the single next_token through the model, updating cache at cur_pos
        # tokens shape: (1, 1)
        token_input = next_token.view(1, 1)
        logits, _ = model(token_input, use_cache=True, start_pos=cur_pos)
        
        # Sample next token with repetition penalty
        next_token = sample_top_p_k(logits[0, -1], generated_tokens=generated_tokens, repetition_penalty=repetition_penalty, top_p=top_p, top_k=top_k, temperature=temperature)
        generated_tokens.append(next_token.item())
        
        # Decode and print token in real-time
        token_text = tokenizer.decode([next_token.item()])
        sys.stdout.write(token_text)
        sys.stdout.flush()
        
        cur_pos += 1
        
    print("\n--------------------------")
    return tokenizer.decode(generated_tokens)

def main():
    # Set device (ROCm uses CUDA, DirectML uses torch_directml device)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import torch_directml
            device = torch_directml.device()
        except ImportError:
            device = torch.device("cpu")
    
    parser = argparse.ArgumentParser(description="MIR-LM generation playground")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint .pt (default: auto-pick latest in checkpoints/)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive prompt loop")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Logit penalty scale factor (>1.0)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    gen_args = parser.parse_args()

    # Resolve checkpoint: explicit --checkpoint, else auto-pick latest .pt in checkpoints/.
    checkpoint_path = gen_args.checkpoint
    if checkpoint_path is None:
        os.makedirs("checkpoints", exist_ok=True)
        checkpoints = [f for f in os.listdir("checkpoints") if f.endswith(".pt")]
        if checkpoints:
            def get_step_num(filename):
                try:
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        return int(parts[-1].split('.')[0])
                except ValueError:
                    pass
                return 0
            checkpoints.sort(key=get_step_num)
            checkpoint_path = os.path.join("checkpoints", checkpoints[-1])

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_args = checkpoint["args"]
        model = MIRLM(model_args).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("No model checkpoint found (pass --checkpoint PATH, or train first). Using random weights for dry-run.")
        args = ModelArgs(vocab_size=32000)
        model = MIRLM(args).to(device)
        
    # Load tokenizer
    tokenizer_path = "tokenizer/vocab.json"
    if not os.path.exists(tokenizer_path):
        print("Error: Tokenizer files not found in './tokenizer/'. Generating raw character/BPE placeholder first.")
        # Make a basic print and exit, or import tokenizers
        print("Please train the tokenizer first: python train_tokenizer.py")
        return
        
    # Load BPE tokenizer
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json") if os.path.exists("tokenizer/tokenizer.json") else None
    if tokenizer is None:
        from tokenizers import ByteLevelBPETokenizer
        tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
        
    # Prompt list
    prompt_list = [
        "def hello_world():\n    #",
        "नमस्ते भारत! हम आज बात करेंगे",
        "// A Javascript function to calculate the average of an array\nfunction getAverage(arr) {",
        "SELECT users.name, COUNT(orders.id) FROM users"
    ]
    
    print("Welcome to MIR-LM generation playground!")
    print(f"Running on device: {device}")
    
    if gen_args.interactive:
        print("\nInteractive mode active. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                prompt = input("\nEnter prompt > ")
                if prompt.strip().lower() in ["exit", "quit"]:
                    break
                if not prompt.strip():
                    continue
                generate(model, prompt, tokenizer, max_gen_len=128, temperature=gen_args.temperature, repetition_penalty=gen_args.repetition_penalty, device=device)
            except KeyboardInterrupt:
                print()
                break
    else:
        # Generate from list or read user input
        for i, prompt in enumerate(prompt_list):
            print(f"\n--- Demo {i+1} ---")
            generate(model, prompt, tokenizer, max_gen_len=64, temperature=gen_args.temperature, repetition_penalty=gen_args.repetition_penalty, device=device)

if __name__ == "__main__":
    main()

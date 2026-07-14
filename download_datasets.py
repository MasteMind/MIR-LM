import os
import sys
import numpy as np
from tokenizers import Tokenizer
from datasets import load_dataset
import tqdm

def main():
    print("=== MIR-LM Large Dataset Downloader & Preprocessor ===")
    
    # 1. Load the trained BPE tokenizer
    tokenizer_path = "tokenizer/vocab.json"
    if not os.path.exists(tokenizer_path):
        print("Error: Tokenizer not found! Please train the tokenizer first.")
        sys.exit(1)
        
    print("Loading BPE tokenizer...")
    if os.path.exists("tokenizer/tokenizer.json"):
        tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
    else:
        from tokenizers import ByteLevelBPETokenizer
        tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
        
    bos_token_id = tokenizer.token_to_id("<s>")
    eos_token_id = tokenizer.token_to_id("</s>")
    
    # Target token counts: Total ~30M tokens for dev-scale pre-training
    # Code (50%): ~15M tokens
    # Indian Languages (25%): ~7.5M tokens
    # English Reasoning (25%): ~7.5M tokens
    target_code_tokens = 15_000_000
    target_indian_tokens = 7_500_000
    target_english_tokens = 7_500_000
    
    # Direct output binaries
    os.makedirs("data_bin", exist_ok=True)
    train_bin_path = "data_bin/large_train.bin"
    val_bin_path = "data_bin/large_val.bin"
    
    # We will buffer tokens and write them in chunks of uint16
    all_tokens = []
    
    # Helper to stream and tokenize
    def stream_and_tokenize(dataset_name, dataset_config, split, text_extractor, target_tokens, desc):
        print(f"\nStreaming '{desc}' from Hugging Face ({dataset_name})...")
        tokens_collected = 0
        pbar = tqdm.tqdm(total=target_tokens, unit="tokens")
        
        try:
            # Use streaming=True to load dataset on-the-fly without downloading the whole archive
            if dataset_config:
                ds = load_dataset(dataset_name, name=dataset_config, split=split, streaming=True, trust_remote_code=True)
            else:
                ds = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
                
            for row in ds:
                if tokens_collected >= target_tokens:
                    break
                text = text_extractor(row)
                if not text:
                    continue
                # Encode text
                encoded = tokenizer.encode(text)
                ids = [bos_token_id] + encoded.ids + [eos_token_id]
                
                all_tokens.extend(ids)
                tokens_collected += len(ids)
                pbar.update(len(ids))
                
        except Exception as e:
            print(f"Error streaming {dataset_name}: {e}")
            print("Continuing to next dataset subset...")
            
        pbar.close()
        print(f"Finished {desc}: Collected {tokens_collected:,} tokens.")

    # 1. Code Datasets (Target: 15M tokens total)
    # Python Code (10M tokens)
    stream_and_tokenize(
        dataset_name="Flytech/python-codes-25k",
        dataset_config=None,
        split="train",
        text_extractor=lambda row: row["text"] if "text" in row else None,
        target_tokens=10_000_000,
        desc="Python Codes 25k (Flytech)"
    )
    # Multi-language Code (5M tokens)
    stream_and_tokenize(
        dataset_name="ise-uiuc/Magicoder-OSS-Instruct-75K",
        dataset_config=None,
        split="train",
        text_extractor=lambda row: f"{row['problem']}\n{row['solution']}" if "problem" in row and "solution" in row else None,
        target_tokens=5_000_000,
        desc="Multi-Language Code (Magicoder)"
    )
    
    # 2. Indian Languages (Target: 7.5M tokens total)
    # Hindi (3.5M tokens)
    stream_and_tokenize(
        dataset_name="cfilt/iitb-english-hindi",
        dataset_config=None,
        split="train",
        text_extractor=lambda row: row["translation"]["hi"] if "translation" in row and "hi" in row["translation"] else None,
        target_tokens=3_500_000,
        desc="IITB English-Hindi (Hindi Text)"
    )
    # Tamil (2M tokens)
    stream_and_tokenize(
        dataset_name="csebuetnlp/xlsum",
        dataset_config="tamil",
        split="train",
        text_extractor=lambda row: row["text"] if "text" in row else None,
        target_tokens=2_000_000,
        desc="Tamil Prose (XL-Sum)"
    )
    # Telugu (2M tokens)
    stream_and_tokenize(
        dataset_name="csebuetnlp/xlsum",
        dataset_config="telugu",
        split="train",
        text_extractor=lambda row: row["text"] if "text" in row else None,
        target_tokens=2_000_000,
        desc="Telugu Prose (XL-Sum)"
    )
    
    # 3. English Reasoning: FineWeb-Edu (Target: 7.5M tokens)
    stream_and_tokenize(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
        split="train",
        text_extractor=lambda row: row["text"] if "text" in row else None,
        target_tokens=target_english_tokens,
        desc="FineWeb-Edu (English Reasoning)"
    )
    
    total_tokens = len(all_tokens)
    print(f"\nTotal tokens collected across all sources: {total_tokens:,}")
    
    if total_tokens == 0:
        print("Error: No tokens were collected. Please check network connection.")
        sys.exit(1)
        
    # Split into 90% train and 10% validation
    split_idx = int(total_tokens * 0.9)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]
    
    # Pack into sequence chunks of size 2048
    seq_len = 2048
    
    def pack_to_npy(tokens, name):
        arr = np.array(tokens, dtype=np.uint16)
        num_chunks = len(arr) // seq_len
        arr = arr[:num_chunks * seq_len]
        reshaped = arr.reshape(-1, seq_len)
        print(f"Packed {name} dataset shape: {reshaped.shape} ({len(arr):,} tokens)")
        return reshaped

    train_data = pack_to_npy(train_tokens, "train")
    val_data = pack_to_npy(val_tokens, "validation")
    
    # Save datasets
    print("\nSaving binary datasets to data_bin/...")
    np.save("data_bin/train.npy", train_data)
    np.save("data_bin/val.npy", val_data)
    
    # Save metadata
    with open("data_bin/metadata.txt", "w") as f:
        f.write(f"seq_len={seq_len}\n")
        f.write(f"vocab_size={tokenizer.get_vocab_size() if hasattr(tokenizer, 'get_vocab_size') else len(tokenizer.get_vocab())}\n")
        
    print("Dataset preparation complete! Ready for pre-training.")

if __name__ == "__main__":
    main()

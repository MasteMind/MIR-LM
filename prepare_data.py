import os
import numpy as np
from tokenizers import Tokenizer

def pack_tokens(token_ids, seq_len=2048):
    """
    Pack a flat list of token IDs into chunks of size `seq_len`.
    If the last chunk is incomplete, it is discarded or padded.
    """
    arr = np.array(token_ids, dtype=np.uint16)
    num_chunks = len(arr) // seq_len
    if num_chunks == 0:
        return np.empty((0, seq_len), dtype=np.uint16)
    # Truncate to exact multiple of seq_len
    arr = arr[:num_chunks * seq_len]
    return arr.reshape(-1, seq_len)

def main():
    print("=== MIR-LM Data Preparation ===")
    
    # 1. Load the trained tokenizer
    tokenizer_path = "tokenizer/vocab.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError("Tokenizer files not found in './tokenizer/'. Please run 'train_tokenizer.py' first!")
        
    print(f"Loading tokenizer from {tokenizer_path}...")
    # Load the trained BPE tokenizer using Hugging Face's tokenizers library
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json") if os.path.exists("tokenizer/tokenizer.json") else None
    
    if tokenizer is None:
        # Fallback to loading vocab and merges manually if tokenizer.json wasn't saved
        from tokenizers import ByteLevelBPETokenizer
        tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
        
    # Special token mapping
    eos_token_id = tokenizer.token_to_id("</s>")
    bos_token_id = tokenizer.token_to_id("<s>")
    
    # 2. Gather dataset files
    # In a production run, we would stream datasets from Hugging Face (e.g. 'codeparrot' and 'sangraha').
    # For our local pre-training setup, we will read text/code files from a local directory 'data_raw/'.
    # We will search for any .py, .txt, .js, .md files in the repository as well.
    raw_dir = "data_raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    # Create simple dummy text files in data_raw if they don't exist, to ensure the pipeline runs
    sample_files = {
        "code_sample.py": (
            "def calculate_fibonacci(n):\n"
            "    if n <= 0:\n"
            "        return []\n"
            "    elif n == 1:\n"
            "        return [0]\n"
            "    fib = [0, 1]\n"
            "    while len(fib) < n:\n"
            "        fib.append(fib[-1] + fib[-2])\n"
            "    return fib\n\n"
            "print(calculate_fibonacci(10))\n"
        ),
        "hindi_sample.txt": (
            "यह भारतीय भाषाओं के लिए एक स्थानीय भाषा मॉडल है।\n"
            "मशीन लर्निंग और कृत्रिम बुद्धिमत्ता भारत के विकास के लिए महत्वपूर्ण हैं।\n"
            "हम इस मॉडल का उपयोग कोडिंग और तार्किक सोच सिखाने के लिए करेंगे।\n"
        ),
        "tamil_sample.txt": (
            "இது தமிழ் மற்றும் பிற இந்திய மொழிகளுக்கான செயற்கை நுண்ணறிவு மாதிரி ஆகும்.\n"
            "கணினி நிரலாக்கம் மற்றும் தரவு அறிவியல் ஆகியவை முக்கிய துறைகள் ஆகும்.\n"
        )
    }
    
    for filename, content in sample_files.items():
        filepath = os.path.join(raw_dir, filename)
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
                
    # 3. Read and tokenize all files in data_raw
    print("Reading and tokenizing raw files...")
    all_tokens = []
    
    # Gather files
    files = []
    for root, _, filenames in os.walk(raw_dir):
        for f in filenames:
            files.append(os.path.join(root, f))
            
    # Also add the project's source code and docs files for training signal!
    import glob
    for ext in ["*.py", "*.md", "docs/*.md"]:
        for f in glob.glob(ext):
            if os.path.exists(f) and f not in files:
                files.append(f)
            
    print(f"Found {len(files)} files to process.")
    
    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            # Encode text
            encoded = tokenizer.encode(text)
            # Add BOS and EOS tokens
            tokens = [bos_token_id] + encoded.ids + [eos_token_id]
            all_tokens.extend(tokens)
            print(f"Processed {filepath}: {len(tokens)} tokens")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
    print(f"Total tokens collected: {len(all_tokens)}")
    
    if len(all_tokens) == 0:
        print("No tokens collected. Please add text files to 'data_raw/' and run again.")
        return
        
    # 4. Split into Train and Validation sets (90% train, 10% val)
    split_idx = int(len(all_tokens) * 0.90)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]
    
    # 5. Pack into 2048-token chunks
    seq_len = 2048
    # Since we have small data for tests, let's temporarily reduce chunk size to 256
    # if total tokens are small, so that we can have at least one training block.
    if len(train_tokens) < seq_len:
        print(f"Warning: Total train tokens ({len(train_tokens)}) is less than seq_len ({seq_len}).")
        seq_len = 256
        print(f"Temporarily using seq_len = {seq_len} to fit data.")
        
    train_chunks = pack_tokens(train_tokens, seq_len=seq_len)
    val_chunks = pack_tokens(val_tokens, seq_len=seq_len)
    
    print(f"Train data shape: {train_chunks.shape} ({train_chunks.size} tokens)")
    print(f"Val data shape: {val_chunks.shape} ({val_chunks.size} tokens)")
    
    # 6. Save as bin files
    os.makedirs("data_bin", exist_ok=True)
    np.save(os.path.join("data_bin", "train.npy"), train_chunks)
    np.save(os.path.join("data_bin", "val.npy"), val_chunks)
    
    # Also save metadata for training config
    with open(os.path.join("data_bin", "metadata.txt"), "w") as f:
        f.write(f"seq_len={seq_len}\n")
        f.write(f"vocab_size={tokenizer.get_vocab_size() if hasattr(tokenizer, 'get_vocab_size') else 32000}\n")
        
    print("Saved binary tokenized datasets to folder 'data_bin/'")
    print("Data preparation complete!")

if __name__ == "__main__":
    main()

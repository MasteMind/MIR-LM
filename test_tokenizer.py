import os
from tokenizers import Tokenizer

def main():
    tokenizer_path = "tokenizer/vocab.json"
    if not os.path.exists(tokenizer_path):
        print("Tokenizer files not found. Please train the tokenizer first.")
        return
        
    print("Loading BPE tokenizer...")
    # Try loading from full tokenizer.json config first, or fallback to manual vocab/merges loading
    if os.path.exists("tokenizer/tokenizer.json"):
        tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
    else:
        from tokenizers import ByteLevelBPETokenizer
        tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
        
    # Define test sentences representing code and multilingual Indian scripts
    test_cases = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "नमस्ते भारत! यह कोडिंग और एआई का भविष्य है।",
        "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "class MIRLM(nn.Module):\n\tdef __init__(self, args):\n\t\tsuper().__init__()"
    ]
    
    print("\n=== Tokenizer Verification ===")
    for i, text in enumerate(test_cases):
        print(f"\n--- Case {i+1} ---")
        print(f"Original String:\n{text}")
        
        # Encode
        encoded = tokenizer.encode(text)
        print(f"Number of tokens: {len(encoded.ids)}")
        print(f"Token IDs: {encoded.ids}")
        print(f"Subword tokens: {encoded.tokens}")
        
        # Decode (Round-trip verification)
        decoded = tokenizer.decode(encoded.ids)
        print(f"Decoded String:\n{decoded}")
        
        # Assert round-trip is identical
        # Strip trailing newlines or spacing changes if any, but it should match exactly
        assert text.strip() == decoded.strip(), f"Round-trip mismatch!\nExpected: {text}\nGot: {decoded}"
        print("✔ Round-trip verification successful!")

if __name__ == "__main__":
    main()

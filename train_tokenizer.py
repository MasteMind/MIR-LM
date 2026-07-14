import os
import glob

# ByteLevelBPETokenizer trains and saves vocab.json + merges.txt (the HF "slow" format).
# We also persist tokenizer.json (the HF "fast" format) at the end so callers that prefer
# Tokenizer.from_file(...) — generate.py, prepare_data.py, download_datasets.py — work too.
from tokenizers import ByteLevelBPETokenizer

# Same three sources download_datasets.py uses, targeted to roughly mirror the pre-training
# mix (~50% code / ~30% Indian / ~20% English, per docs/reasoning_design.md "30% Indian" goal).
# Note: download_datasets.py currently targets 25% Indian; that 25-vs-30 discrepancy is left
# as a documented decision rather than changed here unilaterally.
HF_CORPUS_SOURCES = [
    {
        "name": "Flytech/python-codes-25k",
        "config": None,
        "split": "train",
        "field": "text",
        "target_chars": 1_000_000,
        "desc": "Python code (Flytech)",
    },
    {
        "name": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "config": None,
        "split": "train",
        "field": "solution",
        "target_chars": 600_000,
        "desc": "Multi-language code (Magicoder)",
    },
    {
        "name": "cfilt/iitb-english-hindi",
        "config": None,
        "split": "train",
        "field": ("translation", "hi"),
        "target_chars": 600_000,
        "desc": "Hindi (IITB en-hi)",
    },
    {
        "name": "csebuetnlp/xlsum",
        "config": "tamil",
        "split": "train",
        "field": "text",
        "target_chars": 400_000,
        "desc": "Tamil prose (XL-Sum)",
    },
    {
        "name": "csebuetnlp/xlsum",
        "config": "telugu",
        "split": "train",
        "field": "text",
        "target_chars": 400_000,
        "desc": "Telugu prose (XL-Sum)",
    },
    {
        "name": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "field": "text",
        "target_chars": 400_000,
        "desc": "English reasoning (FineWeb-Edu)",
    },
]

VOCAB_SIZE = 32000
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


def _extract_field(row, field):
    """Resolve nested (a, b) tuples vs plain field names from a dataset row."""
    if isinstance(field, tuple):
        cur = row
        for key in field:
            cur = cur.get(key) if isinstance(cur, dict) else None
            if cur is None:
                return None
        return cur
    return row.get(field)


def build_corpus_from_hf(out_path):
    """
    Build a tokenizer-training corpus by streaming ~2M characters of real text from
    Hugging Face (code + Hindi + English), mirroring the pre-training distribution.

    Needs network + the `datasets` library. If either is unavailable, the caller falls
    back to the inline placeholder corpus in build_corpus() — so this script still runs
    offline, but the resulting tokenizer will be ~byte-level (no real merges).
    """
    from datasets import load_dataset  # lazy import; only needed on this path

    total_chars = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for src in HF_CORPUS_SOURCES:
            chars_collected = 0
            print(f"  streaming '{src['desc']}' ({src['name']})...", flush=True)
            try:
                if src["config"]:
                    ds = load_dataset(src["name"], name=src["config"], split=src["split"], streaming=True, trust_remote_code=True)
                else:
                    ds = load_dataset(src["name"], split=src["split"], streaming=True, trust_remote_code=True)
                for row in ds:
                    if chars_collected >= src["target_chars"]:
                        break
                    text = _extract_field(row, src["field"])
                    if not text:
                        continue
                    out_f.write(text + "\n")
                    chars_collected += len(text)
            except Exception as e:
                print(f"  WARNING: streaming {src['name']} failed: {e}")
                print(f"  Continuing with what was collected ({chars_collected} chars).")
            total_chars += chars_collected
            print(f"  -> {src['desc']}: {chars_collected:,} chars")
    print(f"Corpus written to {out_path} ({total_chars:,} chars total).")
    return out_path


def build_corpus(sample_dir="corpus_temp"):
    """
    Offline fallback: build a tiny inline corpus from hardcoded code + Indian-language
    samples plus the workspace's own *.py files. This is a placeholder — the resulting
    BPE merges will be ~byte-level. Prefer build_corpus_from_hf() when network + the
    `datasets` library are available (run this script on Ubuntu after setup_ubuntu.sh).
    """
    print("Building (offline placeholder) tokenizer training corpus...")
    os.makedirs(sample_dir, exist_ok=True)
    corpus_file = os.path.join(sample_dir, "tokenizer_corpus.txt")

    with open(corpus_file, "w", encoding="utf-8") as out_f:
        code_samples = [
            "def hello_world():\n\tprint('Hello World')\n\tfor i in range(10):\n\t\tprint(i)\n",
            "function add(a, b) {\n    return a + b;\n}\n\nconst mult = (x, y) => x * y;\n",
            "#include <iostream>\nusing namespace std;\nint main() {\n    cout << \"Hello\" << endl;\n    return 0;\n}\n",
            "SELECT id, name, COUNT(orders) FROM users JOIN orders ON users.id = orders.user_id GROUP BY id HAVING COUNT(orders) > 5;\n",
            "<!DOCTYPE html>\n<html>\n<head>\n    <title>MIR-LM</title>\n</head>\n<body>\n    <h1>First LLM of India</h1>\n</body>\n</html>\n",
        ]
        for sample in code_samples:
            out_f.write(sample + "\n")

        indian_samples = [
            "नमस्ते भारत! यह MIR-LM है, भारत का अपना स्थानीय भाषा मॉडल।",  # Hindi
            "வணக்கம் இந்தியா! இது MIR-LM, இந்தியாவின் முதல் உள்ளூர் மொழி மாதிரி.",  # Tamil
            "నమస్తే ఇండియా! ఇది MIR-LM, భారతదేశం యొక్క మొట్టమొదటి స్థానిక భాషా మోడల్.",  # Telugu
            "નમસ્તે ભારત! આ MIR-LM છે, ભારતનું પોતાનું સ્થાનિક ભાષા મોડેલ।",  # Gujarati
            "হ্যালো ভারত! এটি MIR-LM, ভারতের প্রথম নিজস্ব লোকাল ল্যাঙ্গুয়েজ মডেল।",  # Bengali
            "ഹലോ ഇന്ത്യ! ഇത് MIR-LM, ഇന്ത്യയുടെ ആദ്യത്തെ പ്രാദേശിക ഭാഷാ മോഡൽ ആണ്.",  # Malayalam
            "ನಮಸ್ತೆ ಭಾರತ! ಇದು MIR-LM, ಭಾರತದ ಮೊದಲ ಸ್ಥಳೀಯ ಭಾಷಾ ಮಾದರಿ.",  # Kannada
        ]
        for sample in indian_samples:
            out_f.write(sample + "\n")

        # Self-referential seed: include the workspace's own source for code tokens.
        for py_file in glob.glob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as in_f:
                    out_f.write(in_f.read() + "\n")
            except Exception as e:
                print(f"Skipping {py_file} for corpus due to: {e}")

    return corpus_file


def main():
    sample_dir = "corpus_temp"
    os.makedirs(sample_dir, exist_ok=True)
    corpus_path = os.path.join(sample_dir, "tokenizer_corpus.txt")

    # Prefer a real corpus streamed from Hugging Face; fall back to the inline
    # placeholder if network or the `datasets` library is unavailable.
    use_hf = True
    try:
        from datasets import load_dataset  # noqa: F401 — probe availability
    except ImportError:
        use_hf = False
        print("`datasets` library not found — falling back to the offline placeholder corpus.")
        print("For a real multilingual BPE: pip install datasets tqdm, then rerun this script.")

    if use_hf:
        try:
            print("Building tokenizer training corpus from Hugging Face...")
            build_corpus_from_hf(corpus_path)
        except Exception as e:
            print(f"HF corpus build failed ({e}); falling back to the offline placeholder corpus.")
            build_corpus(sample_dir)
    else:
        build_corpus(sample_dir)

    print("Initializing Byte-Level BPE Tokenizer...")
    tokenizer = ByteLevelBPETokenizer()

    print(f"Training tokenizer (vocab_size={VOCAB_SIZE}, min_frequency={MIN_FREQUENCY})...")
    tokenizer.train(
        files=[corpus_path],
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
    )

    # Save the HF "slow" format (vocab.json + merges.txt) — used by ByteLevelBPETokenizer.
    os.makedirs("tokenizer", exist_ok=True)
    tokenizer.save_model("tokenizer")
    print("Saved vocab.json + merges.txt to './tokenizer'")

    # Also save tokenizer.json (HF "fast" format) so Tokenizer.from_file(...) works —
    # generate.py / prepare_data.py / download_datasets.py all prefer that path.
    tokenizer._tokenizer.save("tokenizer/tokenizer.json")
    print("Saved tokenizer.json to './tokenizer'")

    print("\nTesting tokenizer on sample text:")
    sample_text = "def train_mirlm(epoch):\n    print('ट्रेनिंग शुरू')"
    output = tokenizer.encode(sample_text)
    print(f"Original Text:\n{sample_text}")
    print(f"Tokens: {output.tokens}")
    print(f"Token IDs: {output.ids}")


if __name__ == "__main__":
    main()

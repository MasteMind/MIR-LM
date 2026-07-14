# Walkthrough: MIR-LM Pipeline & Dataset Preparation Complete

We have successfully prepared the large-scale pre-training dataset and verified the entire coding, preprocessing, training, and inference pipeline on your Windows host utilizing PyTorch DirectML. The next phase is to perform the actual scale pre-training run on your native Ubuntu partition.

---

## 1. What was Accomplished & Resolved

### 📊 Real Pre-Training Dataset Generated (26.3 Million Tokens)
We implemented a streaming preprocessor (`download_datasets.py`) that successfully streamed, tokenized, and packed a balanced corpus directly into NumPy binary arrays (`train.npy`, `val.npy` — saved via `np.save` and loaded into RAM, **not** memory-mapped):
- **Python Code (50% target)**: **15.0M tokens** of Python coding instructions and outputs from `Flytech/python-codes-25k` (script-free stable dataset).
- **Indian Languages (25% target)**: **7.5M tokens** of Hindi translations from the `cfilt/iitb-english-hindi` corpus.
- **English Reasoning (25% target)**: **7.5M tokens** of high-quality educational web pages from `HuggingFaceFW/fineweb-edu`.
- The 15.0/7.5/7.5M figures are `download_datasets.py`'s **targets** (summing to 30M); actual collection stopped at **26.3M tokens** when streams hit their limits. The chunk counts below reflect the 26.3M actual total (12,850 × 2048 = 26.3M).
- **Total Chunks**: `11,565` training sequences and `1,285` validation sequences of context window size `2048`.

### 🖥️ Native Windows Conda Environment
- Set up a native Windows Miniconda environment (`mir-lm-win`) with Python 3.10.
- Installed PyTorch CPU base alongside Microsoft DirectML (`torch-directml`), `datasets`, `tokenizers`, `transformers`, and `numpy`.
- Verified that DirectML successfully detects the RX 9070 XT as device `privateuseone:0` and performs matrix multiplications.

### 🐛 Major Code & Math Fixes Resolved
- **Dataloader Clamping**: Fixed a dataloader StopIteration infinite loop crash when validation dataset size was smaller than micro-batch size.
- **DirectML NaN Mask Fix**: Modified `model.py` to create the causal mask with `-1e9` and explicitly specify `dtype=torch.float32`. This fixed DirectML compiler NaN propagation and type promotion errors.
- **Deserialization Type Mismatch**: Patched `generate.py` to load checkpoints on `"cpu"` first to bypass a DirectML serialization compatibility bug.

---

## 2. Pre-Training Run Specifications

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Model Size** | **~124.7 Million** | Computed from `ModelArgs` (untied embeddings + output head, 12× GQA+SwiGLU layers, `bias=False`). Run `python model.py` to print the exact `sum(p.numel())`. The **135M** figure in `decisions.md` is aspirational — reaching it would require a larger `d_ff`/vocab/layers. A prior draft of this doc cited **78.89M**, which does not reconcile with the current architecture and appears to be from an earlier config. |
| **Context Window** | **2,048 tokens** | Context window size |
| **Train Dataset** | **23.68M tokens** | Precompiled NumPy binary (`train.npy`) |
| **Val Dataset** | **2.63M tokens** | Precompiled NumPy binary (`val.npy`) |
| **Batch Size** | **4** (micro-batch) | Micro-batch size per step |
| **Grad Accumulation**| **64** steps | Yields a global batch size of 256 sequences (524k tokens) |
| **Warmup Steps** | **2000** (default in `train.py`) | Matches `architecture.md` §3 (intended for the 5–10B token run). **For this 26M-token dataset (~276 steps @ 3 epochs), pass `--warmup_steps 50`** — 2000 would never finish warmup. `train.py` warns if `warmup_steps >= total_steps`. |
| **Base LR** | **3e-4** | Peak learning rate (AdamW optimizer) |

---

## 3. How to Pre-Train on Native Ubuntu (Radeon ROCm)

This is the canonical runbook. All paths are relative to the repo root — `cd` into the repo before running. (The WSL2 alternative lives in `setup_wsl.sh`; the DirectML-dev path on Windows is what produced the §1 verification.)

### Step 0: GPU driver + ROCm (manual prerequisites, not automated)
Before any script: install the `amdgpu` kernel driver and ROCm on your Ubuntu partition. The RX 9070 XT is RDNA4 (`gfx1201`); **ROCm 6.0 wheels may predate full RDNA4 support** — if `torch.cuda.is_available()` returns False after Step 1, use a newer ROCm wheel index (6.2+) or a system ROCm build. Verify the GPU is reachable before installing Python deps:
```bash
rocminfo | grep gfx      # should list gfx1201
rocm-smi                # should show the 9070 XT
```

### Step 1: Environment setup
`setup_ubuntu.sh` installs miniconda + a `mir-lm` conda env + `torch tokenizers transformers numpy datasets tqdm` from the ROCm wheel index (note: the old walkthrough missed `numpy datasets tqdm`, which `prepare_data.py`/`download_datasets.py`/`train_tokenizer.py` need).
```bash
bash setup_ubuntu.sh
conda activate mir-lm
python test_gpu.py      # expect 'CUDA (ROCm) available: True' + a GPU matmul
python -m py_compile model.py train.py generate.py prepare_data.py download_datasets.py train_tokenizer.py test_tokenizer.py test_gpu.py test_gpu_win.py   # syntax sanity
```

### Step 2: Tokenizer + dataset (needs network)
The tokenizer saved during the DirectML phase was trained on the inline placeholder corpus (~byte-level merges). **Retrain it on a real multilingual corpus** before pre-training, then stream the dataset:
```bash
python train_tokenizer.py    # streams ~2M chars (code+Hindi+English) from HF → tokenizer/{vocab.json,merges.txt,tokenizer.json}
python download_datasets.py  # streams ~30M tokens → data_bin/{train,val}.npy + metadata.txt
```
> If `data_bin/` from the DirectML phase is already present and you'd rather not re-stream, `download_datasets.py` can be skipped — but **retrain the tokenizer regardless**: the existing vocab's merges don't match the real corpus, and the scripts all prefer `tokenizer/tokenizer.json` (now saved by `train_tokenizer.py`).

### Step 3: Pre-train
`train.py` auto-detects ROCm as `"cuda"` and enables BF16/FP16 AMP. The `--warmup_steps` default is **2000** (matches `architecture.md` §3). **For this 26M-token dataset, pass `--warmup_steps 50`** (~276 total steps @ 3 epochs; 2000 would never finish warmup). `train.py` warns if `warmup_steps >= total_steps`.
```bash
python train.py --epochs 3 --batch_size 4 --grad_accum 64 --warmup_steps 50 --eval_interval 50 --save_interval 100
```
Resume a crashed/OOM'd run from a checkpoint (restores model + optimizer + scaler + step + epoch via `map_location="cpu"`):
```bash
python train.py --resume checkpoints/mirlm_step_100.pt --warmup_steps 50
```

### Step 4: Generate
```bash
python generate.py                                            # auto-picks the latest checkpoint in checkpoints/
python generate.py --checkpoint checkpoints/mirlm_step_100.pt # explicit checkpoint
```
**Verify any DirectML-saved checkpoint on ROCm before trusting it:** checkpoints saved from the Windows DirectML phase carry `privateuseone` device tags. `generate.py` loads via `map_location="cpu"` then `.to(device)`, which *should* remap them — but load one and confirm before relying on it. For a clean ROCm-only run, train from scratch (Step 3) and this caveat does not apply.

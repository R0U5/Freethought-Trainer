# Freethought Trainer

A single-file, **offline-first** LoRA fine-tuning script for Hugging Face causal language models — with optional **multimodal (image+text)** training via **Phi-3-Vision**.

Built for Windows with an emphasis on:
- minimal setup
- safe checkpointing / resumability
- automatic dataset handling
- dynamic token-length sizing + batching
- statistically gated early stopping
- reliable LoRA merge + save

---

## What this script does

Given a dataset (usually a `.parquet` file), Freethought Trainer will:

1. Load + shuffle the dataset
2. Auto-detect training schema (SFT / causal / chat / multimodal)
3. Auto-map dataset columns to roles (question/chosen/image/etc.)
4. Clean + synthesize text prompts
5. Measure token length distribution (p50/p95/p99) and set `max_length` automatically
6. Tokenize and build a length-aware training dataset
7. Train with LoRA (resume-safe)
8. Cleanup checkpoints + keep `checkpoint-last` and `last_good_ckpt`
9. Merge LoRA into the base model and save to `merged/`
10. Write `merged/success.txt` only on clean completion (prevents bad resumes)

---

## Requirements

### Hardware / OS
- **NVIDIA CUDA GPU is required** (the script loads models with `.to("cuda")`)
- Windows 10/11 is the primary target (includes a GUI file picker)

### Python + packages
Python 3.10 recommended.

Core deps:
- `torch` (CUDA build)
- `transformers`
- `datasets`
- `peft`
- `accelerate`
- `pandas`, `pyarrow`, `numpy`
- `Pillow`

Optional:
- `emoji` (for emoji stripping in the cleaning pipeline)

---

## Quick start

1) Put your base model(s) on disk (HF format folders, not GGUF):
- Example defaults inside the script:
  - `D:/HF_Models/phi-2`
  - `D:/HF_Models/phi-3-vision`

2) Edit the constants near the top of `Freethought Trainer.py` if you want:
- `HF_ROOT`, `TMP_ROOT`
- `BASE_MODEL`, `BASE_MODEL_IMAGE`
- `OUTPUT_DIR`

3) Run:

```powershell
# Basic run (select a local .parquet via GUI picker on Windows)
python ".\Freethought Trainer.py"
```

---

## CLI flags

```text
--force        Disable early stopping (run the full configured training)
--epoch N      Set epochs (early stop still applies unless --force or --steps is used)
--steps N      Train for an exact number of steps (disables early stopping)
--latex        Preserve LaTeX and scientific equations (cleaning mode)
--code         Preserve code formatting/indentation (cleaning mode)
--image        Enable multimodal training (Phi-3-Vision: image + text)
--hf_dataset   Load a dataset by HF Hub ID (e.g. username/dataset-name)
--base_model   Override base model path (text or image base depending on mode)
```

Examples:

```powershell
# Preserve LaTeX
python ".\Freethought Trainer.py" --latex

# Preserve code formatting
python ".\Freethought Trainer.py" --code

# Both
python ".\Freethought Trainer.py" --latex --code

# Fixed number of steps (no early stop)
python ".\Freethought Trainer.py" --steps 500

# Disable early stopping explicitly
python ".\Freethought Trainer.py" --force --epoch 2

# Multimodal (Phi-3-Vision)
python ".\Freethought Trainer.py" --image

# HF dataset ID (will attempt Hub load if not cached)
python ".\Freethought Trainer.py" --hf_dataset "username/dataset-name"

# Override base model (example)
python ".\Freethought Trainer.py" --base_model "D:\HF_Models\phi-3"
```

---

## Offline-first behavior (important)

The script sets offline-related Hugging Face env vars at startup:
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `HF_DATASETS_OFFLINE=1`

**Models/tokenizers/processors are loaded with `local_files_only=True`.**

If you use `--hf_dataset` (or enter a Hub ID manually), the script temporarily lifts dataset-offline flags during dataset load. That means:
- If the dataset is already cached, you stay effectively offline.
- If it isn’t cached, it will try to fetch it.

---

## Dataset support (schemas + auto column mapping)

Input is typically a **`.parquet`** file.

The trainer inspects column names (case-insensitive) and maps them via alias tables.

### Supported training modes
- **SFT (`sft`)**: instruction/question + chosen answer → prompt text
- **Causal (`causal`)**: raw text continuation (single text column)
- **Chat (`chat`)**: message-style column (detected by `messages`)
- **Multimodal (`multimodal`)**: image + chosen (optional question)

### Common column aliases (high level)

**Question/instruction aliases** include:
`instruction`, `question`, `prompt`, `input`, `query`, `context`, `problem`, `document`

**Chosen/answer aliases** include:
`response`, `chosen`, `answer`, `completion`, `output`, `solution`, `messages`, `summary`

**Image aliases** include:
`image`, `img`, `photo`, `image_path`, `picture`

The script prints a “Mapped Roles” summary at runtime so you can confirm what it picked.

---

## Multimodal mode (Phi-3-Vision)

Enable with `--image`.

Expected dataset (parquet or HF dataset) should include:
- an image field (bytes/dict/path, depending on dataset)
- a chosen/answer field
- optional question/instruction field

The trainer builds a Phi-3-Vision style prompt internally:

```text
<|user|>
<|image_1|>
{text}
<|end|>
<|assistant|>
```

Labels are masked so the model is supervised primarily on the assistant answer portion.

---

## Model profiles (automatic tuning per base model)

Profiles are selected by substring match on the **base model folder name**.

Included profiles:
- `phi-2`
- `phi-3`
- `phi-3-vision`

Each profile controls:
- attention backend (`sdpa` vs `eager`)
- tokenizer behavior (fast vs slow)
- special-token injection (Phi-2)
- LoRA target modules + rank/alpha
- grad accumulation + learning rate
- safe serialization behavior
- input-gradient forcing for some remote-code models

If no profile matches, a sane default profile is used.

---

## Checkpoints, resume, and safety

### Output layout

```text
OUTPUT_DIR/
├─ checkpoint-XXXX/          # rolling checkpoints
├─ checkpoint-last/          # most recent checkpoint (symlink or copy)
├─ last_good_ckpt/           # safety backup of the most recent checkpoint
└─ merged/
   ├─ config.json
   ├─ tokenizer*.json
   ├─ model*.safetensors (or shards)
   └─ success.txt            # only written on clean completion
```

### Resume rules
- If `merged/success.txt` exists: the next run **uses `merged/` as the base model** (continual training chain).
- If `merged/` exists but `success.txt` does **not**: the script **refuses to run** (prevents silent overwrite of an incomplete/unknown state).
- If `checkpoint-last/` exists: LoRA resumes from it (unless base mismatch is detected).

### Training chain
A `training_chain.txt` file (stored next to the script) tracks dataset filenames used.
When a new dataset name is logged, older checkpoints are purged to avoid cross-contamination.

---

## Troubleshooting

### “It’s training but feels like CPU”
Run this quick check:

```powershell
python -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

If CUDA is false, install a CUDA-enabled PyTorch build.

### Multimodal dataset OOM / high RAM usage
Multimodal path avoids converting the dataset to pandas to prevent pulling all image bytes into memory at once. If you still hit memory issues:
- reduce dataset size for a test run
- ensure images are reasonably sized
- confirm you’re not accidentally duplicating image columns

---

## License / notes
Non Commercial Open Software License — Attribution & Copyleft (NC OSL‑A 1.0)

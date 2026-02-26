# Freethought AI Trainer

A single-file, offline-first LoRA fine-tuning script for causal language models. Built for Windows with a focus on robustness, intelligent early stopping, and automatic dataset handling — designed to produce good training runs with minimal manual tuning.

---

## Overview

Freethought AI Trainer wraps HuggingFace `transformers` + `peft` into a self-contained script that handles the entire fine-tuning pipeline: dataset ingestion, cleaning, tokenization, LoRA training, and final model merging. It is designed to run **fully offline** against locally cached models and datasets, with no internet connection required after initial setup.

The script is opinionated by design. Rather than exposing every hyperparameter as an argument, it makes intelligent decisions automatically — sequence length, learning rate schedule, early stopping thresholds, and warmup steps are all derived from your data at runtime.

---

## Key Features

### Fully Offline Operation
All HuggingFace environment variables are configured at startup to force offline mode (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`). The script will never attempt a network call during a training run.

### Automatic Dataset Schema Detection
The script inspects your `.parquet` file's column names and automatically detects which training mode to use and which columns map to which roles — no configuration needed.

**Supported training modes:**

| Mode | Description | Detected by |
|------|-------------|-------------|
| `sft` | Supervised fine-tuning (instruction + response) | Presence of columns like `output`, `answer`, `chosen`, `response`, `completion`, etc. |
| `causal` | Pure causal LM (raw text continuation) | Presence of columns like `prompt`, `text` |
| `chat` | Chat/messages format | Presence of a `messages` column |

Column name matching is alias-based and case-insensitive. For example, a `chosen` role will be satisfied by any of: `response`, `chosen`, `answer`, `answers`, `completion`, `output`, `solution`, `expected_answer`, `long_answer`, `messages`, `summary`.

### Adaptive Sequence Length
Rather than using a fixed `MAX_LENGTH`, the script tokenizes the entire dataset, measures the actual length distribution (p50/p95/p99), and sets `max_length` to the **p95 value** clamped between 128 and 1024. This avoids both wasteful over-padding and silent truncation of the majority of your data.

```
Length Statistics - p50:280, p95:715, p99:1098 / Max Length:715
```

### Intelligent Data Cleaning Pipeline
The `clean_string()` function applies an extensive regex-based cleaning pipeline before tokenization. It is mode-aware — behavior changes depending on whether you are training on general text, math, LaTeX, or code.

**What gets stripped in default (`math`) mode:**
- LaTeX environments (`\begin{...}...\end{...}`)
- Inline math wrappers (`\mathrm{}`, `\text{}`, `\operatorname{}`)
- Inline `$...$` expressions (preserving bare numeric currency like `$4.99`)
- TeX delimiters `\(...\)` and `\[...\]`
- HTML tags
- Markdown headers
- Chat format markers (`[INST]`, `[SYS]`, `[USER]`, `[ASSISTANT]`, `<|...|>`)
- Template placeholders (`{{...}}`)
- Junk words (Subscribe, Click here, Advertisement, etc.)
- E-commerce tags (SKU, ASIN, MSRP, Price)
- Zero-width and bidirectional Unicode control characters
- Media placeholders (`[image]`, `[video]`, etc.)
- Emoji (if the `emoji` package is installed)
- Non-ASCII characters
- A curated blocklist of known dataset corruption tokens (KSP field names, mojibake sequences, stack traces, null pointers, placeholder strings, etc.)
- `\boxed{...}` unwrapping — extracts the inner content rather than discarding it
- Math symbol normalisation: `\cdot` → `*`, `\times` → `*`, `\le` → `<=`, `\ge` → `>=`, `\neq` → `!=`, etc.

**Cleaning modes:**

| Flag | Mode | Behaviour |
|------|------|-----------|
| *(default)* | `math` | Strips LaTeX, normalises math symbols, removes non-ASCII |
| `--latex` | `latex` | Preserves LaTeX and scientific notation |
| `--code` | `code` | Preserves code formatting and indentation |
| `--latex --code` | `latex+code` | Preserves both |

### Smart Early Stopping (`EarlyStopByLoss`)
The centrepiece of the script. A custom `TrainerCallback` that monitors training loss using a multi-stage statistical decision system to stop training at the right moment — not too early, not too late.

**How it works:**

1. **Exposure floor gate** — Early stopping cannot trigger until at least 18% of the estimated total steps have been completed. This prevents premature stopping during the noisy early phase.

2. **Warmup gate** — No stopping decisions are made during the LR warmup period.

3. **LR quality gate** — Stopping is only evaluated once the learning rate has declined to ≤ 60% of its peak value. This ensures the model has had time to make meaningful progress before any plateau judgement.

4. **EMA smoothing** — Raw per-step loss is smoothed using an Exponential Moving Average (β=0.90) to filter out step-to-step noise. All improvement and plateau decisions operate on EMA loss, not raw loss.

5. **Robust sigma (MAD)** — Variability in the EMA series is measured using Median Absolute Deviation (MAD) rather than standard deviation. MAD is resistant to outliers and gives a more stable floor: `σ ≈ 1.4826 × MAD`.

6. **Improvement test** — A step counts as "improved" if either:
   - The absolute EMA drop ≥ `min_abs_improve` (0.04), OR
   - The EMA drop in robust sigma units ≥ `min_sigma_improve` (0.50)

7. **Plateau detection** — A plateau is declared when all of the following are true:
   - Steps since last improvement ≥ `patience` (8)
   - |slope of recent EMA| ≤ `slope_thresh` (0.010)
   - σ ≤ `max(0.5 × std_floor, 0.05)`

8. **Worsening detection** — Training is stopped immediately if the EMA slope ≥ `slope_thresh`, meaning loss is actively trending upward.

9. **Hard cap** — A logarithmically-scaled hard cap prevents training from ever exceeding a fraction of total estimated steps (65%–95% depending on run size), regardless of loss behaviour.

10. **Cooldown** — After each new best EMA, a 3-step cooldown prevents the plateau counter from penalising normal post-improvement stabilisation.

### Dynamic LR Scheduler Selection
The scheduler is chosen automatically based on dataset size, epoch count, and early stopping pressure:

| Condition | Scheduler |
|-----------|-----------|
| Small dataset or single epoch | `linear` |
| High early-stop pressure | `linear` |
| Medium dataset, multi-epoch, aggressive early stop | `cosine_with_restarts` |
| Everything else | `cosine` |

### Dynamic Warmup
Warmup steps are computed as `max(75, 1% of total_steps)`, scaling naturally with run length.

### Training Chain & Checkpoint Safety
- Every dataset used is logged by filename to `training_chain.txt`. When a new dataset is detected (i.e., not previously seen), all old checkpoints are automatically purged to prevent cross-contamination between training runs.
- The most recent checkpoint is symlinked to `checkpoint-last` for reliable resume detection.
- A `last_good_ckpt` backup copy is maintained separately as a safety net.
- The merged model directory is protected: if `merged/` exists but `merged/success.txt` does not, the script refuses to run and raises an error, preventing silent overwrites of a potentially valid but incompletely-flagged previous merge.

### Iterative Training / Continual Learning
The script automatically detects whether a previously merged model exists (`merged/success.txt`). If it does, that merged model is used as the base for the next LoRA run instead of the original base model. This allows you to chain multiple training datasets sequentially, each building on the last.

### Safe Keyboard Interrupt Handling
If you press `Ctrl+C` during training, the script catches the interrupt, saves the current model state immediately, then continues through the checkpoint cleanup and merge pipeline rather than exiting dirty.

### Dynamic Padding Collator
The `DynamicCausalCollator` pads each batch only to the length of its longest sequence (aligned to a multiple of 8 for tensor core efficiency), rather than padding everything to `MAX_LENGTH`. This significantly reduces wasted compute on short-sequence batches.

### VRAM Management
A `roc()` ("Release Old CUDA") utility function is called after dataset preparation. It runs `gc.collect()`, `torch.cuda.empty_cache()`, and `torch.cuda.ipc_collect()` with full synchronisation before and after, then reports exactly how much allocated memory, reserved memory, and driver-level memory was freed.

---

## Requirements

```
Python        3.10
torch         2.1.2+cu121
transformers  4.41.1
peft          0.11.1
accelerate    0.30.1
datasets      2.19.0
tokenizers    0.19.1
pandas        2.2.3
pyarrow       20.0.0
numpy         1.26.4
safetensors   0.5.3
emoji         2.11.0   (optional, for emoji stripping)
```

> **Note:** This stack is intentionally pinned. The `auto_gptq` dependency in the environment is compiled against torch 2.1.2+cu121 specifically. Upgrading PyTorch will break it.

---

## Configuration

Edit the constants near the top of the script before running:

```python
BASE_MODEL   = "C:/Your/Path"           # Path to your base model
OUTPUT_DIR   = "C:/Your/Path"  # Where checkpoints and merged model are saved
HF_ROOT      = "C:\HF_Cache"                 # HuggingFace cache root
TMP_ROOT     = "C:\HF_Temp"                  # Temp directory

BATCH_SIZE   = 1      # Per-device batch size
GRAD_ACCUM   = 12     # Gradient accumulation steps (effective batch = 12)
SAVE_STEPS   = 50     # Save a checkpoint every N steps
SAVE_LIMITS  = 2      # Maximum number of checkpoints to keep
LOG_STEPS    = 1      # Log loss every N steps
```

---

## Usage

```bash
# Basic run (auto-detects everything, math cleaning mode)
python AI_Trainer.py

# Preserve LaTeX in dataset
python AI_Trainer.py --latex

# Preserve code formatting
python AI_Trainer.py --code

# Both
python AI_Trainer.py --latex --code

# Train for a specific number of epochs (disables early stopping)
python AI_Trainer.py --epoch 3

# Train for a fixed number of steps
python AI_Trainer.py --steps 500

# Force full epoch completion (disable early stopping, use --epoch count)
python AI_Trainer.py --force --epoch 2
```

On Windows, a file picker dialog will open automatically to select your `.parquet` dataset. On other platforms, you will be prompted to enter the path manually.

---

## Training Pipeline (Step by Step)

```
1. Select .parquet dataset (GUI picker on Windows)
2. Load dataset → shuffle with random seed
3. Detect training mode (sft / causal / chat) from column names
4. Map column roles (question, chosen, rejected) via alias table
5. Resolve model path (merged model if available, else base model)
6. Load tokenizer → add special tokens (pad, bos, eos)
7. Build prompt text per row via synthesize_prompt_dataset()
8. Measure token length distribution → set max_length = p95 (capped 128–1024)
9. Tokenize all rows with dynamic max_length
10. Cast dataset columns to int64/int32, set torch format
11. Release VRAM (roc())
12. Log dataset name → purge checkpoints if new dataset
13. Compute estimated steps, warmup, early stop cap
14. Select LR scheduler dynamically
15. Load base model in bfloat16 → wrap with LoRA (r=16, alpha=32)
16. Resize embeddings if new tokens were added
17. Train with DynamicCausalCollator + EarlyStopByLoss callback
18. On completion: merge LoRA weights into base model
19. Save merged model via save_pretrained()
20. Write success.txt flag
```

---

## LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (`r`) | 16 |
| Alpha | 32 |
| Target modules | `q_proj`, `k_proj`, `v_proj` |
| Dropout | 0.1 |
| Bias | none |
| Task type | `CAUSAL_LM` |

Trainable parameter count is printed at startup (typically ~0.28% of total parameters for phi-2).

---

## Output Structure

```
OUTPUT_DIR/
├── checkpoint-NNNN/        # Rolling checkpoints (max 2 kept)
├── checkpoint-last/        # Symlink → most recent checkpoint
├── last_good_ckpt/         # Backup copy of most recent checkpoint
└── merged/
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── model.safetensors   (or sharded)
    └── success.txt         ← presence of this file = safe to use/continue from
```

---

## Dataset Format

Input must be a `.parquet` file. Column names are matched case-insensitively against the alias tables — no renaming required for most standard HuggingFace datasets.

**SFT example (instruction + response):**
```
| instruction              | output                  |
|--------------------------|-------------------------|
| Explain photosynthesis.  | Photosynthesis is ...   |
```

**Causal LM example:**
```
| text                                      |
|-------------------------------------------|
| The quick brown fox jumps over the ...    |
```

---

## Training Chain (Continual Learning)

The file `training_chain.txt` (written next to the script) tracks every dataset filename that has been trained on. When you run the script with a new dataset:

1. The new dataset name is appended to `training_chain.txt`
2. All existing checkpoints are purged (clean slate for the new run)
3. If `merged/success.txt` exists, the merged model from the previous run is used as the base

This allows sequential training across many datasets without manually managing model paths:

```
Run 1: base_model  + dataset_A  → merged_v1
Run 2: merged_v1   + dataset_B  → merged_v2
Run 3: merged_v2   + dataset_C  → merged_v3
```

---

## Tested Environment

- **OS:** Windows 10/11
- **GPU:** NVIDIA (CUDA 12.1)
- **Python:** 3.10
- **CUDA toolkit:** 12.1

> The script runs on CPU if no GPU is detected, but training will be impractically slow for any non-trivial dataset.

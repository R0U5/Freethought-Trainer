# Freethought PoC Training Summary & Tester Guide

> **What is this?**  
> A small, proof-of-concept LoRA fine-tune of **Microsoft Phi-2 (2.7B)** built with the **Freethought** training pipeline. This README explains **how it was trained** and **how to test it**.

## TL;DR (for testers)

- **Base model:** Microsoft **Phi-2 (2.7B)**
- **Method:** Parameter-efficient fine-tuning (**LoRA**)
- **Stages:**
  1) **SFT** (instruction-tuning) → next-token loss  
  2) **DPO** (optional pass) → preference alignment using chosen/rejected pairs
- **Objective:** Improve instruction-following, reasoning scaffolding, and general helpfulness on short/medium prompts.
- **Intended use:** Local/offline experiments, reproducible demos, and benchmarking **not** production.

---

## Training Recipe (What was actually done)

### 1) Base & Adapters
- **Base:** `microsoft/phi-2` (no architecture changes)
- **Adapters:** **LoRA** applied to attention/projection modules (rank/alpha typical for small-GPU runs; exact values not critical for testing)
- **Precision:** Mixed precision on GPU (auto/bfloat16/float16 depending on device)

### 2) Supervised Fine-Tuning (SFT)
- **Goal:** Teach the model to follow instructions and format answers cleanly.
- **Data (high-level):** Public, permissively-licensed instruction/answer samples; light cleaning (schema-first prompts, deduplication, short/medium sequence focus).
- **Loss:** Standard next-token cross-entropy on the answer portion; prompts masked as needed.

### 3) Direct Preference Optimization (DPO) - optional alignment pass
- **Goal:** Prefer better responses over weaker ones using **(chosen, rejected)** pairs.
- **Effect:** Nudges style toward concise, helpful answers without over-length verbosity.

### 4) Overfit Guards & Early Stop
- **Dynamic early stopping** (variance-aware, normalized loss-drop criteria), with a **minimum exposure floor** so the model sees enough of the data before stopping.
- **Shuffle/restarts** to reduce order effects; validation on held-out prompts for sanity checks.

---

## Inference & Testing

> You can evaluate either **(a)** base + LoRA adapters or **(b)** a merged checkpoint (if provided).

### A) Load base + LoRA (PEFT)
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = "microsoft/phi-2"
adapters = "./lora"  # path to LoRA folder in this repo or release

tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapters)
model.eval()

prompt = "Explain what WebRCON is and show one safe example command."
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, temperature=0.2, top_p=0.9)
print(tok.decode(out[0], skip_special_tokens=True))
```

### B) If a merged model is available
Point to the merged folder instead of using PEFT.

### Recommended test settings
- `max_new_tokens: 128–756`
- `temperature: 0.2–0.7`
- `top_p: 0.9`
- **Stop tokens:** newline(s) if you need crisp endings
- **Prompting style:** Plain natural language instructions (no special tags required)

### Quick sanity prompts
- “List three U.S. Presidents in Chronological Order.”  
- “Summarize what Direct Preference Optimization (DPO) does in one paragraph.”  
- “Write a small bash snippet that tails a log file and highlights lines containing ‘ERROR’.”

---

## What this model is **good for** (POC scope)

- Short to medium **instruction-following**
- Clear, **concise** answers
- Light **reasoning** and step-by-step explanations

## Known Limitations

- Not safety-tuned for high-risk domains; may hallucinate facts under pressure.
- Long-context tasks and tool-use are outside scope.
- DPO pass is **small-scale**; preferences may not generalize broadly.
- **Do not** rely on it for medical, legal, or financial advice.

---

## Repro Notes (so you can replicate)

- **Optimizer/Schedule:** Standard AdamW with linear/cosine decay (typical small-GPU defaults)
- **Batching:** Small per-device batch with gradient accumulation (to fit 12 GB GPUs)
- **Sequence lengths:** Favor short/medium sequences to maximize useful tokens per step
- **Tokenization:** Stock Phi-2 tokenizer; basic prompt/answer masking
- **Determinism:** Fixed seed for headline runs; minor non-determinism from GPU kernels possible

> Exact hyperparameters vary by run; this POC emphasizes **method** (LoRA → SFT → optional DPO) and **guardrails** (dynamic early stop) over chasing state-of-the-art metrics.

---

## Data & Licensing

- **Data:** Mix of permissively-licensed public instruction data and small curated samples; filtered/deduped.
- **This repo (code & weight deltas):** **NC OSL‑A 1.0 License**
- **Base model (Phi-2):** retains its own license; comply with upstream terms.
- **Any datasets:** keep their original licenses/attribution.

---

## Responsible Use

This project is for **research & evaluation**. Do not deploy for decisions impacting health, safety, finance, or law without rigorous, domain-specific evaluation and safeguards!


*NC OSL‑A 1.0 © William Furtado. See `LICENSE`.*

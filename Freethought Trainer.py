import os
import re
import gc
import sys
import json
import shutil
import math
import random
import argparse
import platform
import subprocess
import torch
import numpy as np
import pyarrow as pa
import pandas as pd
from pathlib import Path
import multiprocessing
from datasets import Dataset, load_dataset, Sequence, Value
import warnings
# Suppress deprecation warnings originating inside PyTorch/setuptools internals
# that we have no control over. Remove these once PyTorch is updated.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="TypedStorage is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="Special tokens have been added", category=UserWarning)

import logging as _logging
# Silence the HuggingFace datasets "Generating train split" progress line and
# the transformers "Special tokens have been added" warning — both bypass the
# standard Python warnings system and must be suppressed via the logging module.
_logging.getLogger("datasets").setLevel(_logging.ERROR)
_logging.getLogger("transformers.tokenization_utils_base").setLevel(_logging.ERROR)
_logging.getLogger("transformers").setLevel(_logging.ERROR)

# Optional emoji stripping — imported once at module level
try:
    import emoji as _emoji_mod
    _EMOJI_AVAILABLE = True
except ImportError:
    _emoji_mod = None
    _EMOJI_AVAILABLE = False


from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoProcessor, AutoModelForVision2Seq,
    TrainingArguments, Trainer, TrainerCallback
)
from peft import (
    LoraConfig, get_peft_model,
    PeftModel
)
from PIL import Image as PILImage
import io


# ── CLI output helpers ────────────────────────────────────────────────────────
def _banner(title: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")

def _ok(msg: str)   -> None: print(f"  ✔  {msg}")
def _info(msg: str) -> None: print(f"  ·  {msg}")
def _warn(msg: str) -> None: print(f"  ⚠  {msg}")
def _err(msg: str)  -> None: print(f"  ✖  {msg}")
# ──────────────────────────────────────────────────────────────────────────────

HF_ROOT = r"D:\HF_Cache"
TMP_ROOT = r"D:\HF_Temp"
os.environ["HF_HOME"] = HF_ROOT
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_ROOT, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_ROOT, "datasets")
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ["PYTORCH_HUB"] = HF_ROOT
os.environ["TMP"] = TMP_ROOT
os.environ["TEMP"] = TMP_ROOT
os.environ["TMPDIR"] = TMP_ROOT
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,garbage_collection_threshold:0.6"  # 128 was tuned for Phi-2; 512 reduces fragmentation for 4B+ models

MAX_LENGTH = 750
SAVE_STEPS = 50
SAVE_LIMITS = 2
LOG_STEPS = 1
BATCH_SIZE = 1  # always 1; effective batch size is controlled by GRAD_ACCUM in each profile
# Default paths — override via argparse or edit here.
# Kept as Windows-style defaults; cross-platform paths can be passed at runtime.
BASE_MODEL = "D:/HF_Models/phi-2"
BASE_MODEL_IMAGE = "D:/HF_Models/phi-3-vision"
OUTPUT_DIR = "D:/Trainer_Data/Merged_model"
MERGED_DIR = os.path.join(OUTPUT_DIR, "merged")
TRAINING_CHAIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_chain.txt")
MERGE_SUCCESS_FLAG = os.path.join(MERGED_DIR, "success.txt")
# MODEL_NAME is resolved at runtime inside main() / load_and_prepare_dataset()
# so that path validation errors surface there, not at import time.

# ── Model profiles ────────────────────────────────────────────────────────────
# Each entry maps a lowercase substring of the model path to a config dict.
# Resolution order: entries are checked in definition order; first match wins.
# Add a new model by inserting a new entry — no other code changes needed.
#
# Keys:
#   attn_implementation  str   PyTorch attention backend for text/causal models
#   trust_remote_code    bool  Required for models with custom modelling code
#   use_fast_tokenizer   bool  False only for SentencePiece-based vocabs (Phi-2)
#   inject_special_tokens bool  True only for models that ship without pad/bos/eos
#   resize_embeddings    bool  True only when inject_special_tokens adds new ids
#   lora_r               int   LoRA rank
#   lora_alpha           int   LoRA scaling factor (typically 2x rank)
#   lora_targets         list  Attention projection layers to attach adapters to
#   modules_to_save      list|None  Layers PEFT fully copies & owns (grad safety)
#   grad_accum           int   Gradient accumulation steps (effective batch size)
#   learning_rate        float Base learning rate
#   safe_serialization   bool  False for models with shared weight tensors
#   enable_input_grads   bool  True for custom-forward models that detach embeddings
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PROFILES = {
    "phi-2": {
        "attn_implementation":   "sdpa",
        "trust_remote_code":     False,
        "use_fast_tokenizer":    False,
        "inject_special_tokens": True,
        "resize_embeddings":     True,
        "lora_r":                16,
        "lora_alpha":            32,
        "lora_targets":          ["q_proj", "k_proj", "v_proj", "o_proj"],
        "modules_to_save":       None,
        "grad_accum":            16,
        "learning_rate":         2e-5,
        "safe_serialization":    True,
        "enable_input_grads":    False,
    },
    # phi-3-vision MUST appear before "phi-3" — first-match-wins.
    # Phi3VForCausalLM (the vision model class) does not support SDPA in
    # transformers 4.37.x even in text-only mode, so it must stay on eager.
    "phi-3-vision": {
        "attn_implementation":   "eager",
        "trust_remote_code":     True,
        "use_fast_tokenizer":    True,
        "inject_special_tokens": False,
        "resize_embeddings":     False,
        "lora_r":                16,
        "lora_alpha":            32,
        "lora_targets":          ["q_proj", "k_proj", "v_proj", "o_proj"],
        "modules_to_save":       ["lm_head"],
        "grad_accum":            8,
        "learning_rate":         1e-5,
        "safe_serialization":    False,
        "enable_input_grads":    True,
    },
    "phi-3": {
        "attn_implementation":   "sdpa",
        "trust_remote_code":     True,
        "use_fast_tokenizer":    True,
        "inject_special_tokens": False,
        "resize_embeddings":     False,
        "lora_r":                16,
        "lora_alpha":            32,
        "lora_targets":          ["q_proj", "k_proj", "v_proj", "o_proj"],
        "modules_to_save":       ["lm_head"],
        "grad_accum":            8,
        "learning_rate":         1e-5,
        "safe_serialization":    False,
        "enable_input_grads":    True,
    },
    # ── Add new models below this line ────────────────────────────────────────
    # "mistral": { ... },
    # "llama":   { ... },
    # "qwen":    { ... },
    # ─────────────────────────────────────────────────────────────────────────
}

# Fallback profile for models not listed above.
# Assumes a modern HF model with fast tokenizer and standard forward pass.
MODEL_PROFILE_DEFAULT = {
    "attn_implementation":   "sdpa",
    "trust_remote_code":     True,
    "use_fast_tokenizer":    True,
    "inject_special_tokens": False,
    "resize_embeddings":     False,
    "lora_r":                16,
    "lora_alpha":            32,
    "lora_targets":          ["q_proj", "k_proj", "v_proj", "o_proj"],
    "modules_to_save":       ["lm_head"],
    "grad_accum":            8,
    "learning_rate":         2e-5,
    "safe_serialization":    False,
    "enable_input_grads":    True,
}

def resolve_model_profile(model_path: str) -> dict:
    """Return the profile for the given model path, falling back to default."""
    name = os.path.basename(model_path).lower()
    for key, profile in MODEL_PROFILES.items():
        if key in name:
            return profile
    return MODEL_PROFILE_DEFAULT

def _resolve_model_name(merged_dir: str, merge_success_flag: str, base_model: str) -> str:
    """Return the correct model path to load from, raising clearly if state is ambiguous."""
    if os.path.exists(merge_success_flag):
        return merged_dir
    if os.path.exists(merged_dir):
        raise RuntimeError(
            f"[ERROR] Refusing to overwrite {merged_dir} — no success.txt found, but directory exists. "
            "You may be about to overwrite a previous model. Please verify or delete the folder manually."
        )
    return base_model


def roc(label: str = "") -> None:
    MiB = 1024 * 1024

    if not torch.cuda.is_available():
        freed_objs = gc.collect()
        if freed_objs > 0:
            _info(f"GC: {freed_objs} objs collected (CPU only)")
        return

    torch.cuda.synchronize()
    before_alloc = torch.cuda.memory_allocated()
    before_res   = torch.cuda.memory_reserved()
    free_before, _ = torch.cuda.mem_get_info()
    freed_objs = gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
    after_alloc = torch.cuda.memory_allocated()
    after_res   = torch.cuda.memory_reserved()
    free_after,  _ = torch.cuda.mem_get_info()
    freed_alloc    = (before_alloc - after_alloc) / MiB
    freed_reserved = (before_res   - after_res)   / MiB
    freed_driver   = (free_after   - free_before) / MiB

    def log_roc(label, freed_objs, freed_alloc, freed_reserved, freed_driver, eps=0.05):
        parts = []
        if freed_objs > 0:
            parts.append(f"{freed_objs} objs")
        def add(name, val):
            if not math.isclose(val, 0.0, abs_tol=eps):
                parts.append(f"{name} {val:+.1f} MiB")
        add("alloc", freed_alloc)
        add("reserved", freed_reserved)
        add("driver", freed_driver)
        if parts:
            tag = f" [{label}]" if label else ""
            _info(f"GC{tag}: " + "  |  ".join(parts))

    log_roc(label, freed_objs, freed_alloc, freed_reserved, freed_driver)



TRAINING_SCHEMAS = {
    "sft": {
        "required": ["chosen"],
        "optional": ["question"],
        "aliases": {
        
            "chosen": ["response", "chosen", "answer", "answers", "completion",
            "output", "solution",  "expected_answer", "long_answer", "messages","summary"],
            
            "question": ["instruction", "question", "prompt", "input",
            "query", "context", "problem","document"]
        }
    },
    "causal": {
        "required": ["chosen"],
        "optional": [],
        "aliases": {
            "chosen": ["prompt", "text"]
        }
    },
    "chat": {
        "required": ["question"],
        "optional": [],
        "aliases": {
            "question": ["messages"]
        }
    },
    "multimodal": {
        "required": ["image", "chosen"],
        "optional": ["question"],
        "aliases": {
            "image":    ["image", "img", "pixel", "photo", "image_path", "picture"],
            "chosen":   ["answer", "response", "chosen", "caption", "completion",
                         "output", "description", "label"],
            "question": ["question", "prompt", "instruction", "query", "text"]
        }
    }
}

# --- Added: Dynamic padding collator for causal LM
class DynamicCausalCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8, image_mode=False):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.image_mode = image_mode

    def __call__(self, features):
        def _to_py(x):
            if isinstance(x, torch.Tensor):
                return x.tolist()
            return x
        base_feats = []
        for f in features:
            base_feats.append({
                "input_ids": _to_py(f["input_ids"]),
                "attention_mask": _to_py(f.get("attention_mask", []))
            })
        batch = self.tokenizer.pad(
            base_feats,
            padding=True,
            max_length=None,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].masked_fill(batch["attention_mask"].eq(0), -100)

        # --- Image mode: stack pixel_values into the batch
        if self.image_mode:
            pixel_list = [f["pixel_values"] for f in features if "pixel_values" in f]
            if pixel_list:
                try:
                    batch["pixel_values"] = torch.stack(pixel_list)
                except Exception:
                    # fallback: cat along batch dim if shapes differ slightly
                    batch["pixel_values"] = torch.cat([p.unsqueeze(0) if p.dim() == 3 else p for p in pixel_list], dim=0)

        return batch

def compute_metrics(eval_pred):
    """Reserved for future eval use. Not wired to Trainer by default (eval_dataset=None)."""
    logits, labels = eval_pred
    # Use from_numpy to avoid an unnecessary copy when inputs are already numpy arrays
    shift_logits = torch.from_numpy(logits[..., :-1, :].copy())
    shift_labels = torch.from_numpy(labels[..., 1:].copy()).long()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
    try:
        perplexity = math.exp(loss.item())
    except OverflowError:
        perplexity = float("inf")
    return {"perplexity": perplexity, "eval_loss": loss.item()}

def auto_map_roles(columns, training_mode):
    lowered = {col.lower(): col for col in columns}
    aliases = TRAINING_SCHEMAS[training_mode]["aliases"]
    roles = TRAINING_SCHEMAS[training_mode]["required"] + TRAINING_SCHEMAS[training_mode].get("optional", [])

    mapped = {role: None for role in roles}
    for role in roles:
        for alias in aliases.get(role, []):
            if alias in lowered:
                mapped[role] = lowered[alias]
                break
    return mapped.get("question"), mapped.get("chosen"), mapped.get("rejected"), mapped.get("image")

def resolve_column(role, available_columns, training_mode):
    lowered = {col.lower(): col for col in available_columns}
    aliases = TRAINING_SCHEMAS[training_mode]["aliases"]
    for alias in aliases.get(role, []):
        if alias in lowered:
            return lowered[alias]
    return None

def mapped_summary(col_question, col_chosen, col_rejected, col_image=None):
    used = []
    if col_image:    used.append(f" image    → {col_image}")
    if col_question: used.append(f" question → {col_question}")
    if col_chosen: used.append(f" chosen   → {col_chosen}")
    if col_rejected: used.append(f" rejected → {col_rejected}")
    if used:
        print("\nMapped Roles:")
        for line in used:
            print(" ", line)
    else:
        print("No usable column roles mapped!")

#Garbage in -> diamonds out
RE_LATEX_INLINE = re.compile(r"(?<!\$)\$(?:\\.|[^$\\])+\$(?!\$)")                   # $...$
ANSWER_MARKER_RE = re.compile(r'^\s*#{3,6}\s*(?:final\s*answer\s*:)?\s*(?:answer\s*:)?\s*', re.I)
RE_TEX_INLINE = re.compile(r"\\\((?:\\.|[^\\])+\\\)|\\\[(?:\\.|[^\\])+\\\]", re.DOTALL)  # \(...\) or \[...\]
RE_LATEX_ENV = re.compile(r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}", re.DOTALL)       # \begin{env}...\end{env}
RE_HTML = re.compile(r"<[^>]+>")                                            # simple HTML tags
RE_MD_HEADER = re.compile(r"^\s*#{1,6}\s.*$", re.MULTILINE)                      # markdown headers
RE_JUNK_WORDS = re.compile(r"\b(?:Click here|Subscribe|Follow us|Advertisement)\b", re.IGNORECASE)
RE_ECOM_TAGS = re.compile(r"\b(?:SKU|ASIN|MSRP|Price:?\s*\$?\d[\d,\.]*)\b", re.IGNORECASE)
RE_UNICODE_GARBAGE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]")      # zero-width & bidi
RE_MEDIA_PLACEHOLDERS = re.compile(r"\[(?:image|video|audio|figure)[^\]]*\]", re.IGNORECASE)
RE_TRASH_PLACEHOLDER = re.compile(r"\[TRASH\]")
UNBOX_RE = re.compile(r'^[\(\[\{"]?\s*(?:\$+)?\\boxed\{([^}]*)\}(?:\$+)?\s*[\)\]\}"]?\.?\s*$')
BOX_INNER_WRAP_RE = re.compile(r'\\(?:text|mathrm)\{([^}]*)\}')
WHITESPACE_ESC_RE = re.compile(r'(?:[\n\r\t]+|\\\\[ntr])+')
FORMAT_STRIP_RE = re.compile(r"(?:[#*_`]+|[!?]{2,}|&[a-z]+;)")
PUNC_TRANS = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "→": "->", "←": "<-"})
MATH_SPACING_RE   = re.compile(r'\\(?:,|;|:|!|quad|qquad)\s*')
MATH_WRAPPERS_RE  = re.compile(r'\\(?:mathrm|text|operatorname)\{([^}]*)\}')
DOLLAR_INLINE_RE  = re.compile(r'(?<!\$)\$([^\$]+)\$(?!\$)')  # $QU$ -> QU (leaves $$...$$ intact)
_MATH_SIMPLE_MAP = {
    r'\cdot': '*',
    r'\times': '*',
    r'\le': '<=',
    r'\ge': '>=',
    r'\neq': '!=',
    r'\pm': '+/-',
    r'\div': '/',
    r'\triangle': 'triangle',
    r'\quad': ' ',
}
MATH_SIMPLE_MAP_RE = re.compile("|".join(map(re.escape, _MATH_SIMPLE_MAP.keys())))

BAD_TOKENS = [
    "embedreportprint","cloneembedreport","rawdownload","printrawdownload","reportprintclone","embedreport","printclone",
    "guiActive","guiActiveUnfocused","externalToEVA","externalToEVAOnly","PsyNetMessage","vesselType","activeRadarLock",
    "unfocusedRange","targetType","guiIcon","stockLegacySensor","KSPField","persistent","ÃÂ","�","Ã¢","â€™","â€œ","â€",
    "â€“","â€","Ã¼","Ã¶","ÃŸ","[REJECTED ANSWER PLACEHOLDER]","[PLACEHOLDER]","[DUMMY]","[INSERT]",
    "Traceback (most recent call last):","NullPointerException","undefined is not a function","at com.",
    "You are a helpful assistant.","LanguageModelOutput","<|eot|>","<unk>"
]
BAD_TOKENS_RE = re.compile("|".join(map(re.escape, BAD_TOKENS)))

def unbox_field(text: str) -> str:
    t = text.strip()
    m = UNBOX_RE.match(t)
    if not m:
        return text
    inner = BOX_INNER_WRAP_RE.sub(r"\1", m.group(1))
    return inner.strip()

def strip_latex(match) -> str:
    val = match.group(0)
    return val if re.fullmatch(r"\$\d[\d,]*(\.\d{1,2})?\$", val) else ""

def clean_string(s, mode: str = "sft"):
    s = "" if s is None else str(s)
    if not s:
        return ""

    s = unbox_field(s)
    s = WHITESPACE_ESC_RE.sub(" ", s).strip()
    s = FORMAT_STRIP_RE.sub("", s)

    if mode == "math":
        s = MATH_WRAPPERS_RE.sub(r"\1", s)                  # \mathrm{AB} / \text{AB} -> AB
        s = DOLLAR_INLINE_RE.sub(r"\1", s)                  # $QU$ -> QU
        s = MATH_SIMPLE_MAP_RE.sub(lambda m: _MATH_SIMPLE_MAP[m.group(0)], s)  # \cdot -> *, etc.
        s = MATH_SPACING_RE.sub(" ", s)                     # \quad, \, etc. -> space
        s = re.sub(r"\s{2,}", " ", s).strip()

    if mode not in {"latex", "math"}:
        s = RE_LATEX_INLINE.sub(strip_latex, s)
        s = RE_TEX_INLINE.sub("", s)
        s = re.sub(r"\\mathrm\{.*?\}", "", s)
    s = RE_LATEX_ENV.sub("", s)
    s = RE_HTML.sub("", s)
    s = re.sub(r'(?:\$+)?\\boxed\{([^}]*)\}(?:\$+)?', r'\1', s)
    s = re.sub(r"\[/?(?:INST|SYS|USER|ASSISTANT)\]", "", s)
    s = ANSWER_MARKER_RE.sub("", s)
    s = re.sub(r"\{\{.*?\}\}", "", s)
    s = re.sub(r"<\|.*?\|>", "", s)
    s = RE_MD_HEADER.sub("", s)
    s = RE_JUNK_WORDS.sub("", s)
    s = RE_ECOM_TAGS.sub("", s)
    s = re.sub(r"\*\*.*?\*\*", "", s)
    s = RE_UNICODE_GARBAGE.sub("", s)
    s = RE_MEDIA_PLACEHOLDERS.sub("", s)
    if mode != "latex":
        s = RE_TRASH_PLACEHOLDER.sub("", s)
    s = s.translate(PUNC_TRANS)
    if _EMOJI_AVAILABLE:
        s = _emoji_mod.replace_emoji(s, "")
    if mode not in {"latex"}:
        s = re.sub(r"[^\x00-\x7F]+", "", s)
    if BAD_TOKENS_RE.search(s):
        return ""
    return s.strip()

def calc_floor(steps_total: int) -> float:
    if steps_total < 4000:
        return 0.22  
    return 0.18


def synthesize_prompt_dataset(raw_dataset, col_question, col_chosen, col_rejected, training_mode, clean, col_image=None):
    original_count = len(raw_dataset)

    # For multimodal we avoid .to_pandas() entirely — converting to a DataFrame
    # pulls ALL image bytes out of the compressed Arrow cache and into RAM at once,
    # which causes the 5–11 GB realloc OOM seen with even modest image datasets.
    # Instead we use .map() so images stay compressed in Arrow until tokenization.
    if training_mode == "multimodal":
        def build_multimodal_row(row):
            parts = []
            if col_question:
                q = row.get(col_question, "") or ""
                if str(q).strip():
                    parts.append(clean(str(q)))
            if col_chosen:
                a = row.get(col_chosen, "") or ""
                if str(a).strip():
                    parts.append(clean(str(a)))
            text = "\n\n".join(parts)
            result = {"text": text}
            if col_image and col_image != "image":
                result["image"] = row[col_image]
            return result

        # num_proc=1: Arrow map with image columns is not fork-safe on Windows
        # writer_batch_size=50: flush to disk every 50 rows — prevents large RAM accumulation
        processed = raw_dataset.map(
            build_multimodal_row,
            num_proc=1,
            writer_batch_size=50,
        )
        # Rename image col to "image" if needed and drop everything else
        if col_image and col_image != "image" and col_image in processed.column_names:
            processed = processed.rename_column(col_image, "image")
        cols_to_drop = [c for c in processed.column_names if c not in ("text", "image")]
        if cols_to_drop:
            processed = processed.remove_columns(cols_to_drop)
        # Filter empty text rows
        processed = processed.filter(lambda e: bool(str(e["text"]).strip()), num_proc=1)
        _info(f"Retained {len(processed):,} / {original_count:,} rows")

    else:
        # Text-only path: pandas is fine, no image bytes in memory
        df = raw_dataset.to_pandas()

        def build_prompt(row):
            parts = []
            if col_question and pd.notna(row.get(col_question, "")):
                parts.append(clean(str(row[col_question])))
            if training_mode == "sft":
                if col_chosen and pd.notna(row.get(col_chosen, "")):
                    parts.append(clean(str(row[col_chosen])))
            return "\n\n".join(parts)

        df["text"] = df.apply(build_prompt, axis=1)
        df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
        df = df[["text"]]
        _info(f"Retained {len(df):,} / {original_count:,} rows")
        processed = Dataset.from_pandas(df)

    # Preview first row
    if len(processed) > 0:
        first_raw = raw_dataset[0]
        q = clean(str(first_raw.get(col_question, ""))) if col_question else ""
        a = clean(str(first_raw.get(col_chosen, ""))) if col_chosen else ""
        _width = 58
        if q or a:
            print(f"\n  {'─' * _width}")
            print(f"  Sample row")
            print(f"  {'─' * _width}")
        if q:
            print(f"  Prompt   : {q[:120]}{'…' if len(q) > 120 else ''}")
        if a:
            print(f"  Response : {a[:120]}{'…' if len(a) > 120 else ''}")
        if q or a:
            print()
    else:
        _err("No usable prompt found!")

    return processed


# --- Added: quick length measurement (P95 auto window)
def measure_lengths(ds, tokenizer):
    _num_proc = min(12, os.cpu_count() or 4)

    def _len_map(e):
        ids = tokenizer(
            e["text"],
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False
        )["input_ids"]
        return {"length": len(ids)}

    ds = ds.map(_len_map, num_proc=_num_proc)
    lengths = ds["length"]
    if not lengths:
        raise ValueError("[ERROR] Dataset is empty after processing — no rows to measure lengths from.")
    p50 = int(np.percentile(lengths, 50))
    p95 = int(np.percentile(lengths, 95))
    p99 = int(np.percentile(lengths, 99))
    auto_len = max(128, min(1024, p95))
    _info(f"Lengths — p50: {p50}  p95: {p95}  p99: {p99}  →  max_length set to {auto_len}")
    return ds, auto_len



def _decode_image(raw) -> "PILImage.Image":
    """Convert raw image field (bytes, dict with 'bytes', or PIL) to a PIL Image."""
    if isinstance(raw, PILImage.Image):
        return raw.convert("RGB")
    if isinstance(raw, dict):
        raw = raw.get("bytes") or raw.get("path") or raw
    if isinstance(raw, (bytes, bytearray)):
        return PILImage.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(raw, str) and os.path.isfile(raw):
        return PILImage.open(raw).convert("RGB")
    raise ValueError(f"Cannot decode image from type: {type(raw)}")


def tokenize(example, tokenizer, training_mode="sft", max_length=MAX_LENGTH, processor=None):
    # --- Multimodal path: use processor to encode image + text together
    if processor is not None and training_mode == "multimodal" and "image" in example:
        try:
            image = _decode_image(example["image"])
        except Exception as e:
            print(f"[WARN] Skipping unreadable image: {e}")
            return {"input_ids": [], "attention_mask": [], "labels": [], "pixel_values": None}

        # Phi-3-Vision prompt format
        text = example.get("text", "")
        prompt = f"<|user|>\n<|image_1|>\n{text}<|end|>\n<|assistant|>"
        encoded = processor(
            text=prompt,
            images=image,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids      = encoded["input_ids"][0].tolist()
        attention_mask = encoded["attention_mask"][0].tolist()
        pixel_values   = encoded["pixel_values"][0]

        # --- Fix 5: mask question/image tokens so only the answer is supervised.
        # Find the assistant turn boundary — everything from <|assistant|> onward
        # is the answer we want to train on; everything before gets -100.
        # Phi-3-Vision uses token ID for "<|assistant|>" as the boundary marker.
        assistant_token_ids = tokenizer.encode("<|assistant|>", add_special_tokens=False)
        labels = [-100] * len(input_ids)
        if assistant_token_ids:
            # Find last occurrence of the assistant marker (handles edge cases where
            # the string "<|assistant|>" might appear in the question text)
            a_id = assistant_token_ids[0]
            boundary = -1
            for i in range(len(input_ids) - 1, -1, -1):
                if input_ids[i] == a_id:
                    boundary = i + 1  # supervise from the token AFTER <|assistant|>
                    break
            if boundary > 0:
                labels[boundary:] = input_ids[boundary:]
            else:
                # Fallback: if marker not found, supervise full sequence
                labels = list(input_ids)

        return {
            "input_ids":      [int(x) for x in input_ids],
            "attention_mask": [int(x) for x in attention_mask],
            "labels":         [int(x) for x in labels],
            "pixel_values":   pixel_values,
            **({"length": len(input_ids)} if "length" not in example else {"length": int(example["length"])}),
        }

    # --- Text-only path (unchanged)
    tokens = tokenizer(
        example["text"],
        padding=False,
        truncation=True,
        max_length=max_length
    )
    out = {
        "input_ids": [int(x) for x in tokens["input_ids"]],
        "attention_mask": [int(x) for x in tokens["attention_mask"]],
        "labels": [int(x) for x in tokens["input_ids"]],
    }
    if "length" in example:
        try:
            out["length"] = int(example["length"])
        except Exception:
            pass
    return out

def detect_training_mode(columns):
    lowered = {col.lower(): col for col in columns}

    # Determine check order: prioritise multimodal when an image-named column is
    # present, otherwise use schema insertion order.  Without this, "sft" (which
    # only requires one "chosen"-aliased column) fires before "multimodal" and
    # silently swallows image datasets.
    image_aliases = set(TRAINING_SCHEMAS["multimodal"]["aliases"]["image"])
    has_image_col = any(col in image_aliases for col in lowered)
    if has_image_col:
        check_order = ["multimodal"] + [m for m in TRAINING_SCHEMAS if m != "multimodal"]
    else:
        check_order = list(TRAINING_SCHEMAS.keys())

    for mode in check_order:
        schema  = TRAINING_SCHEMAS[mode]
        required = schema["required"]
        aliases  = schema["aliases"]
        if all(
            any(alias in lowered for alias in aliases.get(role, []))
            for role in required
        ):
            return mode
    raise ValueError(f"[ERROR] Could not detect training mode from columns: {list(columns)}")
    
def log_dataset(dataset_path, log_file_path):
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    try:
        with open(log_file_path, "a+", encoding="utf-8") as f:
            f.seek(0)
            content = f.read()
            if dataset_name not in content:
                f.write(dataset_name + "\n")
                _ok(f"Logged '{dataset_name}' to training chain")
                return True
            else:
                _info(f"'{dataset_name}' already in training chain")
                return False
    except Exception as e:
        _warn(f"Could not log dataset: {e}")
        return False
        
def purge_checkpoints():
    try:
        removed = 0
        for name in os.listdir(OUTPUT_DIR):
            if name.startswith("checkpoint-"):
                path = os.path.join(OUTPUT_DIR, name)
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                    removed += 1

        last_link = os.path.join(OUTPUT_DIR, "checkpoint-last")
        if os.path.islink(last_link):
            os.unlink(last_link)
        elif os.path.isdir(last_link):
            shutil.rmtree(last_link, ignore_errors=True)

        last_good = os.path.join(OUTPUT_DIR, "last_good_ckpt")
        if os.path.isdir(last_good):
            shutil.rmtree(last_good, ignore_errors=True)

        _info(f"Purged {removed} old checkpoints")
    except Exception as e:
        _warn(f"Failed to purge checkpoints: {e}")

def load_and_prepare_dataset(dataset_path, cleaning_mode, image_mode=False, base_model_override=None):
    _num_proc = min(12, os.cpu_count() or 4)

    # --- Support HF Hub dataset IDs (e.g. "username/dataset-name") as well as local parquet
    is_hf_hub = not os.path.exists(dataset_path) and "/" in dataset_path
    if is_hf_hub:
        _info(f"Source: HF Hub — {dataset_path}")
        # Disable offline mode temporarily for HF Hub load
        _prev_offline = os.environ.pop("HF_DATASETS_OFFLINE", None)
        _prev_tr_offline = os.environ.pop("TRANSFORMERS_OFFLINE", None)
        _prev_hf_offline = os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            raw_dataset = load_dataset(dataset_path, split="train")
        finally:
            if _prev_offline:    os.environ["HF_DATASETS_OFFLINE"]  = _prev_offline
            if _prev_tr_offline: os.environ["TRANSFORMERS_OFFLINE"]  = _prev_tr_offline
            if _prev_hf_offline: os.environ["HF_HUB_OFFLINE"]        = _prev_hf_offline
    else:
        raw_dataset = load_dataset("parquet", data_files=dataset_path, split="train")

    shuffle_seed = random.randint(1, 999999)
    raw_dataset = raw_dataset.shuffle(seed=shuffle_seed)
    columns = raw_dataset.column_names

    # Force multimodal mode if --image flag is set, else auto-detect
    if image_mode:
        training_mode = "multimodal"
    else:
        training_mode = detect_training_mode(columns)

    col_question, col_chosen, col_rejected, col_image = auto_map_roles(columns, training_mode)
    _banner("Dataset")
    _info(f"Mode: {training_mode.upper()}  |  Cleaning: {cleaning_mode or 'math'}  |  Seed: {shuffle_seed}")
    _info(f"Image    → {'—' if not col_image    else col_image}")
    _info(f"Question → {'—' if not col_question else col_question}")
    _info(f"Answer   → {'—' if not col_chosen   else col_chosen}")
    _info(f"Rejected → {'—' if not col_rejected else col_rejected}")
    effective_mode = cleaning_mode or "math"
    clean = lambda s: clean_string(s, mode=effective_mode)

    # Resolve model path — CLI override > hardcoded constant
    _default_base = base_model_override or (BASE_MODEL_IMAGE if image_mode else BASE_MODEL)
    model_name = _resolve_model_name(MERGED_DIR, MERGE_SUCCESS_FLAG, _default_base)
    if os.path.exists(MERGE_SUCCESS_FLAG):
        _ok(f"Resuming from previous merged model: {MERGED_DIR}")
    else:
        base_label = "image base (Phi-3-Vision)" if image_mode else f"text base ({os.path.basename(_default_base)})"
        if base_model_override:
            base_label = f"override: {base_model_override}"
        _info(f"Base model: {model_name}")

    # --- Processor (image mode) vs Tokenizer (text mode)
    profile = resolve_model_profile(model_name)
    processor = None
    if image_mode:
        _info("Loading AutoProcessor (Phi-3-Vision)...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True
        )
        tokenizer = processor.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=profile["use_fast_tokenizer"],
            trust_remote_code=profile["trust_remote_code"],
            local_files_only=True
        )

    if not image_mode and profile["inject_special_tokens"]:
        special_tokens = {
            "pad_token": "<|pad|>",
            "bos_token": "<|startoftext|>",
            "eos_token": "<|endoftext|>"
        }
        tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token or tokenizer.unk_token

    if training_mode == "causal":
        _info("Causal mode — GPT-style synthesis will be applied")
        def synth_gpt_prompt(example):
            instruction = example.get("instruction", "").strip()
            input_ = example.get("input", "").strip()
            output = example.get("output", "").strip()
            if instruction and input_:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_}\n\n### Response:\n{output}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            return {"text": clean(text)}
        processed = raw_dataset.map(synth_gpt_prompt, num_proc=_num_proc)
    else:
        processed = synthesize_prompt_dataset(
            raw_dataset, col_question, col_chosen, col_rejected,
            training_mode, clean, col_image=col_image
        )

    # Measure lengths and derive a robust max_length
    _banner("Tokenization")
    _info("Measuring sequence lengths...")
    processed, auto_len = measure_lengths(processed, tokenizer)

    _info("Tokenizing dataset...")

    # Columns to drop after tokenization.
    # "length" is kept as it was computed pre-tokenize and is reused.
    # "image" must also be kept alive through the .map() call in image mode —
    # remove_columns executes BEFORE the map function runs on each row, so
    # stripping "image" here would make it invisible to tokenize().
    protected = {"length"}
    if image_mode:
        protected.add("image")
    cols_to_remove = [c for c in processed.column_names if c not in protected]

    tokenized = processed.map(
        lambda e: tokenize(
            example=e,
            tokenizer=tokenizer,
            training_mode=training_mode,
            max_length=auto_len,
            processor=processor
        ),
        remove_columns=cols_to_remove,
        batched=False,
        num_proc=1 if image_mode else _num_proc,  # image decode not safe for multiproc
        writer_batch_size=50 if image_mode else 1000,  # small batches prevent OOM realloc
        keep_in_memory=False,   # stream through Arrow disk cache, never buffer full dataset
    )

    # Filter out rows where image decoding failed.
    # Two failure signatures to catch:
    #   (a) input_ids=[]  — processor never ran (early decode error)
    #   (b) pixel_values=None — image decode failed after processor started
    if image_mode:
        before = len(tokenized)
        tokenized = tokenized.filter(
            lambda e: len(e["input_ids"]) > 0 and e.get("pixel_values") is not None
        )
        dropped = before - len(tokenized)
        if dropped:
            _warn(f"Dropped {dropped} rows with unreadable images")

    _info("Casting columns to tensor format...")
    tokenized = tokenized.cast_column("input_ids", Sequence(Value("int64")))
    tokenized = tokenized.cast_column("attention_mask", Sequence(Value("int64")))
    tokenized = tokenized.cast_column("labels", Sequence(Value("int64")))
    if "length" in tokenized.column_names:
        tokenized = tokenized.cast_column("length", Value("int32"))

    fmt_cols = ["input_ids", "attention_mask", "labels", "length"]
    if image_mode and "pixel_values" in tokenized.column_names:
        fmt_cols.append("pixel_values")

    tokenized.set_format(type="torch", columns=fmt_cols)
    roc()

    train_dataset = tokenized
    eval_dataset = None
    just_logged = log_dataset(dataset_path, TRAINING_CHAIN_PATH)
    if just_logged:
        purge_checkpoints()
    return train_dataset, eval_dataset, tokenizer, training_mode, model_name, processor


def select_scheduler(dataset_rows: int, epochs: int, min_stop_steps: int) -> str:
    total = max(1, dataset_rows) * max(1, epochs) 
    pressure = min_stop_steps / total              
    EXPOSURE_FLOOR = 0.18                          
    if total < 1500 or epochs <= 1:
        return "linear"
    if pressure >= 2 * EXPOSURE_FLOOR:
        return "linear"
    if 1500 <= total <= 40000 and epochs >= 2 and pressure <= EXPOSURE_FLOOR:
        if (total / 3) < min_stop_steps:
            return "cosine_with_restarts"
    return "cosine"

def load_model(tokenizer, model_name, image_mode=False, base_model_override=None):
    profile = resolve_model_profile(model_name)

    if image_mode:
        _banner("Model")
        _info("Loading Phi-3-Vision (AutoModelForVision2Seq)...")
        base_model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
            # Vision models stay on eager — Phi-3-Vision's vision encoder is
            # incompatible with SDPA's dispatcher. flash_attention_2 is also
            # unavailable on Windows 11.
            _attn_implementation="eager",
        ).to("cuda")
    else:
        _banner("Model")
        _info(f"Loading {os.path.basename(model_name)} (AutoModelForCausalLM)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=profile["trust_remote_code"],
            local_files_only=True,
            _attn_implementation=profile["attn_implementation"],
        ).to("cuda")

    # Disable KV cache on the BASE model before any PEFT wrapping.
    # Must happen here — setting it on the PeftModel wrapper afterwards does not
    # reliably propagate for all architectures, leaving output tensors detached
    # from the autograd graph and causing "does not require grad" crashes.
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    last_ckpt = os.path.join(OUTPUT_DIR, "checkpoint-last")

    use_last = os.path.isdir(last_ckpt)
    if use_last:
        recorded_base = None
        try:
            with open(os.path.join(last_ckpt, "adapter_config.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)
            recorded_base = cfg.get("base_model_name_or_path")
        except Exception:
            pass

        base_has_config = os.path.isfile(os.path.join(model_name, "config.json"))
        if (not base_has_config) or (recorded_base and os.path.abspath(recorded_base) != os.path.abspath(model_name)):
            _warn("Checkpoint base mismatch — starting fresh LoRA")
            use_last = False

    if use_last:
        _ok("Resuming LoRA adapter from checkpoint-last")
        if profile["resize_embeddings"] and tokenizer.added_tokens_encoder:
            base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, last_ckpt, is_trainable=True)
    else:
        _info("Initialising fresh LoRA adapter")
        if image_mode:
            # VLM: smaller rank for VRAM headroom; no modules_to_save needed
            lora_r      = 8
            lora_alpha  = 16
            lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
            lora_modules_to_save = None
        else:
            lora_r      = profile["lora_r"]
            lora_alpha  = profile["lora_alpha"]
            lora_targets = profile["lora_targets"]
            lora_modules_to_save = profile["modules_to_save"]

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_targets,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            base_model_name_or_path=os.path.abspath(model_name),
            modules_to_save=lora_modules_to_save,
        )

        # Resize BEFORE wrapping with LoRA so new embedding rows are covered by
        # the adapter and the gradient graph stays intact. Only done for models
        # that inject special tokens (currently Phi-2 only).
        if profile["resize_embeddings"] and tokenizer.added_tokens_encoder:
            _info(f"Resizing embeddings: +{len(tokenizer.added_tokens_encoder)} new tokens")
            base_model.resize_token_embeddings(len(tokenizer))

        model = get_peft_model(base_model, peft_config)

        # modules_to_save wraps lm_head so PEFT owns it and gradient flow is
        # guaranteed through it — but we don't want to actually *train* 98M lm_head
        # weights on top of the LoRA params. Freeze them immediately after wrapping.
        # The gradient path stays intact; only LoRA adapter weights update.
        if lora_modules_to_save:
            for name, param in model.named_parameters():
                if any(
                    f"modules_to_save.default.{m}" in name
                    for m in lora_modules_to_save
                ):
                    param.requires_grad_(False)

    # Some architectures (e.g. Phi-3 with trust_remote_code) can detach input
    # embeddings from the autograd graph. This call forces gradient flow through
    # inputs regardless of the custom forward implementation.
    if profile["enable_input_grads"]:
        model.enable_input_require_grads()

    print_trainable_parameters(model)

    # Gradient checkpointing for image mode — saves ~3-4 GB VRAM on 12 GB card
    if image_mode:
        _info("Gradient checkpointing enabled (VRAM optimisation)")
        model.gradient_checkpointing_enable()

    return model

def print_trainable_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _ok(f"Trainable params: {trainable:,} / {total:,}  ({100 * trainable / total:.2f}%)")
        
class EarlyStopByLoss(TrainerCallback):


    def __init__(
        self,
        steps_total: int,
        mode: str = "sft",
        active: bool = True,
        hard_cap_steps: int | None = None,

        # Gates
        exposure_floor: float = 0.10,   # fraction of an epoch seen before evaluating
        quality_lr_frac: float = 0.90,  # allow checks once LR <= 85% of max seen (use 1.0 to disable)

        # Windows / thresholds
        window: int = 128,              # will be scaled below
        ema_beta: float = 0.90,
        std_floor: float = 0.10,        # smaller, realistic floor for common loss scales
        min_abs_improve: float = 0.03,  # absolute EMA drop to count as "good"
        min_sigma_improve: float = 0.50,# EMA drop measured in robust sigmas to count as "good"
        slope_window: int | None = None,# if None → use scaled window
        slope_thresh: float = 0.002,    # >= thresh means worsening
        patience: int = 80,             # steps (not "events"): how long we can go without a "good" improvement
        cooldown_after_best: int = 15,  # steps after a new best where we don't punish lack of improvement
        verbose_every: int = 0          # 0 = quiet; else print summary every N steps
    ):
        super().__init__()
        self.steps_total = max(1, int(steps_total))
        self.mode = mode
        self.active = active
        self.hard_cap_steps = hard_cap_steps

        self.exposure_floor = float(exposure_floor)
        self.quality_lr_frac = float(quality_lr_frac)

        # Scale windows with run length
        base_w = max(96, int(0.015 * self.steps_total))
        self.window = max(min(window, 512), base_w)
        self.ema_beta = float(ema_beta)
        self.std_floor = float(std_floor)
        self.min_abs_improve = float(min_abs_improve)
        self.min_sigma_improve = float(min_sigma_improve)
        self.slope_window = max(48, int(0.75 * self.window)) if slope_window is None else int(slope_window)
        self.slope_thresh = float(slope_thresh)
        self.patience = int(patience)
        self.cooldown_after_best = int(cooldown_after_best)
        self.verbose_every = int(verbose_every)

        # State
        self.losses: list[float] = []
        self.ema_series: list[float] = []
        self.ema: float | None = None
        self.prev_ema: float | None = None
        self.best_ema: float = float("inf")
        self.since_best: int = 0
        self.cooldown: int = 0
        self.triggered: bool = False
        self.max_lr_seen: float = 0.0
        self.last_step_seen: int = -1

    @staticmethod
    def _mad_sigma(arr: list[float]) -> float:
        if len(arr) < 3:
            return 0.0
        a = np.asarray(arr, dtype=np.float32)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        return 1.4826 * mad  # ≈ σ for normal

    def _epoch_fraction(self, state) -> float:
        step = max(0, int(getattr(state, "global_step", 0)))
        return step / float(self.steps_total)

    def _update_ema(self, loss: float) -> None:
        if self.ema is None:
            self.ema = loss
        else:
            b = self.ema_beta
            self.ema = b * self.ema + (1.0 - b) * loss

    def _lr_gate_ok(self, logs: dict) -> bool:
        if self.quality_lr_frac >= 1.0:
            return True
        lr = logs.get("learning_rate", None)
        if lr is None:
            return True
        try:
            lr = float(lr)
        except Exception:
            return True
        if lr > self.max_lr_seen:
            self.max_lr_seen = lr
        return lr <= (self.max_lr_seen * self.quality_lr_frac + 1e-12)

    def _recent_slice(self, arr: list[float], k: int) -> list[float]:
        k = max(1, int(k))
        if len(arr) <= k:
            return arr[:]
        return arr[-k:]

    def _slope(self, series: list[float]) -> float:
        if len(series) < 8:
            return 0.0
        x = np.arange(len(series), dtype=np.float32)
        y = np.asarray(series, dtype=np.float32)
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.active or logs is None:
            return

        # Optional hard cap (set None to fully disable)
        if self.hard_cap_steps is not None:
            if int(getattr(state, "global_step", 0)) >= int(self.hard_cap_steps):
                control.should_training_stop = True
                self.triggered = True
                return

        if "loss" not in logs:
            return

        step = int(getattr(state, "global_step", 0))
        if step == self.last_step_seen:
            return
        self.last_step_seen = step

        loss = float(logs["loss"])
        warmup = int(getattr(args, "warmup_steps", 0) or 0)

        # Update EMA & series
        self._update_ema(loss)
        if self.prev_ema is None:
            self.prev_ema = self.ema
        self.ema_series.append(self.ema)

        # Track best
        if self.ema < self.best_ema - 1e-12:
            self.best_ema = self.ema
            self.since_best = 0
            self.cooldown = self.cooldown_after_best
        else:
            self.since_best += 1
            if self.cooldown > 0:
                self.cooldown -= 1

        # Gates
        if step < warmup:
            return
        if self._epoch_fraction(state) < self.exposure_floor:
            return
        if not self._lr_gate_ok(logs):
            return

        # Need enough history
        min_history = max(24, self.window // 3)
        if len(self.ema_series) < min_history:
            self.prev_ema = self.ema
            return

        # Robust variability over EMA series
        recent_ema = self._recent_slice(self.ema_series, self.window)
        sigma = max(self._mad_sigma(recent_ema), self.std_floor)

        # Improvement tests
        abs_drop = (self.prev_ema - self.ema)
        sigma_drop = abs_drop / (sigma + 1e-12)
        improved = (abs_drop >= self.min_abs_improve) or (sigma_drop >= self.min_sigma_improve)

        # Worsening / plateau tests on EMA slope
        slope_win = self._recent_slice(self.ema_series, self.slope_window)
        slope = self._slope(slope_win)

        worsening = slope >= self.slope_thresh
        # Plateau: long time since best, tiny slope magnitude, and low variability
        plateau = (self.since_best >= self.patience) and (abs(slope) <= self.slope_thresh) and (sigma <= max(0.5 * self.std_floor, 0.05))

        # Decision
        stop_now = False
        if improved:
            # reset prev reference for next drop calc
            self.prev_ema = self.ema
        else:
            # If not improved for a while and statistics say "flat", stop.
            stop_now = plateau or worsening

        # Occasional debug (quiet by default)
        if self.verbose_every and (step % self.verbose_every == 0):
            print(f"[ES] step={step} ema={self.ema:.4f} best={self.best_ema:.4f} "
                  f"abs_drop={abs_drop:.4f} sigma_drop={sigma_drop:.2f} "
                  f"sigma={sigma:.4f} slope={slope:.5f} since_best={self.since_best} "
                  f"plateau={plateau} worsening={worsening}")

        if stop_now:
            control.should_training_stop = True
            self.triggered = True
            return

    def on_train_end(self, args, state, control, **kwargs):
        if self.hard_cap_steps is not None and int(getattr(state, "global_step", 0)) >= int(self.hard_cap_steps):
            self.triggered = True

            
def compute_min_steps(train_dataset, batch_size=1, grad_accum=6):
    steps_per_epoch = max(1, len(train_dataset) // (batch_size * grad_accum))
    dyn_frac = calc_floor(steps_per_epoch)
    return max(int(steps_per_epoch * dyn_frac), 50)
    
def dynamic_early_stop_cap(steps_total):
    min_cap = 0.65
    max_cap = 0.95
    log_steps = math.log10(max(steps_total, 10))
    scale = min(1.0, max(0.0, (log_steps - 1) / 4))  # 10 to 100k steps
    cap = max_cap - (max_cap - min_cap) * scale
    return round(cap, 4)
    
def select_file() -> str:
    try:
        if platform.system() == "Windows":
            ps_script = r"""
Add-Type -AssemblyName System.Windows.Forms | Out-Null
$ofd = New-Object System.Windows.Forms.OpenFileDialog
$ofd.Filter = 'Parquet files (*.parquet)|*.parquet'
$ofd.Title = 'Select a Parquet file'
$ofd.Multiselect = $false
if ($ofd.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    Write-Output $ofd.FileName
}
"""
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-STA", "-Command", ps_script],
                capture_output=True, text=True, check=False
            )
            path = (proc.stdout or "").strip()
            if path:
                return path
    except Exception as e:
        _warn(f"PowerShell file picker failed — falling back to text input")
    return input("Dataset path (.parquet): ").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force',      action='store_true', help="Disable early stopping")
    parser.add_argument('--latex',      action='store_true', help="Preserve LaTeX and scientific equations")
    parser.add_argument('--code',       action='store_true', help="Preserve code formatting and indentation")
    parser.add_argument('--epoch',      type=int, default=1, help="Number of training epochs (default: 1)")
    parser.add_argument('--steps',      type=int, help="Terminate after a fixed number of steps")
    parser.add_argument('--image',      action='store_true', help="Enable multimodal (image+text) training via Phi-3-Vision")
    parser.add_argument('--hf_dataset', type=str, default=None, help="HuggingFace Hub dataset ID (e.g. username/dataset-name)")
    parser.add_argument('--base_model', type=str, default=None, help="Override base model path (default: BASE_MODEL or BASE_MODEL_IMAGE constant)")
    parsed_args, _ = parser.parse_known_args()
    force_epoch  = parsed_args.force
    image_mode   = parsed_args.image

    # Resolve base model: CLI arg > hardcoded constant
    _banner("Freethought Trainer")
    if parsed_args.base_model:
        resolved_base = parsed_args.base_model
        _info(f"Base model override: {resolved_base}")
    else:
        resolved_base = BASE_MODEL_IMAGE if image_mode else BASE_MODEL

    modes = []
    if parsed_args.latex:
        modes.append("latex")
    if parsed_args.code:
        modes.append("code")

    cleaning_mode = "+".join(modes) if modes else "math"

    # --- Dataset selection: HF Hub flag > file picker
    if parsed_args.hf_dataset:
        dataset_path = parsed_args.hf_dataset
        _info(f"HF Hub dataset: {dataset_path}")
    else:
        dataset_path = select_file()
        if image_mode and dataset_path and not dataset_path.lower().endswith(".parquet"):
            # Allow non-parquet for HF Hub paths entered via input()
            if not ("/" in dataset_path and not os.path.exists(dataset_path)):
                _err("Please select a valid .parquet file or pass --hf_dataset")
                return
        elif not image_mode:
            if (not dataset_path) or (not dataset_path.lower().endswith(".parquet")):
                _err("Please select a valid .parquet file")
                return

    if cleaning_mode in {"latex", "math"} and not image_mode:
        try:
            df_columns = pd.read_parquet(dataset_path, engine="pyarrow", columns=None).columns
            training_mode_peek = detect_training_mode(df_columns)
        except Exception as e:
            _warn(f"Could not auto-detect cleaning mode: {e}")

    if image_mode:
        _info("Image mode enabled (Phi-3-Vision / multimodal)")

    train_dataset, eval_dataset, tokenizer, training_mode, model_name, processor = load_and_prepare_dataset(
        dataset_path, cleaning_mode, image_mode=image_mode, base_model_override=resolved_base
    )

    profile = resolve_model_profile(model_name)
    _info(f"Model profile: {os.path.basename(model_name)}  |  grad_accum: {profile['grad_accum']}  |  lr: {profile['learning_rate']}")
    estimated_steps = len(train_dataset) // (BATCH_SIZE * profile["grad_accum"])
    use_steps = parsed_args.steps is not None

    if use_steps:
        total_steps = parsed_args.steps
        num_train_epochs = max(1, total_steps // max(estimated_steps, 1))
        _info(f"Fixed steps: {total_steps}")
    else:
        num_train_epochs = parsed_args.epoch
        total_steps = estimated_steps * num_train_epochs
        _info(f"Estimated steps: {total_steps}")

    use_early_stop = not force_epoch and not use_steps
    min_stop_steps = compute_min_steps(train_dataset, BATCH_SIZE, profile["grad_accum"])
    early_stop = None

    if use_early_stop:
        cap = dynamic_early_stop_cap(total_steps)
        hard_cap_steps = int(total_steps * cap)
        early_stop = EarlyStopByLoss(
            steps_total=total_steps,
            mode=training_mode,
            active=True,
            hard_cap_steps=hard_cap_steps,
            exposure_floor=0.18,
            quality_lr_frac=0.60,
            window=128,
            ema_beta=0.90,
            std_floor=0.22,
            min_abs_improve=0.04,
            min_sigma_improve=0.50,
            slope_window=None,
            slope_thresh=0.010,
            patience=8,
            cooldown_after_best=3
        )
        _info(f"Early stop: floor ≥ {int(100 * early_stop.exposure_floor)}%  |  hard cap {int(cap * 100)}% ({hard_cap_steps} steps)")
    else:
        _info("Early stop: disabled")

    selected_scheduler = select_scheduler(train_dataset.num_rows, num_train_epochs, min_stop_steps)
    _banner("Training Config")
    _info(f"Rows: {len(train_dataset):,}  |  Steps: {total_steps}  |  Epochs: {num_train_epochs}")
    _info(f"Scheduler: {selected_scheduler.upper()}  |  LR: {profile['learning_rate']}  |  Batch: {BATCH_SIZE}  |  Grad accum: {profile['grad_accum']}")
    dynamic_warmup = max(75, int(0.01 * total_steps))

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=profile["grad_accum"],
        learning_rate=profile["learning_rate"],
        log_level="error",
        warmup_steps=dynamic_warmup,
        remove_unused_columns=False,
        num_train_epochs=num_train_epochs if not use_steps else 1,
        max_steps=total_steps if use_steps else -1,
        eval_strategy="no",
        logging_steps=LOG_STEPS,
        logging_strategy="steps",
        logging_first_step=True,
        disable_tqdm=False,
        report_to=[],
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_LIMITS,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
        lr_scheduler_type=selected_scheduler,
        bf16=True,
        group_by_length=True,
        length_column_name="length",
        # Gradient checkpointing enabled via model.gradient_checkpointing_enable() in load_model
        # when image_mode=True; setting here too ensures TrainingArguments consistency
        gradient_checkpointing=image_mode,
    )

    model = load_model(tokenizer, model_name, image_mode=image_mode, base_model_override=resolved_base)
    data_collator = DynamicCausalCollator(tokenizer, pad_to_multiple_of=8, image_mode=image_mode)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=[] if early_stop is None else [early_stop],
    )

    _training_error = None
    try:
        if eval_dataset is not None:
            _info(f"Eval rows: {len(eval_dataset):,}")

        if total_steps < 1:
            _err("Not enough data to train!")
            exit()

        _banner("Training")
        _info(f"Train rows: {len(train_dataset):,}  |  Steps: {total_steps}  |  Warmup: {dynamic_warmup}")
        last_ckpt = os.path.join(OUTPUT_DIR, "checkpoint-last")

        if os.path.isdir(last_ckpt):
            _ok(f"Resuming from checkpoint: {last_ckpt}")
            trainer.train(resume_from_checkpoint=last_ckpt)
        else:
            trainer.train()
        roc()
    except KeyboardInterrupt:
        _warn("Keyboard interrupt — saving current state...")
        if isinstance(model, PeftModel):
            model.save_pretrained(MERGED_DIR)
        else:
            trainer.save_model(MERGED_DIR)
    except Exception as e:
        # Unexpected training error — record it and fall through to merge+save
        # so the model isn't lost, then re-raise at the very end.
        _training_error = e
        _err(f"Training error: {e}")
        _warn("Attempting emergency save before merge...")
        try:
            if isinstance(model, PeftModel):
                model.save_pretrained(MERGED_DIR)
            else:
                trainer.save_model(MERGED_DIR)
        except Exception as save_err:
            _err(f"Emergency save also failed: {save_err}")
    finally:
        _banner("Cleanup")
        _info("Cleaning checkpoints...")
        ckpts = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        ckpts = sorted(
            [d for d in ckpts if d.split("-")[-1].isdigit()],
            key=lambda d: int(d.split("-")[-1])
        )

        if ckpts:
            latest = ckpts[-1]
            last_path = os.path.join(OUTPUT_DIR, latest)
            symlink_path = os.path.join(OUTPUT_DIR, "checkpoint-last")

            if os.path.islink(symlink_path) or os.path.exists(symlink_path):
                try:
                    os.unlink(symlink_path)
                except OSError:
                    shutil.rmtree(symlink_path)
            try:
                os.symlink(last_path, symlink_path, target_is_directory=True)
            except (OSError, NotImplementedError):
                _warn("Could not create symlink — copying instead")
                shutil.copytree(last_path, symlink_path)
                _ok(f"checkpoint-last → {latest}")

            backup_path = os.path.join(OUTPUT_DIR, "last_good_ckpt")
            shutil.copytree(last_path, backup_path, dirs_exist_ok=True)
            _ok("Last checkpoint backed up")

    if isinstance(model, PeftModel):
        _banner("Saving")
        _info("Merging LoRA weights into base model...")
        try:
            merged_model = model.merge_and_unload()
            final_model_to_save = merged_model
            _ok("LoRA merge complete")
        except Exception as e:
            _warn(f"Merge failed: {e} — saving adapter as-is")
            final_model_to_save = model
    else:
        _banner("Saving")
        _info("Model already merged")
        final_model_to_save = model

    _info(f"Saving to: {MERGED_DIR}")
    os.makedirs(MERGED_DIR, exist_ok=True)
    save_profile = resolve_model_profile(resolved_base)
    try:
        final_model_to_save.save_pretrained(
            MERGED_DIR,
            safe_serialization=save_profile["safe_serialization"],
        )
    except Exception as e:
        _err(f"save_pretrained failed: {e}")
        _warn("Retrying with rebuilt base model...")
        if image_mode:
            base = AutoModelForVision2Seq.from_pretrained(
                resolved_base,
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                _attn_implementation="eager",
            ).to("cuda")
        else:
            base = AutoModelForCausalLM.from_pretrained(
                resolved_base,
                local_files_only=True,
                trust_remote_code=save_profile["trust_remote_code"],
                torch_dtype=torch.bfloat16,
                _attn_implementation="eager",
            ).to("cuda")
        if save_profile["resize_embeddings"]:
            try:
                base.resize_token_embeddings(len(tokenizer))
            except Exception:
                pass
        sd = {k: v.cpu() for k, v in final_model_to_save.state_dict().items()}
        missing_unexp = base.load_state_dict(sd, strict=False)
        _info(f"State-dict loaded — missing/unexpected: {missing_unexp}")
        base.save_pretrained(MERGED_DIR, safe_serialization=save_profile["safe_serialization"])

    # Save processor (image mode) or tokenizer (text mode)
    if image_mode and processor is not None:
        processor.save_pretrained(MERGED_DIR)
        _ok("Processor saved")
    else:
        tokenizer.save_pretrained(MERGED_DIR)

    # Only write success.txt if training completed without error.
    # If there was a training error, the merged model is from a partial/crashed
    # run — do not mark it as successful or the next run will load it as a base.
    if _training_error is None:
        with open(os.path.join(MERGED_DIR, "success.txt"), "w", encoding="utf-8") as f:
            f.write("Merge complete")
        _banner("Done")
        _ok(f"Model saved to {MERGED_DIR}")
        print()
    else:
        raise _training_error


if __name__ == '__main__':
    main()

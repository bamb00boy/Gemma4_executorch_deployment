"""
Repo-local cache paths.

Import this module BEFORE `transformers`, `huggingface_hub`, `torch`, or
`torchao` — it sets the cache-related environment variables so every
download (model weights, tokenizers, datasets, torch hub artifacts) lands
inside the repo at <repo>/cache/, and exported artifacts at <repo>/models/.

The same variables are exported by scripts/run.sh so CLI tools like
`hf download` / `huggingface-cli` also honor the repo-local paths.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "cache"
MODELS_DIR = REPO_ROOT / "models"

HF_HOME = CACHE_DIR / "huggingface"
HF_HUB_CACHE = HF_HOME / "hub"
HF_DATASETS_CACHE = HF_HOME / "datasets"
TRANSFORMERS_CACHE = HF_HOME / "transformers"
TORCH_HOME = CACHE_DIR / "torch"
XDG_CACHE_HOME = CACHE_DIR / "xdg"
TRITON_CACHE_DIR = CACHE_DIR / "triton"
PIP_CACHE_DIR = CACHE_DIR / "pip"

for d in (
    CACHE_DIR, MODELS_DIR,
    HF_HOME, HF_HUB_CACHE, HF_DATASETS_CACHE, TRANSFORMERS_CACHE,
    TORCH_HOME, XDG_CACHE_HOME, TRITON_CACHE_DIR, PIP_CACHE_DIR,
):
    d.mkdir(parents=True, exist_ok=True)

# setdefault so an explicit shell export wins over this default
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(TRANSFORMERS_CACHE))
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))
os.environ.setdefault("TRITON_CACHE_DIR", str(TRITON_CACHE_DIR))
os.environ.setdefault("PIP_CACHE_DIR", str(PIP_CACHE_DIR))

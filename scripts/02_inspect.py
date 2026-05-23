"""
Phase 3 (prep): Inspect Gemma 4 E2B architecture for export planning.
Captures forward signatures, config fields, source code of novel modules,
and parameter distribution.

Usage:
    scripts/run.sh scripts/02_inspect.py | tee results/02_inspect.txt
"""

import _paths  # noqa: F401 — must come before transformers/torch imports

import inspect
import torch
from transformers import AutoModelForImageTextToText

MODEL_ID = "google/gemma-4-e2b-it"
print("Loading model on CPU (inspection only, no generation)...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.float16
).eval()


def banner(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# --- 1. Top-level model forward signature ---
banner("1. model.forward signature")
print(inspect.signature(model.forward))

# --- 2. Language model forward signature ---
banner("2. model.language_model.forward signature")
lm = model.model.language_model
print(inspect.signature(lm.forward))
print(f"\nClass: {type(lm).__name__}")
print(f"Module: {type(lm).__module__}")

# --- 3. Config: the numbers that drive export shapes ---
banner("3. Config (shape-relevant fields)")
cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
for field in [
    "hidden_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "intermediate_size",
    "vocab_size", "max_position_embeddings",
    "rope_theta", "sliding_window", "attn_logit_softcapping",
    "final_logit_softcapping", "hidden_activation",
]:
    val = getattr(cfg, field, "<not set>")
    print(f"  {field}: {val}")

# --- 4. Per-layer embedding machinery (the Gemma-4-specific bit) ---
banner("4. Per-layer embedding modules (new in Gemma 4)")
for name in [
    "embed_tokens_per_layer",
    "per_layer_model_projection",
    "per_layer_projection_norm",
]:
    mod = getattr(lm, name, None)
    if mod is None:
        print(f"  {name}: not present")
        continue
    print(f"\n  {name}:")
    print(f"    type: {type(mod).__name__}")
    print(f"    forward sig: {inspect.signature(mod.forward)}")
    params = sum(p.numel() for p in mod.parameters())
    print(f"    param count: {params:,}")
    # Print a peek at source if it's small
    try:
        src = inspect.getsource(mod.forward)
        if len(src) < 1500:
            print(f"    source:\n{src}")
    except Exception:
        pass

# --- 5. Rotary embedding ---
banner("5. Rotary embedding")
rope = lm.rotary_emb
print(f"  type: {type(rope).__name__}")
print(f"  forward sig: {inspect.signature(rope.forward)}")
try:
    src = inspect.getsource(rope.forward)
    if len(src) < 2000:
        print(f"  source:\n{src}")
except Exception:
    pass

# --- 6. Single decoder layer (representative) ---
banner("6. One decoder layer")
layer0 = lm.layers[0]
print(f"  type: {type(layer0).__name__}")
print(f"  forward sig: {inspect.signature(layer0.forward)}")
print(f"  submodules:")
for n, _ in layer0.named_children():
    print(f"    - {n}")
print(f"\n  total layers: {len(lm.layers)}")

# --- 7. Parameter breakdown: where do the params live? ---
banner("7. Parameter distribution")
total = 0
for name, child in model.model.named_children():
    p = sum(x.numel() for x in child.parameters())
    total += p
    print(f"  model.{name}: {p:,}")
lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
print(f"  lm_head: {lm_head_params:,}")
total += lm_head_params
print(f"  ---")
print(f"  TOTAL: {total:,}")

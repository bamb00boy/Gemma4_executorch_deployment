"""
Phase 4: INT4 quantization + export with EXTERNALIZED KV cache.

Single .pt2 (seq=1, mask_len dynamic). The exported program is stateless:
cache tensors are inputs, not internal buffers. The runner (Mac or Pi)
allocates cache tensors once and threads them across every call.
Same .pt2 handles both prompt token-by-token feed and generation —
no two-program prefill/decode state-sync problem.

See RESULTS.md "Phase 6 — cache externalization" for why we abandoned
the earlier prefill/decode split.

Quant scheme: Int8DynamicActivationIntxWeightConfig(weight_dtype=int4,
PerGroup(128)) — same as the old flow. Only the wrapper differs.

Usage:
    scripts/run.sh scripts/04_quantize.py 2>&1 | tee results/04_quantize.log
"""

import _paths  # noqa: F401 — must come before transformers/torch imports

import argparse
import gc
import os
import zipfile
import traceback

import torch
import torch.nn as nn
from torch.export import export, Dim
from torchao.quantization import (
    quantize_,
    Int8DynamicActivationIntxWeightConfig,
)
from torchao.quantization.granularity import PerGroup

from _int8_embedding import Int8Embedding, replace_embedding
from transformers import AutoModelForImageTextToText, AutoProcessor

from _wrapper import TextWrapperExternal, MAX_SEQ
from _external_cache import compute_layer_specs, allocate_cache_tensors

MODEL_ID = "google/gemma-4-e2b-it"
DEVICE = "cpu"
# FP32 — temporarily restored to isolate a Phase 5 lowering failure that
# appears only with BF16. (BF16 + INT4 Linears + Int8Embedding -> torchao
# ::quantize_affine ops end up in portable, no out-variants -> fail.)
# Will re-attempt BF16 after diagnosing.
DTYPE = torch.float32
GROUP_SIZE = 128
PROMPT = "The capital of France is"
N_TOKENS = 10
MASK_MAX = MAX_SEQ - 1  # 511
PT2_PATH = str(_paths.MODELS_DIR / "gemma4_e2b_text_int4_extcache.pt2")

# Cached FP32 reference (avoid the ~5 min FP32 generation that nearly OOM'd v4)
FP32_REFERENCE_IDS = [818, 5279, 529, 7001, 563, 5213, 50429, 84750, 106]
FP32_REFERENCE_TEXT = "The capital of France is **Paris**."


def gen(model, processor, n_tokens):
    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=n_tokens, do_sample=False, use_cache=True,
        )
    new_ids = out[0, inputs["input_ids"].shape[1]:].tolist()
    return new_ids, processor.decode(new_ids, skip_special_tokens=True)


def model_size_bytes(model):
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total


def export_ext_cache(wrapper, layer_specs):
    """Export TextWrapperExternal with external cache tensors as inputs."""
    k_caches, v_caches, cumlen_caches = allocate_cache_tensors(layer_specs)

    # seq=1 (static), mask_len dynamic. Use a real-ish example for mask_len
    # (5 here; any value in [2, MAX_SEQ-1] works since mask_len is dynamic).
    example_mask_len = 5
    example_inputs = (
        torch.zeros(1, 1, dtype=torch.long),                              # input_ids
        torch.ones(1, example_mask_len, dtype=torch.long),                # attention_mask
        torch.zeros(1, 1, dtype=torch.long),                              # position_ids
        torch.zeros(1, dtype=torch.long),                                 # cache_position
        k_caches, v_caches, cumlen_caches,                                # cache tensors as lists
    )

    # Quick eager sanity before export
    print("  [export] eager sanity check (seq=1)...")
    with torch.no_grad():
        out = wrapper(*example_inputs)
    print(f"  [export] eager logits shape: {tuple(out[0].shape)}")

    mask_dim = Dim("mask_len", min=2, max=MASK_MAX)
    dynamic_shapes = {
        "input_ids":      None,
        "attention_mask": {1: mask_dim},
        "position_ids":   None,
        "cache_position": None,
        "k_caches":       [None] * len(layer_specs),
        "v_caches":       [None] * len(layer_specs),
        "cumlen_caches":  [None] * len(layer_specs),
    }

    print("  [export] tracing...")
    ep = export(wrapper, example_inputs, dynamic_shapes=dynamic_shapes)
    print(f"  [export] OK. user inputs: {len(ep.graph_signature.user_inputs)}, "
          f"user outputs: {len(ep.graph_signature.user_outputs)}")
    print(f"  [export] mutated user inputs: {len(ep.graph_signature.user_inputs_to_mutate)}")

    with open(PT2_PATH, "wb") as f:
        torch.export.save(ep, f)
    assert zipfile.is_zipfile(PT2_PATH), "saved .pt2 is not a valid zip"
    _ = torch.export.load(PT2_PATH)  # round-trip
    size_gb = os.path.getsize(PT2_PATH) / 1e9
    print(f"  [export] saved: {PT2_PATH} ({size_gb:.2f} GB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-fp32-baseline", action="store_true",
        help="Also run FP32 model.generate in-process (slow, ~5 min on CPU).",
    )
    args = parser.parse_args()

    print(f"Loading model ({DEVICE}, {DTYPE})...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    fp_size = model_size_bytes(model)
    print(f"  FP32 model footprint: {fp_size / 1e9:.2f} GB")

    if args.with_fp32_baseline:
        print(f"\n[FP32 baseline] generating {N_TOKENS} tokens on prompt: {PROMPT!r}")
        fp_ids, fp_text = gen(model, processor, N_TOKENS)
    else:
        print(f"\n[FP32 baseline] SKIPPED — using cached reference")
        fp_ids, fp_text = FP32_REFERENCE_IDS, FP32_REFERENCE_TEXT
    print(f"  ids:  {fp_ids}")
    print(f"  text: {fp_text!r}")

    print(f"\n[quantize-linears] Int8DynamicActivationIntxWeightConfig(weight=int4, "
          f"PerGroup({GROUP_SIZE})) on all nn.Linear...")
    quantize_(
        model,
        Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(group_size=GROUP_SIZE),
        ),
        filter_fn=lambda m, fqn: isinstance(m, nn.Linear),
    )
    gc.collect()
    after_lin_size = model_size_bytes(model)
    print(f"  after Linear INT4 quant: {after_lin_size / 1e9:.2f} GB")

    # Also quantize embeddings to INT8. embed_tokens_per_layer alone is
    # 2.35 B params — at FP32 it dominated the .pte (9.4 GB). INT8 brings
    # that to 2.35 GB and is conservative enough that quality should hold.
    # Two embeddings in the text path:
    #   embed_tokens             (262144 x 1536) — 0.8 GB BF16
    #   embed_tokens_per_layer   (262144 x 8960) — 2.35 B params, 4.7 GB BF16
    #
    # Only embed_tokens_per_layer is worth quantizing for size (5×). The
    # small one CANNOT be quantized — Gemma 4's model code does direct 2D
    # weight slicing on it (modeling_gemma4.py:2219:
    #   `pad_embedding = self.language_model.embed_tokens.weight[pad_token_id, :]`
    # ), incompatible with any quantized tensor wrapper.
    #
    # For embed_tokens_per_layer: torchao's Int8WeightOnlyConfig works in
    # eager but lowers to torchao::quantize_affine / dequantize_affine,
    # which have no ExecuTorch out-variants -> to_executorch() FAILS.
    # IntxUnpackedToInt8Tensor on the .weight has the same problem.
    # So we replace the entire Embedding module with `Int8Embedding`
    # (scripts/_int8_embedding.py) which uses only standard aten ops
    # (index_select + cast + mul) — all have out-variants, all lower
    # cleanly through XNNPACK.
    print("[quantize-embeds] replacing embed_tokens_per_layer with hand-rolled Int8Embedding...")
    target_embeds = [
        name for name, m in model.named_modules()
        if isinstance(m, nn.Embedding) and name.endswith("embed_tokens_per_layer")
    ]
    for fqn in target_embeds:
        # navigate to the embed (need to skip the leading 'model.' for the
        # top-level Gemma4ForConditionalGeneration -> model attribute hop)
        parts = fqn.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        old = getattr(parent, parts[-1])
        new = Int8Embedding.from_embedding(old, compute_dtype=DTYPE)
        setattr(parent, parts[-1], new)
        print(f"  {fqn}: nn.Embedding -> Int8Embedding "
              f"(vocab={new.num_embeddings}, hidden={new.embedding_dim}, "
              f"INT8 weight + per-row {DTYPE} scale)")
    gc.collect()
    int4_size = model_size_bytes(model)
    print(f"  final quantized model footprint: {int4_size / 1e9:.2f} GB "
          f"(unpacked INT4-as-INT8 + INT8 embeds + BF16 norms/scales)")

    print(f"\n[INT4] generating {N_TOKENS} tokens (sanity check)...")
    int4_ids, int4_text = gen(model, processor, N_TOKENS)
    print(f"  ids:  {int4_ids}")
    print(f"  text: {int4_text!r}")
    gc.collect()

    print("\n" + "=" * 70)
    print("  QUANTIZATION QUALITY")
    print("=" * 70)
    compare_len = min(len(fp_ids), len(int4_ids))
    n_match = sum(a == b for a, b in zip(fp_ids[:compare_len], int4_ids[:compare_len]))
    print(f"  prefix token match: {n_match}/{compare_len}")
    print(f"  FP32 text: {fp_text!r}")
    print(f"  INT4 text: {int4_text!r}")
    text_match = fp_text == int4_text
    print("=" * 70)

    # Build wrapper + export
    print("\n[export] building TextWrapperExternal + cache layout...")
    layer_specs = compute_layer_specs(
        model, max_batch_size=1, max_cache_len=MAX_SEQ, dtype=DTYPE, device=DEVICE,
    )
    print(f"  cache layers: {len(layer_specs)} "
          f"(sliding: {sum(s['is_sliding'] for s in layer_specs)}, "
          f"full: {sum(not s['is_sliding'] for s in layer_specs)})")
    wrapper = TextWrapperExternal(model, layer_specs).eval()

    print(f"\n[export] external-cache .pt2 (seq=1 static, mask_len dynamic [2, {MASK_MAX}])...")
    try:
        export_ext_cache(wrapper, layer_specs)
        export_ok = True
    except Exception as e:
        print(f"  [export] FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        export_ok = False

    print("\n" + "=" * 70)
    print("  PHASE 4 (external cache) SUMMARY")
    print("=" * 70)
    print(f"  INT4 vs FP32 text match: {'PASS' if text_match else 'FAIL'}")
    print(f"  External-cache .pt2 export: {'PASS' if export_ok else 'FAIL'}")
    print("=" * 70)

    raise SystemExit(0 if (text_match and export_ok) else 1)


if __name__ == "__main__":
    main()

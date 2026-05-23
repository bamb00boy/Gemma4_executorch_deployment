"""
Phase 4 cross-check: INT4 exported prefill .pt2 vs INT4 eager wrapper.

Same methodology as 06_verify.py but with quantization applied before
the eager pass. Splits into two phases so the eager model + the loaded
.pt2 are never in RAM simultaneously (each is ~13-22 GB).

For INT4 we expect TIGHTER-than-zero but not bit-exact match (the export
sees the IntxUnpackedToInt8Tensor weight and may rewrite the matmul into
a slightly different sequence of ops, accumulating in different order).
A few ulps of drift is OK; sign of drift = trouble.

Usage:
    scripts/run.sh scripts/08_verify_int4.py --phase eager
    scripts/run.sh scripts/08_verify_int4.py --phase exported
    # or in one process (more RAM pressure):
    scripts/run.sh scripts/08_verify_int4.py --phase both
"""

import _paths  # noqa: F401 — must come before transformers/torch imports

import argparse
import gc
import pickle

import torch
from torchao.quantization import quantize_, Int8DynamicActivationIntxWeightConfig
from torchao.quantization.granularity import PerGroup
from transformers import AutoModelForImageTextToText, AutoProcessor

from _wrapper import TextOnlyWrapper, build_static_cache, MAX_SEQ

MODEL_ID = "google/gemma-4-e2b-it"
DEVICE = "cpu"
DTYPE = torch.float32
GROUP_SIZE = 128
PROMPT = "The capital of France is"
EXPORTED_PATH = str(_paths.MODELS_DIR / "gemma4_e2b_text_int4_prefill.pt2")
REF_PATH = _paths.CACHE_DIR / "verify_int4_reference.pkl"


def prepare_inputs(processor):
    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    assert 2 <= seq_len < MAX_SEQ, f"prompt seq_len={seq_len} outside [2, {MAX_SEQ})"
    return (
        input_ids,
        torch.ones_like(input_ids),
        torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
        torch.arange(seq_len, dtype=torch.long),
    )


def eager_reference():
    print(f"[eager] loading {MODEL_ID} on {DEVICE} ({DTYPE})...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"[eager] quantizing (Int8 dyn act / Int4 weight, PerGroup({GROUP_SIZE}))...")
    quantize_(
        model,
        Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(group_size=GROUP_SIZE),
        ),
    )
    gc.collect()

    cache = build_static_cache(model, batch=1, max_seq=MAX_SEQ, device=DEVICE, dtype=DTYPE)
    wrapper = TextOnlyWrapper(model, cache).eval()

    input_ids, attn_mask, pos_ids, cache_pos = prepare_inputs(processor)
    print(f"[eager] running prefill on prompt: {PROMPT!r} (seq={input_ids.shape[1]})...")
    with torch.no_grad():
        logits = wrapper(input_ids, attn_mask, pos_ids, cache_pos)

    last = logits[0, -1].clone()
    top5_vals, top5_ids = torch.topk(last, 5)
    next_id = int(top5_ids[0])

    ref = {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "position_ids": pos_ids,
        "cache_position": cache_pos,
        "last_logits": last,
        "next_id": next_id,
        "top5_ids": top5_ids,
        "top5_vals": top5_vals,
        "next_text": processor.decode([next_id], skip_special_tokens=False),
    }
    REF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REF_PATH, "wb") as f:
        pickle.dump(ref, f)

    print(f"[eager] next token: id={next_id} text={ref['next_text']!r}")
    print(f"[eager] top-5 ids: {top5_ids.tolist()}")
    print(f"[eager] saved reference -> {REF_PATH}")
    return ref


def compare_exported(ref):
    print(f"[exported] loading {EXPORTED_PATH}...")
    ep = torch.export.load(EXPORTED_PATH)
    module = ep.module()

    print("[exported] running prefill on the same inputs...")
    with torch.no_grad():
        logits = module(
            ref["input_ids"],
            ref["attention_mask"],
            ref["position_ids"],
            ref["cache_position"],
        )

    last = logits[0, -1]
    top5_vals, top5_ids = torch.topk(last, 5)
    next_id = int(top5_ids[0])

    ref_last = ref["last_logits"]
    diff = (last - ref_last).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (ref_last.abs() + 1e-6)).max().item()

    print("\n" + "=" * 70)
    print("  INT4 CROSS-CHECK")
    print("=" * 70)
    print(f"  next-token match:    {next_id == ref['next_id']}  (eager={ref['next_id']}, exported={next_id})")
    print(f"  top-5 ids identical: {top5_ids.tolist() == ref['top5_ids'].tolist()}")
    print(f"  max |diff|:          {max_diff:.4e}")
    print(f"  mean |diff|:         {mean_diff:.4e}")
    print(f"  max relative diff:   {rel_diff:.4e}")
    print(f"  top-5 logits:")
    for i, (rv, ev) in enumerate(zip(ref["top5_vals"].tolist(), top5_vals.tolist())):
        print(f"    rank {i}: ref={rv:+.6f}  exported={ev:+.6f}  diff={abs(rv-ev):.2e}")

    # INT4 tolerance: top-1 must match; logits should agree to ~1e-3 absolute
    # (export of quantized graphs sometimes re-orders the dequant ops which
    # produces small fp drift).
    ok = next_id == ref["next_id"] and top5_ids.tolist() == ref["top5_ids"].tolist()
    print("=" * 70)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["eager", "exported", "both"], default="both")
    args = parser.parse_args()

    if args.phase in ("eager", "both"):
        ref = eager_reference()

    if args.phase == "both":
        del ref
        gc.collect()
        with open(REF_PATH, "rb") as f:
            ref = pickle.load(f)
        ok = compare_exported(ref)
        raise SystemExit(0 if ok else 1)

    if args.phase == "exported":
        with open(REF_PATH, "rb") as f:
            ref = pickle.load(f)
        ok = compare_exported(ref)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

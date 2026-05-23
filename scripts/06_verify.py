"""
Phase 3 cross-check: numerical equivalence of the exported FP32 prefill
graph against the eager wrapper.

NOTE: `models/gemma4_e2b_text_fp32.pt2` is deleted in Phase 4 prep
(to free disk for the INT4 prefill+decode .pt2s). To rerun this script
you must first regenerate the FP32 prefill via `03_export.py`. For the
current INT4 artifacts, use `08_verify_int4.py` instead — same
methodology, different reference path.

Runs in two phases to keep memory in budget (Mac has only ~25 GB usable
for an FP32 5.5B model + 18.5 GB .pt2 — too tight to hold both at once):

  --phase eager      Load model + run wrapper eager. Dump reference
                     (last-token logits, next token id, top-5) to disk.
  --phase exported   Load the .pt2. Run on the same inputs. Compare.
  --phase both       Run eager then exported in one process, with an
                     explicit gc.collect() in between. Convenient for
                     small machines that have swap available.

Usage:
    scripts/run.sh scripts/06_verify.py --phase both
    # or two passes:
    scripts/run.sh scripts/06_verify.py --phase eager
    scripts/run.sh scripts/06_verify.py --phase exported
"""

import _paths  # noqa: F401 — must come before transformers/torch imports

import argparse
import gc
import pickle

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.cache_utils import StaticCache

MODEL_ID = "google/gemma-4-e2b-it"
DEVICE = "cpu"
DTYPE = torch.float32
MAX_SEQ = 512
EXPORTED_PATH = _paths.MODELS_DIR / "gemma4_e2b_text_fp32.pt2"
REF_PATH = _paths.CACHE_DIR / "verify_reference.pkl"

# Same prompt as the smoke test for cross-reference
PROMPT = "The capital of France is"


def prepare_inputs():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    assert 2 <= seq_len < MAX_SEQ, f"prompt seq_len={seq_len} outside [2, {MAX_SEQ})"
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cache_position = torch.arange(seq_len, dtype=torch.long)
    return processor, input_ids, attention_mask, position_ids, cache_position


def eager_reference():
    print(f"[eager] loading model on {DEVICE} ({DTYPE})...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
    processor, input_ids, attention_mask, position_ids, cache_position = prepare_inputs()

    cfg = model.config.text_config
    cache = StaticCache(
        config=cfg, max_batch_size=1, max_cache_len=MAX_SEQ,
        device=DEVICE, dtype=DTYPE,
    )

    print(f"[eager] running prefill on prompt: {PROMPT!r} (seq={input_ids.shape[1]})...")
    with torch.no_grad():
        out = model.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
        )
        logits = model.lm_head(out.last_hidden_state)
        cap = getattr(cfg, "final_logit_softcapping", None)
        if cap:
            logits = torch.tanh(logits / cap) * cap

    last = logits[0, -1].clone()  # [vocab]
    top5_vals, top5_ids = torch.topk(last, 5)
    next_id = int(top5_ids[0])

    ref = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "cache_position": cache_position,
        "last_logits": last,
        "next_id": next_id,
        "top5_ids": top5_ids,
        "top5_vals": top5_vals,
        "tokenizer_decode_next": processor.decode([next_id], skip_special_tokens=False),
    }
    REF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REF_PATH, "wb") as f:
        pickle.dump(ref, f)

    print(f"[eager] next token: id={next_id} text={ref['tokenizer_decode_next']!r}")
    print(f"[eager] top-5 ids: {top5_ids.tolist()}")
    print(f"[eager] saved reference -> {REF_PATH}")
    return ref


def compare_exported(ref):
    if not EXPORTED_PATH.exists():
        raise SystemExit(
            f"ERROR: {EXPORTED_PATH} not found.\n"
            "The FP32 prefill .pt2 was deleted in Phase 4 prep to free disk.\n"
            "To recreate it: scripts/run.sh scripts/03_export.py\n"
            "For the current INT4 artifacts: scripts/run.sh scripts/08_verify_int4.py"
        )
    print(f"[exported] loading {EXPORTED_PATH}...")
    ep = torch.export.load(str(EXPORTED_PATH))
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

    print(f"[exported] next token: id={next_id}")
    print(f"[exported] top-5 ids: {top5_ids.tolist()}")

    # Numerical comparison
    ref_last = ref["last_logits"]
    diff = (last - ref_last).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (ref_last.abs() + 1e-6)).max().item()

    print("\n" + "=" * 70)
    print("  CROSS-CHECK")
    print("=" * 70)
    print(f"  next-token match:    {next_id == ref['next_id']}  (eager={ref['next_id']}, exported={next_id})")
    print(f"  top-5 ids identical: {top5_ids.tolist() == ref['top5_ids'].tolist()}")
    print(f"  max |diff|:          {max_diff:.4e}")
    print(f"  mean |diff|:         {mean_diff:.4e}")
    print(f"  max relative diff:   {rel_diff:.4e}")
    print(f"  top-5 logits:")
    for i, (rv, ev) in enumerate(zip(ref["top5_vals"].tolist(), top5_vals.tolist())):
        print(f"    rank {i}: ref={rv:+.6f}  exported={ev:+.6f}  diff={abs(rv-ev):.2e}")

    ok = next_id == ref["next_id"] and max_diff < 1e-3
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
        # Free eager model before loading the .pt2 (memory pressure)
        # Reload from disk in the exported phase to ensure clean state.
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

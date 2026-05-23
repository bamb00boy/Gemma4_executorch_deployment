"""
Phase 5 cross-check: load the .pte via the ExecuTorch Python runtime
and run a single prefill on the same prompt we've used everywhere else.

Output token should match the FP32/INT4 reference (id=818 "The").

This is the .pte analog of 06_verify / 08_verify_int4 — but we no longer
have the eager wrapper accessible here (the .pte stands alone). So we
compare against the cached reference token from Phase 3+4.

Usage:
    scripts/run.sh scripts/09_verify_pte.py
"""

import _paths  # noqa: F401 — must come before transformers/torch imports

import torch
from transformers import AutoProcessor
from executorch.runtime import Runtime, Verification

MODEL_ID = "google/gemma-4-e2b-it"
PROMPT = "The capital of France is"
PREFILL_PTE = str(_paths.MODELS_DIR / "gemma4_e2b_text_int4_prefill.pte")
REFERENCE_NEXT_ID = 818  # "The" — captured in Phase 3+4 cross-checks


def main():
    print(f"Loading processor + tokenizing prompt: {PROMPT!r}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    print(f"  prompt seq_len = {seq_len}")
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cache_position = torch.arange(seq_len, dtype=torch.long)

    print(f"\nLoading {PREFILL_PTE}...")
    rt = Runtime.get()
    program = rt.load_program(PREFILL_PTE, verification=Verification.Minimal)
    method = program.load_method("forward")
    meta = method.metadata
    n_inputs = meta.num_inputs() if callable(meta.num_inputs) else meta.num_inputs
    n_outputs = meta.num_outputs() if callable(meta.num_outputs) else meta.num_outputs
    name = meta.name() if callable(meta.name) else meta.name
    print(f"  method loaded — name={name}, inputs={n_inputs}, outputs={n_outputs}")
    # Repr the full metadata (TensorInfo strings include sizes/dtype)
    print(f"  metadata: {meta!r}"[:500] + "...")
    # Stash the expected input shape (.pte specialized to dim upper-bound)
    pte_seq = None
    try:
        pte_seq = int(repr(meta).split("sizes=[1, ")[1].split("]")[0])
        print(f"  .pte input seq_len: {pte_seq}")
    except Exception:
        pass

    # Pad inputs to the .pte's specialized seq length if needed.
    # ExecuTorch's memory planner allocated buffers for the worst case
    # (upper bound of the Dim), so runtime needs that exact shape.
    target_seq = pte_seq or seq_len
    if target_seq != seq_len:
        print(f"  padding inputs from seq={seq_len} to seq={target_seq}...")
        pad = target_seq - seq_len
        input_ids_p = torch.cat([input_ids, torch.zeros(1, pad, dtype=torch.long)], dim=1)
        attention_mask_p = torch.cat([attention_mask, torch.zeros(1, pad, dtype=torch.long)], dim=1)
        position_ids_p = torch.arange(target_seq, dtype=torch.long).unsqueeze(0)
        cache_position_p = torch.arange(target_seq, dtype=torch.long)
    else:
        input_ids_p = input_ids
        attention_mask_p = attention_mask
        position_ids_p = position_ids
        cache_position_p = cache_position

    print(f"\nRunning forward (seq={target_seq})...")
    outputs = method.execute((input_ids_p, attention_mask_p, position_ids_p, cache_position_p))
    logits = outputs[0]
    print(f"  logits shape: {tuple(logits.shape)}")

    # The valid logits we care about are at position seq_len-1 (last real token)
    last = logits[0, seq_len - 1]
    top5_vals, top5_ids = torch.topk(last, 5)
    next_id = int(top5_ids[0])
    next_text = processor.decode([next_id], skip_special_tokens=False)

    print(f"\n  next token: id={next_id} text={next_text!r}")
    print(f"  top-5 ids: {top5_ids.tolist()}")
    print()
    print("=" * 70)
    print("  .pte CROSS-CHECK")
    print("=" * 70)
    ok = next_id == REFERENCE_NEXT_ID
    print(f"  next-token vs reference: {'PASS' if ok else 'FAIL'}  "
          f"(.pte={next_id}, ref={REFERENCE_NEXT_ID})")
    print("=" * 70)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

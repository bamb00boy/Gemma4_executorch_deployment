"""
Phase 6 prep: can we use decode.pte alone for both prompt prefill and
generation (token-at-a-time)?

This matters for the Pi deployment plan. If yes, we ship one .pte
(~12 GB) instead of two (~24 GB). If no, we need a wrapper redesign
(externalize cache state) or accept hybrid eager/.pte runner.

The catch: decode.pte was exported with a 4-token seed prefill, so
its saved cumulative_length is 4 and cache slots 0-3 have garbage.
First real call would write to slot 4, not slot 0.

Usage:
    scripts/run.sh scripts/10_decode_only_check.py
"""

import _paths  # noqa: F401

import torch
from transformers import AutoProcessor
from executorch.runtime import Runtime, Verification

PROMPT = "The capital of France is"
MODEL_ID = "google/gemma-4-e2b-it"
DECODE_PTE = str(_paths.MODELS_DIR / "gemma4_e2b_text_int4_decode.pte")
MASK_LEN = 511  # decode.pte's specialized mask shape
EXPECTED = [818, 5279, 529, 7001, 563, 5213, 50429, 84750, 106]  # FP32/INT4 ref


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    enc = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    prompt_ids = enc["input_ids"][0].tolist()
    print(f"prompt: {PROMPT!r} ({len(prompt_ids)} tokens)")

    print(f"\nLoading {DECODE_PTE}...")
    rt = Runtime.get()
    prog = rt.load_program(DECODE_PTE, verification=Verification.Minimal)
    decode = prog.load_method("forward")

    # Token-by-token: feed every prompt token, then 9 generation steps
    all_tokens = list(prompt_ids)
    generated = []
    print(f"\nFeeding {len(prompt_ids)} prompt tokens + decoding 9 more...")
    print("(if .pte's saved cumulative_length is non-zero, expect garbage)")

    for step in range(len(prompt_ids) + 9):
        if step < len(prompt_ids):
            tok = prompt_ids[step]
            note = f"prompt[{step}]={tok}"
        else:
            tok = generated[-1] if generated else all_tokens[-1]
            note = f"gen[{step - len(prompt_ids)}]={tok}"

        pos = step  # external position
        input_ids = torch.tensor([[tok]], dtype=torch.long)
        attention_mask = torch.zeros(1, MASK_LEN, dtype=torch.long)
        attention_mask[:, :pos + 1] = 1
        position_ids = torch.tensor([[pos]], dtype=torch.long)
        cache_position = torch.tensor([pos], dtype=torch.long)

        out = decode.execute((input_ids, attention_mask, position_ids, cache_position))
        logits = out[0]
        next_id = int(logits[0, -1].argmax())

        if step >= len(prompt_ids) - 1:
            # After feeding the LAST prompt token, logits give next-token prediction
            generated.append(next_id)
            print(f"  step {step:2d} {note:25s} -> next={next_id} ({processor.decode([next_id])!r})")
        else:
            print(f"  step {step:2d} {note:25s} -> (filling cache)")

    print(f"\nGenerated ids:   {generated[:len(EXPECTED)]}")
    print(f"Reference ids:   {EXPECTED}")
    text = processor.decode(generated, skip_special_tokens=True)
    print(f"Generated text:  {text!r}")
    print(f"Expected text:   'The capital of France is **Paris**.'")
    print()
    matches = sum(1 for a, b in zip(generated, EXPECTED) if a == b)
    print(f"Match: {matches}/{len(EXPECTED)} tokens")
    print(f"VERDICT: {'PASS — single .pte deploy works' if matches == len(EXPECTED) else 'FAIL — need cache externalization or both .pte'}")


if __name__ == "__main__":
    main()

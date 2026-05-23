"""
Phase 6 prep: verify TextWrapperExternal works token-by-token in eager
mode before we re-export it through Phase 4 + Phase 5.

Same prompt + reference as everywhere else: "The capital of France is"
should produce id-sequence [818, 5279, 529, 7001, 563, 5213, 50429, 84750, 106].

Both prefill (one token at a time) and decode use the same wrapper call.

Usage:
    scripts/run.sh scripts/11_external_cache_check.py
"""

import _paths  # noqa: F401

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from _wrapper import TextWrapperExternal, MAX_SEQ
from _external_cache import compute_layer_specs, allocate_cache_tensors

MODEL_ID = "google/gemma-4-e2b-it"
PROMPT = "The capital of France is"
DEVICE = "cpu"
DTYPE = torch.float32
N_NEW = 9
EXPECTED = [818, 5279, 529, 7001, 563, 5213, 50429, 84750, 106]
MASK_LEN = MAX_SEQ - 1  # 511


def main():
    print(f"Loading model ({DEVICE}, {DTYPE})...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    layer_specs = compute_layer_specs(model, max_batch_size=1, max_cache_len=MAX_SEQ, dtype=DTYPE, device=DEVICE)
    print(f"  layer_specs: {len(layer_specs)} cache layers "
          f"(sliding: {sum(s['is_sliding'] for s in layer_specs)}, full: {sum(not s['is_sliding'] for s in layer_specs)})")

    k_caches, v_caches, cumlen_caches = allocate_cache_tensors(layer_specs)
    print(f"  k_caches[0] shape: {tuple(k_caches[0].shape)}, dtype: {k_caches[0].dtype}")

    wrapper = TextWrapperExternal(model, layer_specs).eval()

    # Tokenize prompt
    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    enc = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    prompt_ids = enc["input_ids"][0].tolist()
    print(f"\nprompt: {PROMPT!r} ({len(prompt_ids)} tokens)")

    # Token-by-token: feed every prompt token, then N_NEW generation steps
    generated = []
    last_token = None
    print(f"\nFeeding {len(prompt_ids)} prompt tokens + decoding {N_NEW} more...")
    for step in range(len(prompt_ids) + N_NEW):
        if step < len(prompt_ids):
            tok = prompt_ids[step]
            phase = f"prompt[{step:2d}]"
        else:
            tok = last_token
            phase = f"gen[{step - len(prompt_ids):2d}]   "

        pos = step
        input_ids = torch.tensor([[tok]], dtype=torch.long)
        attention_mask = torch.zeros(1, MASK_LEN, dtype=torch.long)
        attention_mask[:, :pos + 1] = 1
        position_ids = torch.tensor([[pos]], dtype=torch.long)
        cache_position = torch.tensor([pos], dtype=torch.long)

        with torch.no_grad():
            logits, k_caches, v_caches, cumlen_caches = wrapper(
                input_ids, attention_mask, position_ids, cache_position,
                k_caches, v_caches, cumlen_caches,
            )
        last_token = int(logits[0, -1].argmax())
        if step >= len(prompt_ids) - 1:
            generated.append(last_token)
            text = processor.decode([last_token])
            print(f"  step {step:2d} {phase} tok={tok:6d} -> next={last_token} ({text!r})")
        else:
            print(f"  step {step:2d} {phase} tok={tok:6d} -> (filling cache)")

    print(f"\nGenerated ids:   {generated[:len(EXPECTED)]}")
    print(f"Reference ids:   {EXPECTED}")
    text = processor.decode(generated, skip_special_tokens=True)
    print(f"Generated text:  {text!r}")
    matches = sum(1 for a, b in zip(generated, EXPECTED) if a == b)
    print(f"\nMatch: {matches}/{len(EXPECTED)} tokens")
    print(f"VERDICT: {'PASS' if matches == len(EXPECTED) else 'FAIL'}")
    raise SystemExit(0 if matches == len(EXPECTED) else 1)


if __name__ == "__main__":
    main()

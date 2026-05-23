"""
Phase 3 cross-check: prefill + decode round-trip in eager + try seq=1 export.

What this verifies:
  1. The TextOnlyWrapper drives an end-to-end greedy generation correctly:
     wrapper.forward(prefill, seq=14) → next token →
     wrapper.forward(decode, seq=1) × N → matches model.generate(N+1).
  2. torch.export accepts a seq=1 input (no dynamic dim) without firing
     the `seq >= 2` guard. We don't save the .pt2 (disk-tight; we'll save
     both prefill and decode .pt2s after INT4 quantization in Phase 4).

Usage:
    scripts/run.sh scripts/07_decode_check.py 2>&1 | tee results/07_decode_check.log
"""

import _paths  # noqa: F401 — must come before transformers/torch imports

import torch
from torch.export import export
from transformers import AutoModelForImageTextToText, AutoProcessor

from _wrapper import TextOnlyWrapper, build_static_cache, MAX_SEQ

MODEL_ID = "google/gemma-4-e2b-it"
DEVICE = "cpu"
DTYPE = torch.float32
PROMPT = "The capital of France is"
N_NEW_TOKENS = 10  # generate this many tokens past the prompt


def tokenize_prompt(processor):
    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    return inputs["input_ids"], int(inputs["input_ids"].shape[1])


def wrapper_greedy(wrapper, processor, input_ids, prompt_len, n_new):
    """Run wrapper prefill + decode loop, return generated token ids."""
    # Reset the cache before generation
    wrapper.cache.reset()

    # Prefill
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(prompt_len, dtype=torch.long).unsqueeze(0)
    cache_position = torch.arange(prompt_len, dtype=torch.long)
    with torch.no_grad():
        logits = wrapper(input_ids, attention_mask, position_ids, cache_position)
    next_id = int(logits[0, -1].argmax())
    generated = [next_id]
    print(f"  prefill -> next token: id={next_id} text={processor.decode([next_id])!r}")

    # Decode loop (seq=1 each step). Pass attention_mask of shape
    # [batch, cache_position+1] so the new query attends to all valid
    # cache slots, not just itself.
    for step in range(n_new - 1):
        pos = prompt_len + step  # position of the token we just produced
        new_input = torch.tensor([[next_id]], dtype=torch.long)
        # mask covers all populated cache slots + the new token's slot
        new_mask = torch.ones((1, pos + 1), dtype=torch.long)
        new_position = torch.tensor([[pos]], dtype=torch.long)
        new_cache_pos = torch.tensor([pos], dtype=torch.long)
        with torch.no_grad():
            logits = wrapper(new_input, new_mask, new_position, new_cache_pos)
        next_id = int(logits[0, -1].argmax())
        generated.append(next_id)
        print(f"  decode step {step+1} (cache_pos={pos}) -> id={next_id} text={processor.decode([next_id])!r}")
    return generated


def reference_generate(model, processor, input_ids, n_new):
    """Run model.generate for reference."""
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=n_new,
            do_sample=False,
            use_cache=True,
        )
    # out shape: [1, prompt_len + n_new]
    new_ids = out[0, input_ids.shape[1]:].tolist()
    return new_ids


def attempt_decode_export(wrapper):
    """Try to export the wrapper with seq=1 (no dynamic dim). In-process, no save."""
    wrapper.cache.reset()
    # Seed the cache by running a prefill first, so the decode export
    # sees realistic cache state (mostly zero but with valid layout).
    seed_len = 4
    seed_in = torch.ones(1, seed_len, dtype=torch.long)
    seed_mask = torch.ones(1, seed_len, dtype=torch.long)
    seed_pos = torch.arange(seed_len, dtype=torch.long).unsqueeze(0)
    seed_cpos = torch.arange(seed_len, dtype=torch.long)
    with torch.no_grad():
        _ = wrapper(seed_in, seed_mask, seed_pos, seed_cpos)

    # Decode example
    decode_in = torch.tensor([[42]], dtype=torch.long)
    decode_mask = torch.ones((1, 1), dtype=torch.long)
    decode_pos = torch.tensor([[seed_len]], dtype=torch.long)
    decode_cpos = torch.tensor([seed_len], dtype=torch.long)
    example = (decode_in, decode_mask, decode_pos, decode_cpos)

    print("\n[decode-export] attempting torch.export with seq=1 (no Dim)...")
    try:
        ep = export(wrapper, example)  # NO dynamic_shapes -> seq specializes to 1
        print(f"[decode-export] ✓ Export SUCCEEDED")
        n_inputs = len(ep.graph_signature.user_inputs)
        n_outputs = len(ep.graph_signature.user_outputs)
        print(f"  inputs: {n_inputs}, outputs: {n_outputs}")
        return True
    except Exception as e:
        print(f"[decode-export] ✗ Export FAILED: {type(e).__name__}")
        print(f"  {e}")
        return False


def main():
    print(f"Loading model on {DEVICE} ({DTYPE})...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"\nTokenizing prompt: {PROMPT!r}")
    input_ids, prompt_len = tokenize_prompt(processor)
    print(f"  prompt seq_len = {prompt_len}")

    # --- Reference: model.generate ---
    print(f"\n[reference] running model.generate(max_new_tokens={N_NEW_TOKENS})...")
    ref_ids = reference_generate(model, processor, input_ids, N_NEW_TOKENS)
    ref_text = processor.decode(ref_ids, skip_special_tokens=True)
    print(f"[reference] generated ids: {ref_ids}")
    print(f"[reference] generated text: {ref_text!r}")

    # --- Wrapper greedy ---
    cache = build_static_cache(model, batch=1, max_seq=MAX_SEQ, device=DEVICE, dtype=DTYPE)
    wrapper = TextOnlyWrapper(model, cache).eval()
    print(f"\n[wrapper] running prefill + decode loop...")
    wrapper_ids = wrapper_greedy(wrapper, processor, input_ids, prompt_len, N_NEW_TOKENS)
    wrapper_text = processor.decode(wrapper_ids, skip_special_tokens=True)
    print(f"[wrapper] generated ids: {wrapper_ids}")
    print(f"[wrapper] generated text: {wrapper_text!r}")

    # --- Compare ---
    # model.generate stops early at EOS; wrapper has no early-stop. So
    # compare wrapper[:len(ref)] == ref. Any extra wrapper tokens past
    # ref's length are post-EOS continuations and don't count.
    print("\n" + "=" * 70)
    print("  GREEDY COMPARISON: wrapper prefill+decode vs model.generate")
    print("=" * 70)
    compare_len = len(ref_ids)
    greedy_ok = wrapper_ids[:compare_len] == ref_ids
    n_match = sum(a == b for a, b in zip(wrapper_ids[:compare_len], ref_ids))
    print(f"  ref length: {len(ref_ids)} (may be < N_NEW_TOKENS={N_NEW_TOKENS} if EOS hit)")
    print(f"  prefix match: {n_match}/{compare_len} tokens")
    for i in range(compare_len):
        marker = "✓" if wrapper_ids[i] == ref_ids[i] else "✗"
        print(f"    [{i:2d}] {marker} wrapper={wrapper_ids[i]} ref={ref_ids[i]}")
    if len(wrapper_ids) > compare_len:
        extra = wrapper_ids[compare_len:]
        print(f"  (wrapper produced {len(extra)} extra token(s) past ref EOS: {extra})")
    print("=" * 70)

    # --- Decode export attempt ---
    decode_export_ok = attempt_decode_export(wrapper)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  wrapper greedy == model.generate:  {'PASS' if greedy_ok else 'FAIL'}")
    print(f"  decode (seq=1) export traces:       {'PASS' if decode_export_ok else 'FAIL'}")
    print("=" * 70)
    raise SystemExit(0 if (greedy_ok and decode_export_ok) else 1)


if __name__ == "__main__":
    main()

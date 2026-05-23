"""
Self-contained Gemma 4 E2B INT4 runner for Raspberry Pi 5 (or any ARM64).

Loads ONE .pte (external-cache variant), tokenizes a prompt with the
Gemma chat template, runs token-by-token (prompt feed + decode) threading
KV cache tensors across calls, prints generated text + timing.

Designed to run on the Pi with NO project codebase — just the .pte,
the tokenizer files, and this script. Only Python deps:
    pip install executorch transformers

Files expected next to this script (or pass paths via flags):
    gemma4_e2b_text_int4_extcache.pte
    tokenizer/                          # dir with tokenizer.json + tokenizer_config.json + chat_template.jinja

Usage:
    python pi_runner.py "The capital of France is"
    python pi_runner.py "Why is the sky blue?" --max-new-tokens 50
    python pi_runner.py "Hello" --verify  # asserts output matches reference
"""

import argparse
import os
import time

import torch
from transformers import AutoTokenizer
from executorch.runtime import Runtime, Verification

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PTE = os.path.join(HERE, "gemma4_e2b_text_int4_extcache.pte")
DEFAULT_TOK = os.path.join(HERE, "tokenizer")

MAX_CACHE_LEN = 512
MASK_LEN = MAX_CACHE_LEN - 1  # 511; .pte specialized to this upper bound
DTYPE = torch.float32  # matches the model's quantize-time dtype

# Hardcoded cache layout for Gemma 4 E2B. Matches what
# scripts/_external_cache.py:compute_layer_specs derives from the model
# config. 35 decoder layers minus num_kv_shared_layers=20 = 15 cache layers.
# Layer-type pattern (repeats every 5): [sliding, sliding, sliding, sliding, full].
# Sliding layers: head_dim=256. Full layers: global_head_dim=512.
GEMMA4_E2B_LAYER_SHAPES = [
    # (head_dim, is_sliding)
    (256, True),  (256, True),  (256, True),  (256, True),  (512, False),  # layers 0-4
    (256, True),  (256, True),  (256, True),  (256, True),  (512, False),  # layers 5-9
    (256, True),  (256, True),  (256, True),  (256, True),  (512, False),  # layers 10-14
]
NUM_KV_HEADS = 1
BATCH = 1

# Reference for --verify mode (FP32/INT4 token-id sequence for "The capital of France is")
REFERENCE_PROMPT = "The capital of France is"
REFERENCE_IDS = [818, 5279, 529, 7001, 563, 5213, 50429, 84750, 106]
REFERENCE_TEXT = "The capital of France is **Paris**."

# Gemma 4 end-of-turn token id (model stops here in chat)
EOS_TOKEN_IDS = {106, 1, 2}  # <end_of_turn>, <eos>, <bos>-as-sentinel


def allocate_cache_tensors():
    """Allocate one set of K, V, cumulative_length tensors per cache layer."""
    k_caches, v_caches, cumlen_caches = [], [], []
    for head_dim, _is_sliding in GEMMA4_E2B_LAYER_SHAPES:
        shape = (BATCH, NUM_KV_HEADS, MAX_CACHE_LEN, head_dim)
        k_caches.append(torch.zeros(shape, dtype=DTYPE))
        v_caches.append(torch.zeros(shape, dtype=DTYPE))
        cumlen_caches.append(torch.zeros(1, dtype=torch.int64))
    return k_caches, v_caches, cumlen_caches


def step(method, token_id, pos, k_caches, v_caches, cumlen_caches):
    """One forward call: feed `token_id` at position `pos`, get logits +
    updated cache tensors back."""
    input_ids = torch.tensor([[token_id]], dtype=torch.long)
    attention_mask = torch.zeros(1, MASK_LEN, dtype=torch.long)
    attention_mask[:, :pos + 1] = 1
    position_ids = torch.tensor([[pos]], dtype=torch.long)
    cache_position = torch.tensor([pos], dtype=torch.long)

    # The .pte's execute() takes flat positional inputs.
    # Order matches the wrapper's forward signature:
    #   input_ids, attention_mask, position_ids, cache_position, *k_caches, *v_caches, *cumlen_caches
    args = (input_ids, attention_mask, position_ids, cache_position,
            *k_caches, *v_caches, *cumlen_caches)
    if os.environ.get("DEBUG_SHAPES"):
        for i, a in enumerate(args):
            if hasattr(a, "shape"):
                print(f"    arg[{i:2d}]: shape={tuple(a.shape)} dtype={a.dtype}", flush=True)
    outputs = method.execute(args)
    # The .pte emits 91 outputs:
    #   [0..14]   K mutations (auto-emitted by torch.export)
    #   [15..29]  V mutations
    #   [30..44]  cumlen mutations
    #   [45]      logits
    #   [46..60]  K (from wrapper's explicit return — same tensors)
    #   [61..75]  V (from wrapper's explicit return)
    #   [76..90]  cumlen (from wrapper's explicit return)
    # Either copy works; we use the explicit-return half because indices
    # align with the (logits, k, v, cumlen) ordering the wrapper declared.
    n = len(GEMMA4_E2B_LAYER_SHAPES)
    logits = outputs[45]
    base = 46
    k_new = list(outputs[base:base + n])
    v_new = list(outputs[base + n:base + 2 * n])
    cumlen_new = list(outputs[base + 2 * n:base + 3 * n])
    return logits, k_new, v_new, cumlen_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default=REFERENCE_PROMPT,
                        help=f"Prompt text (default: {REFERENCE_PROMPT!r})")
    parser.add_argument("--pte", default=DEFAULT_PTE, help="Path to .pte file")
    parser.add_argument("--tokenizer", default=DEFAULT_TOK, help="Path to tokenizer dir")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--verify", action="store_true",
                        help="Assert prompt+output match the reference (smoke test)")
    args = parser.parse_args()

    if args.verify:
        args.prompt = REFERENCE_PROMPT
        args.max_new_tokens = max(args.max_new_tokens, len(REFERENCE_IDS))

    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Tokenizing prompt: {args.prompt!r}")
    messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    prompt_ids = enc["input_ids"][0].tolist()
    n_prompt = len(prompt_ids)
    if n_prompt + args.max_new_tokens > MASK_LEN:
        raise SystemExit(
            f"prompt ({n_prompt}) + max_new_tokens ({args.max_new_tokens}) "
            f"exceeds mask_len ({MASK_LEN})"
        )
    print(f"  prompt_len = {n_prompt}")

    print(f"\nLoading {args.pte}...")
    t0 = time.time()
    rt = Runtime.get()
    program = rt.load_program(args.pte, verification=Verification.Minimal)
    method = program.load_method("forward")
    print(f"  loaded in {time.time() - t0:.1f}s")

    print("Allocating cache tensors...")
    k_caches, v_caches, cumlen_caches = allocate_cache_tensors()
    cache_mb = sum(t.numel() * t.element_size() for t in k_caches + v_caches) / 1e6
    print(f"  total cache size: {cache_mb:.1f} MB across {len(k_caches)} layers")

    # --- Prompt token-by-token feed ("slow prefill") ---
    print(f"\nFeeding {n_prompt} prompt tokens (token-by-token; no batched prefill in this design)...")
    t_prefill_start = time.time()
    last_logits = None
    for i, tok in enumerate(prompt_ids):
        last_logits, k_caches, v_caches, cumlen_caches = step(
            method, tok, i, k_caches, v_caches, cumlen_caches
        )
    t_prefill = time.time() - t_prefill_start
    print(f"  prompt feed: {t_prefill:.2f}s ({n_prompt} tokens, "
          f"{n_prompt / t_prefill:.2f} tok/s, ttft equivalent)")

    # Next-token prediction from last prompt position's logits
    next_id = int(last_logits[0, -1].argmax())
    generated = [next_id]
    print(f"  first generated token: id={next_id} text={tokenizer.decode([next_id])!r}")

    # --- Decode loop ---
    print(f"\nDecoding up to {args.max_new_tokens - 1} more tokens...")
    t_decode_start = time.time()
    n_decoded = 1
    for step_idx in range(args.max_new_tokens - 1):
        if next_id in EOS_TOKEN_IDS:
            print(f"  hit EOS (id={next_id}) at decode step {step_idx}")
            break
        pos = n_prompt + step_idx  # position of the token we just produced
        last_logits, k_caches, v_caches, cumlen_caches = step(
            method, next_id, pos, k_caches, v_caches, cumlen_caches
        )
        next_id = int(last_logits[0, -1].argmax())
        generated.append(next_id)
        n_decoded += 1
    t_decode = time.time() - t_decode_start

    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\nGenerated ({len(generated)} tokens): {text!r}")
    print(f"\n=== Timing ===")
    print(f"  prompt feed:   {t_prefill*1000:7.0f} ms  ({n_prompt} tok @ {n_prompt/t_prefill:5.2f} tok/s)")
    print(f"  decode:        {t_decode*1000:7.0f} ms  ({n_decoded} tok @ {n_decoded/t_decode:5.2f} tok/s)")
    print(f"  total:         {(t_prefill + t_decode)*1000:7.0f} ms")

    if args.verify:
        compare_len = min(len(generated), len(REFERENCE_IDS))
        match = generated[:compare_len] == REFERENCE_IDS[:compare_len]
        print(f"\n=== Verify ===")
        print(f"  reference text: {REFERENCE_TEXT!r}")
        print(f"  generated text: {text!r}")
        print(f"  reference ids:  {REFERENCE_IDS[:compare_len]}")
        print(f"  generated ids:  {generated[:compare_len]}")
        print(f"  RESULT: {'PASS' if match else 'FAIL'}")
        raise SystemExit(0 if match else 1)


if __name__ == "__main__":
    main()

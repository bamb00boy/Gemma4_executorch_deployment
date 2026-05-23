"""
Phase 3: First attempt at exporting Gemma 4 E2B's text path via torch.export.

Strategy for this iteration:
  - Wrap model.language_model (text-only, vision/audio towers excluded)
  - Use transformers.StaticCache (pre-allocated, export-friendly) instead
    of DynamicCache
  - Plain tensor args: input_ids, attention_mask, position_ids, cache_position
  - Single dynamic dim: sequence length
  - Don't write a custom layer loop yet — let HF's model do its thing and
    catch whatever torch.export complains about. The failures ARE the
    deliverable (see RESULTS.md).

Usage:
    scripts/run.sh scripts/03_export.py 2>&1 | tee results/03_export.log
"""

import _paths  # noqa: F401 — must come before transformers/torch imports

import traceback

import torch
from torch.export import export, Dim
from transformers import AutoModelForImageTextToText

from _wrapper import TextOnlyWrapper, build_static_cache, MAX_SEQ

MODEL_ID = "google/gemma-4-e2b-it"
SEED_SEQ = 16   # example input seq length (any value within [2, MAX_SEQ-1])
DEVICE = "cpu"  # export on CPU so partitioner sees portable ops
DTYPE = torch.float32  # FP32 first; quantize in Phase 4


def main():
    print(f"Loading model on {DEVICE} ({DTYPE})...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=DTYPE
    ).to(DEVICE).eval()

    print(f"Building StaticCache (batch=1, max_seq={MAX_SEQ})...")
    cache = build_static_cache(model, batch=1, max_seq=MAX_SEQ, device=DEVICE, dtype=DTYPE)

    print("Wrapping language model...")
    wrapper = TextOnlyWrapper(model, cache).eval()

    # Example inputs
    input_ids = torch.ones(1, SEED_SEQ, dtype=torch.long)
    attention_mask = torch.ones(1, SEED_SEQ, dtype=torch.long)
    position_ids = torch.arange(SEED_SEQ, dtype=torch.long).unsqueeze(0)
    cache_position = torch.arange(SEED_SEQ, dtype=torch.long)
    example_inputs = (input_ids, attention_mask, position_ids, cache_position)

    print("Eager sanity check (does the wrapper even run?)...")
    with torch.no_grad():
        logits = wrapper(*example_inputs)
    print(f"  eager OK — logits shape: {tuple(logits.shape)}")

    # Reset cache so export sees a clean state
    cache.reset()

    print("\nAttempting torch.export with dynamic seq dim...")
    # min=2 because the model's traced shape guards include `2 <= seq`
    # (a prefill-vs-decode branch). max=MAX_SEQ-1 because sliding_window=512
    # forces the guard `seq < sliding_window`.
    seq = Dim("seq", min=2, max=MAX_SEQ - 1)
    dynamic_shapes = {
        "input_ids": {1: seq},
        "attention_mask": {1: seq},
        "position_ids": {1: seq},
        "cache_position": {0: seq},
    }

    try:
        ep = export(wrapper, example_inputs, dynamic_shapes=dynamic_shapes)
        print("✓ Export SUCCEEDED.")
        print(f"\nGraph signature:\n{ep.graph_signature}")

        # Save the exported program. torch.export.save writes a PT2 zip
        # archive — for >2 GB ExportedPrograms it has been observed to
        # produce a file without a valid central directory if the
        # `f` argument is a Path object. Force str + open-handle path to
        # work around that.
        out_path = str(_paths.MODELS_DIR / "gemma4_e2b_text_fp32.pt2")
        with open(out_path, "wb") as f:
            torch.export.save(ep, f)
        print(f"\nSaved ExportedProgram to: {out_path}")

        # Round-trip verify the save right now — better to catch a
        # corruption here than 30 min later in a downstream script.
        import os
        import zipfile
        assert zipfile.is_zipfile(out_path), \
            "saved .pt2 is not a valid zip archive — central directory missing"
        print(f"  zipfile.is_zipfile: True ({os.path.getsize(out_path) / 1e9:.2f} GB)")
        ep_reloaded = torch.export.load(out_path)
        # Quick sanity: same input/output count
        assert len(ep_reloaded.graph_signature.user_inputs) == \
               len(ep.graph_signature.user_inputs), "input signature mismatch"
        print(f"  round-trip load: OK ({len(ep.graph_signature.user_inputs)} inputs)")
    except Exception as e:
        print(f"\n✗ Export FAILED.")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Message: {e}\n")
        print("=" * 70)
        print("Full traceback (for RESULTS.md):")
        print("=" * 70)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

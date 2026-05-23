"""
Phase 5: Lower the INT4 ExportedPrograms to ExecuTorch via XNNPACK.

Important calibration vs the original README plan: ExecuTorch 1.2.0's
`executorch.backends.arm` is Ethos-U NPU-only (TOSA, Vela, Corstone
targets). The Pi 5 has a Cortex-A76 CPU and no Ethos NPU, so we lower
through XNNPACK instead. XNNPACK is the cross-platform CPU partitioner;
on ARM platforms it dispatches the heavy INT4/INT8 matmul to KleidiAI
kernels automatically. End result is the same KleidiAI fast-path the
README anticipated, just reached through XNNPACK rather than the ARM
backend.

Each .pt2 is lowered separately:
  - models/gemma4_e2b_text_int4_prefill.pt2 → ..._prefill.pte
  - models/gemma4_e2b_text_int4_decode.pt2  → ..._decode.pte

For each, we:
  1. Load the ExportedProgram from disk.
  2. Run `to_edge_transform_and_lower` with XnnpackPartitioner.
  3. Inspect the partitioned graph — count how many ops landed on
     XNNPACK vs portable CPU (the "fallback table").
  4. Serialize to .pte.
  5. Round-trip-load the .pte to confirm.

Usage:
    scripts/run.sh scripts/05_lower.py 2>&1 | tee results/05_lower.log
"""

import _paths  # noqa: F401 — must come before transformers/torch imports

import argparse
import gc
import os
import time
import traceback
from collections import Counter

import torch
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

EXTCACHE_PT2 = str(_paths.MODELS_DIR / "gemma4_e2b_text_int4_extcache.pt2")
EXTCACHE_PTE = str(_paths.MODELS_DIR / "gemma4_e2b_text_int4_extcache.pte")


def summarize_partitions(edge_program_manager) -> dict:
    """Count nodes that ended up in delegated (XNNPACK) subgraphs vs the
    portable graph. The exact attribute path varies across ExecuTorch
    versions, so we try a few."""

    summary = {
        "delegated_nodes": 0,
        "portable_ops": Counter(),
        "total_nodes": 0,
    }
    # ExecuTorch tags delegated subgraphs as nodes with target
    # `executorch.exir.lowered_backend_module.LoweredBackendModule` or
    # similar. Portable nodes show up with their aten op target.
    try:
        # In 1.2.x, edge_program_manager.exported_program() gives back the
        # ExportedProgram with delegate call_function nodes lifted.
        ep = edge_program_manager.exported_program()
        for node in ep.graph.nodes:
            summary["total_nodes"] += 1
            target_str = str(node.target)
            if "lowered_backend_module" in target_str or "LoweredBackend" in target_str or \
               (hasattr(node, "meta") and node.meta.get("delegation_tag")):
                summary["delegated_nodes"] += 1
            elif node.op == "call_function":
                # Use the op name as the bucket key
                name = target_str.replace("aten.", "").split(".default")[0]
                summary["portable_ops"][name] += 1
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
    return summary


def lower_one(name: str, pt2_path: str, pte_path: str) -> bool:
    print(f"\n{'=' * 70}\n  Lowering: {name}\n{'=' * 70}")
    print(f"  in:  {pt2_path}  ({os.path.getsize(pt2_path) / 1e9:.2f} GB)")

    print("  loading ExportedProgram...")
    t0 = time.time()
    ep = torch.export.load(pt2_path)
    print(f"  loaded in {time.time() - t0:.1f}s")

    print("  running to_edge_transform_and_lower (XnnpackPartitioner per_op_mode=True)...")
    # per_op_mode=True: delegate each op to XNNPACK individually instead of
    # fusing into multi-op subgraphs. ARM-side XNNPACK in executorch==1.2.0
    # rejected our fused subgraph (`xnn_status_invalid_parameter` on tensor 6
    # of the first subgraph), while Mac-side XNNPACK accepted the same .pte.
    # Per-op delegation avoids the multi-tensor subgraph entirely.
    t0 = time.time()
    try:
        edge_pm = to_edge_transform_and_lower(
            ep,
            partitioner=[XnnpackPartitioner(per_op_mode=True, verbose=True)],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        print(f"  lowering finished in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"  LOWERING FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    # Partition summary
    summary = summarize_partitions(edge_pm)
    print(f"\n  Partition summary:")
    print(f"    total nodes:        {summary['total_nodes']}")
    print(f"    delegated nodes:    {summary['delegated_nodes']}")
    if summary["portable_ops"]:
        print(f"    portable (un-delegated) ops (top 20 by count):")
        for op, n in summary["portable_ops"].most_common(20):
            print(f"      {n:6d}  {op}")
        n_other = sum(n for _, n in summary["portable_ops"].most_common()[20:])
        if n_other:
            print(f"      {n_other:6d}  <other ops>")
    if "error" in summary:
        print(f"    (partition summary error: {summary['error']})")

    # Defensive: some pass inside to_edge_transform_and_lower flips
    # requires_grad=True on cache buffers (we verified the source .pt2 has
    # them as no-grad). to_executorch() then errors on in-place copy_ to a
    # leaf-with-grad. Clear the flag on every buffer before serializing.
    _n_cleared = 0
    for _ep in (edge_pm.exported_program(),):
        for _, _buf in _ep.named_buffers():
            if _buf.requires_grad:
                _buf.requires_grad_(False)
                _n_cleared += 1
    if _n_cleared:
        print(f"  (cleared requires_grad on {_n_cleared} buffer(s) before serialize)")

    print(f"\n  Serializing to .pte...")
    t0 = time.time()
    try:
        # to_executorch internally evaluates parts of the graph; if any
        # node touches a buffer in-place under enabled autograd, autograd's
        # leaf-with-grad check fires even when the buffers themselves are
        # no-grad. Forcing inference_mode bypasses the autograd
        # bookkeeping entirely.
        with torch.inference_mode():
            prog = edge_pm.to_executorch()
    except Exception as e:
        print(f"  to_executorch() FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    print(f"  to_executorch() finished in {time.time() - t0:.1f}s")

    with open(pte_path, "wb") as f:
        prog.write_to_file(f)
    pte_size_gb = os.path.getsize(pte_path) / 1e9
    print(f"  saved: {pte_path} ({pte_size_gb:.2f} GB)")
    print(f"  size reduction: {os.path.getsize(pt2_path) / os.path.getsize(pte_path):.2f}x vs .pt2")
    return True


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    print("Phase 5: lowering INT4 ExportedProgram to .pte via XNNPACK")
    print("  XnnpackPartitioner -> KleidiAI on ARM, default CPU kernels on x86/Mac")
    print("  Single external-cache .pt2 (handles both prompt feed AND decode)")

    targets = [("extcache", EXTCACHE_PT2, EXTCACHE_PTE)]

    results = {}
    for name, pt2, pte in targets:
        if not os.path.exists(pt2):
            print(f"\nSKIPPING {name} — {pt2} not found. Run scripts/04_quantize.py first.")
            results[name] = None
            continue
        results[name] = lower_one(name, pt2, pte)
        gc.collect()

    print("\n" + "=" * 70)
    print("  PHASE 5 SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        marker = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        print(f"  {name:8s}: {marker}")
    print("=" * 70)

    all_done = all(ok is True for ok in results.values() if ok is not None)
    raise SystemExit(0 if all_done else 1)


if __name__ == "__main__":
    main()

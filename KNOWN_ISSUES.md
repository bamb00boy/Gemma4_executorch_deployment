# Known issues and upstream bug catalog

Each item below documents a reproducible problem encountered during this deployment, the workaround applied, and (where appropriate) the upstream project where a bug report would be valuable. Filing such reports helps improve ExecuTorch and torchao for downstream LLM deployments.

For users encountering one of these issues independently, the workarounds documented here should resolve them. For those reproducing an issue in a minimal test case, filing the corresponding upstream report is encouraged.

---

## 1. ARM XNNPACK rejects fused INT4 subgraphs (`xnn_status_invalid_parameter`) — **highest impact**

**Symptom**

```
[XNNCompiler.cpp:583] Failed to define tensor 6 with code: xnn_status_invalid_parameter
[XNNPACKBackend.cpp:119] XNNCompiler::compileModel failed: 0x1
[method.cpp:110] Init failed for backend XnnpackBackend: 0x1
RuntimeError: Failed to load method forward, error: 0x:1
```

**Where**

ExecuTorch 1.2.0 runtime on `aarch64-linux` (Raspberry Pi 5 / Ubuntu 24.04), loading a `.pte` lowered with default `XnnpackPartitioner()` (multi-op subgraph mode).

Same `.pte`, same `executorch==1.2.0`, **passes** on macOS + Apple Silicon.

**Root cause (suspected)**

The platform-specific XNNPACK binary bundled with the `aarch64-linux` `executorch` pip wheel rejects a tensor parameter in the INT8-dynamic-activation / INT4-weight fused matmul subgraph. The Mac-bundled XNNPACK is more permissive (or differently configured) and accepts it. The exact rejected parameter is not surfaced by ExecuTorch's logging (only "tensor 6" in XNNPACK's internal numbering).

**Workaround applied**

```python
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
edge_pm = to_edge_transform_and_lower(
    ep,
    partitioner=[XnnpackPartitioner(per_op_mode=True)],   # ← was: XnnpackPartitioner()
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
)
```

`per_op_mode=True` makes XNNPACK delegate each op individually instead of fusing into multi-op subgraphs. **This avoids the rejection but loses the kernel-fusion benefits that give KleidiAI's INT4 path its speedup.** Result: Pi 5 decode at **~0.72–0.87 tok/s** vs **6.71 tok/s** for `llama.cpp` + GGUF on the same model on the same hardware ([potato-os/core benchmark](https://github.com/potato-os/core/blob/main/docs/benchmarks/gemma4-pi-benchmark-2026-04-04.md), April 4 2026) — a ~7.7× gap initially attributed entirely to this issue. **Subsequent experiments (below) show the rejection is only part of the story** — even after recovering fused subgraphs, the gap doesn't close.

**Should upstream report?** Yes. ExecuTorch repo. Minimal repro: this entire repository — build the `.pte`, attempt to load on ARM with default partitioner.

**Related but distinct upstream issue:** [ExecuTorch #16896](https://github.com/pytorch/executorch/issues/16896) (Jan 27, 2026, open) reports that XNNPACK *silently skips* integer-tensor ops on certain platforms. Our failure mode is the opposite — XNNPACK *delegates* the op but then *actively rejects* a tensor parameter at compile time. Different failure, same surface area; worth cross-referencing if filing.

### Recovery paths attempted (and what they actually told us)

Three controlled experiments were run to try to close the 7.7× gap. None succeeded — but each ruled out a class of cause and changes our understanding of where the gap actually lives.

**Experiment A — PT2E quantization (KILLED, 2-day spike).**
Hypothesis: torchao's quantization patterns are what ARM XNNPACK rejects; switching to ExecuTorch's native PT2E + `XNNPACKQuantizer` flow should produce patterns it accepts. Result: **structurally incompatible with Gemma 4.** PT2E inserts fake-quant on tensors that are dual-used (consumed by both a Linear AND by `aten.index` for the per-layer embedding lookup). The fake-quant makes them float; the index op requires int. All four PT2E variants attempted (static INT8, dynamic INT8, `set_module_type` restriction, skip-calibration-and-convert-directly) failed at the same line: `aten.index.Tensor(to_4, [unsqueeze_2, unsqueeze_11])`. Not a viable workaround in the current `executorch 1.2.0` / `torchao 0.17.0` / Gemma 4 combination.

**Experiment B — `XnnpackPartitioner(config_precisions=DYNAMIC_QUANT)` (FAILED).**
Hypothesis: restricting XNNPACK to dynamic-quant-only subgraphs may sidestep the fused-INT4 rejection while still allowing some fusion. Result: produced a loadable `.pte` (211 fused subgraphs), bit-exact output on both Mac and Pi. **But ~2× slower on Mac, marginally slower on Pi** (0.70 vs 0.72 tok/s decode). Not a win.

**Experiment C — ExecuTorch nightly 1.4.0.dev (FAILED, but with one important confirmation).**
Hypothesis: six weeks of upstream XNNPACK fixes between stable 1.2.0 (April 1 2026) and nightly 1.4.0.dev20260525 — including one explicitly addressing `getConstantDataPtr` in `XNNCompiler.cpp` (which is exactly where our error fires) — may have fixed the rejection. If so, nightly's default partitioner can emit fused subgraphs that load, and the KleidiAI fast path should light up. Result:
- **The ARM XNNPACK rejection IS fixed in nightly.** The default-partitioner `.pte` (508 fused subgraphs) loads on Pi without `xnn_status_invalid_parameter`. So this issue is resolved upstream and stable 1.3+ should ship the fix.
- **But Pi decode rate does NOT improve.** Three-way bench on the same Pi 5 in one session, bit-exact "Paris" on all: shipped (per_op_mode, 49 unfused) 0.72 tok/s, dynquant (211 fused) 0.70 tok/s, nightly (508 fused) 0.64 tok/s. The 508-vs-49 subgraph count translates to essentially nothing on Pi 5.

**What this changes**

The 7.7× gap is **not partitioner-shaped**. Toggling between unfused per-op delegation and fully fused default partitioning produces decode rates within sampling noise of each other. Three remaining candidates for where the actual gap lives:

1. **Kernel format mismatch.** `llama.cpp` uses GGUF Q4_K_M with hand-tuned ARM NEON / SVE kernels written for that specific block format. The torchao `Int8DynamicActivationIntxWeight` scheme used here is a different quantization layout; even with "4-bit weights" on both sides, the inner matmul loops are different.
2. **KleidiAI may not actually be linked into the XNNPACK build that ships in the executorch pip wheel.** Worth verifying via `nm` / `strings` on `_portable_lib.so`. If absent, no partitioner change can call it.
3. **External KV cache pattern.** Passing K/V cache tensors as graph inputs and outputs forces materialization at the boundary of every attention layer, which may block fusion at a level not exposed to the partitioner config.

**Updated production stance:** shipped (`XnnpackPartitioner(per_op_mode=True)`) stays. It's the fastest of the three measured configurations on Pi 5. Re-test once ExecuTorch 1.3+ stable ships (rejection fix lands without a nightly env); revisit the broader bottleneck only if items 1-3 above warrant investigation.

---

## 2. `transformers.cache_utils.StaticCache` is not an `nn.Module` → `to_edge_transform_and_lower` fails

**Symptom**

```
RuntimeError: Constant language_model.layers.0.self_attn.lifted_tensor_0
is mutated in the forward method. Pls register it as buffer
```

**Where**

ExecuTorch's `to_edge_transform_and_lower` (which calls `run_decompositions`), when the model uses `StaticCache` as its KV cache.

**Root cause**

`StaticCache` and its layer classes (`StaticLayer`, `StaticSlidingWindowLayer`) are plain Python objects. Their K/V tensors are bare attributes. `torch.export` can't tie them to any `nn.Module` buffer registration, so it lifts them as `inputs_to_lifted_tensor_constants`. The wrapper's forward mutates them via `index_copy_`. `run_decompositions` rejects mutated constants.

The `StaticLayer.lazy_initialization` docstring actually warns: *"calling early_initialization on the Cache directly... is required for `torch.export` for example"*. But `early_initialization` alone solves only the lazy-alloc half — it doesn't make the tensors module buffers.

**Workaround applied**

`scripts/_buffer_cache.py` — subclass `StaticLayer`/`StaticSlidingWindowLayer`/`StaticCache` with multiple inheritance to `nn.Module`, register K/V/`cumulative_length` as buffers, put layers in `nn.ModuleList`.

**Should upstream report?** Yes. `transformers` repo. The cleanest fix is for HuggingFace to make `StaticCache` an `nn.Module` natively — would benefit anyone exporting transformer models via `torch.export`.

---

## 3. Two `.pte`s can't share KV-cache state (single-program-state limitation)

**Symptom**

Separate "prefill" and "decode" `.pte` files (each ~13 GB) produce correct output independently, but **cannot be used together** for end-to-end generation. After running prefill, the populated KV cache lives inside prefill's program memory; running decode (a separate program) starts with its own empty cache.

**Where**

ExecuTorch 1.2.0 runtime, Python API. `executorch.runtime.Method.execute` has no way to expose or transfer internal buffer state between programs.

**Root cause**

Each `.pte` is a self-contained ExecuTorch program with its own internal buffer state. The Python runtime doesn't expose a buffer-set API for resetting or copying buffers across programs at runtime.

**Workaround applied**

Externalize the cache: cache tensors become forward inputs and updated cache is returned as forward outputs. One stateless `.pte` handles both prompt feed (called token-by-token) and decode. See `scripts/_external_cache.py` and `TextWrapperExternal` in `scripts/_wrapper.py`.

Cost: token-by-token prompt feed (no batched prefill), so TTFT scales linearly with prompt length.

**Should upstream report?** Probably. ExecuTorch repo. Either:
- Add a Python API for cross-method state transfer, OR
- Add a multi-method `.pte` workflow where methods share program state.

---

## 4. `torch.export.save(ep, pathlib.Path)` truncates the `.pt2` for >2 GB exports

**Symptom**

```
RuntimeError: PytorchStreamReader failed reading zip archive:
failed finding central directory.
```

The `.pt2` file exists, has reasonable size (~8 GB), but is missing its zip central directory. `zipfile.is_zipfile()` returns False. The save reports success silently.

**Where**

`torch==2.11.0`, `torch.export.save(ep, path)` where `path` is a `pathlib.Path` and `ep` serializes to >2 GB.

**Root cause (unknown)**

Calling with a string path or an open file handle produces a valid archive. Likely a path-handling bug in `torch.export.save`'s zip writer.

**Workaround applied**

```python
# Bad:
torch.export.save(ep, _paths.MODELS_DIR / "model.pt2")

# Good:
with open(str(path), "wb") as f:
    torch.export.save(ep, f)
```

Plus immediate round-trip verify (`zipfile.is_zipfile` + `torch.export.load`) at save time so corruption surfaces at write, not in a downstream script.

**Should upstream report?** Yes. PyTorch repo. Minimal repro: any ExportedProgram > 2 GB saved with `pathlib.Path`.

---

## 5. `torchao.int4_weight_only` removed; new configs are GPU-only on CPU

**Symptom**

```
ImportError: cannot import name 'int4_weight_only' from 'torchao.quantization'
ImportError: Requires mslk >= 1.0.0
```

`mslk` is Meta's TINYGEMM CUDA kernel library — only a placeholder `0.0.0` on PyPI. Every other `Int4*Tensor` variant in `torchao 0.17.0` requires CUDA / xpu / npu.

**Where**

`torchao==0.17.0` on CPU, attempting INT4 weight-only quantization.

**Workaround applied**

```python
from torchao.quantization import quantize_, Int8DynamicActivationIntxWeightConfig
from torchao.quantization.granularity import PerGroup

quantize_(
    model,
    Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int4,
        weight_granularity=PerGroup(group_size=128),
    ),
)
```

INT8 dynamic activation + INT4 weight is the standard CPU LLM scheme anyway.

**Should upstream report?** Already known to torchao maintainers. The `mslk` dep is a tracked issue. Skip filing.

---

## 6. `torchao`'s quantized embedding lowering has no ExecuTorch out-variants

**Symptom**

```
RuntimeError: Missing out variants: {
    'torchao::quantize_affine',
    'torchao::choose_qparams_affine',
    'torchao::dequantize_affine',
}
```

at `to_executorch()` when `Int8WeightOnlyConfig` (or `IntxUnpackedToInt8Tensor`) is applied to an `nn.Embedding`.

**Where**

`torchao==0.17.0` + `executorch==1.2.0`.

**Root cause**

The `torchao::*affine` ops are emitted by `AffineQuantizedTensor` / `IntxUnpackedToInt8Tensor` for embedding lookups. They don't have ExecuTorch out-variant implementations registered. Linear layers using the same tensor types succeed because the XNNPACK delegate fuses the matmul + dequant into KleidiAI kernels, hiding these ops from the portable graph; embedding lookups don't pattern-match the delegate.

**Workaround applied**

Hand-rolled `Int8Embedding` (`scripts/_int8_embedding.py`) using only standard aten ops (`index_select` + `to(dtype)` + `mul`). All have out-variants, all lower cleanly through XNNPACK. Big size win: `.pte` dropped from 12.18 GB → 5.14 GB.

**Should upstream report?** Yes — confirmed **not** previously documented. ExecuTorch issue [#1263](https://github.com/pytorch/executorch/issues/1263) (Nov 2023, marked Done) covers the related `quantized_decomposed::quantize_per_tensor` / `dequantize_per_tensor` op family but does **not** cover `torchao::quantize_affine` / `choose_qparams_affine` / `dequantize_affine`. The torchao-namespaced ops are a separate, currently-undocumented gap. File against either ExecuTorch (add out-variants) or torchao (use ExecuTorch-friendly ops for embedding lookup) — the ExecuTorch side is probably the right fix.

---

## 7. BF16 baseline + INT4 dynamic-activation Linears → same out-variant error

**Symptom**

Same `Missing out variants: torchao::quantize_affine` error at `to_executorch()`, but only when `dtype=torch.bfloat16` is used with `Int8DynamicActivationIntxWeightConfig`. FP32 baseline works.

**Where**

`torchao==0.17.0` + `executorch==1.2.0`, BF16 + INT4 dynamic act path.

**Workaround applied**

Use `dtype=torch.float32` baseline. ~250 MB size regression vs BF16, but lowering works.

**Should upstream report?** Yes. torchao or ExecuTorch. The BF16 path emitting different ops than FP32 (and those ops missing out-variants) is the bug.

---

## 8. ExecuTorch `.pte` shape-specializes to upper bound of dynamic `Dim` (no runtime dynamism)

**Symptom**

Exported with `Dim("seq", min=2, max=511)`, expected runtime to accept any seq in that range. Actual: `.pte`'s method metadata reports input shapes as `[1, 511]` — fully specialized to the upper bound.

**Where**

ExecuTorch 1.2.0. The exported `.pt2` has the symbolic dim; the lowered `.pte` collapses to the worst case.

**Root cause**

ExecuTorch's memory planner pre-allocates buffers for the worst-case shape and doesn't support runtime resizing for delegated programs.

**Workaround applied**

Runner pads inputs to `seq=511` and masks padding via `attention_mask=0`. Wastes some compute (the 511 - real_prompt_len padded positions) but functionally correct.

**Should upstream report?** Already a known ExecuTorch limitation; runtime-dynamic shapes are tracked work. Skip filing.

---

## 9. torch version mismatch between `.pte`-build host and runtime host silently breaks XNNPACK

**Symptom**

`.pte` built against `torch==2.11.0`'s XNNPACK; loaded on a host with `torch==2.12.0`'s XNNPACK → `xnn_status_invalid_parameter` (same error class as issue #1, different root cause).

**Where**

ExecuTorch 1.2.0 ships platform-specific XNNPACK binaries tied to the torch version it was built against. Pip can resolve different torch versions on Mac (lowering host) and Pi (runtime host) by default.

**Workaround applied**

Pin `torch==2.11.0` explicitly in both Mac-side `requirements.txt` and Pi-side `requirements_pi.txt` (also handled by `scripts/deploy_pi.sh`).

**Pin is actively load-bearing as of May 2026.** PyTorch 2.12.0 was released on May 13, 2026 and is now the default install target for `pip install torch` on supported platforms. Without the explicit `torch==2.11.0` pin, a fresh Pi install will silently land on 2.12.0 and trigger this failure on `.pte` load.

**Should upstream report?** A documentation note in ExecuTorch stating that the `.pte` build-time and runtime torch versions must match would be appropriate. This is not a fundamental bug, but a footgun that warrants explicit documentation.

---

## 10. ExecuTorch 1.2.0's logging doesn't surface which tensor / op fails

**Symptom**

Errors like `Failed to define tensor 6` (XNNPACK) or `Missing out variants: {ops...}` (to_executorch) don't tell you which model op or which tensor in YOUR graph corresponds. Debugging requires guessing.

**Workaround applied**

None. Use the partition summary output from `scripts/05_lower.py` to narrow down which op types are likely involved. Iterate by isolation (BF16 vs FP32, with/without embed quant, etc.).

**Should upstream report?** Yes. ExecuTorch repo. Even mapping XNNPACK's internal tensor ID back to the source aten op name would dramatically reduce debugging time.

---

## What's NOT a known issue (clarifications)

- **5.14 GB `.pte` is too big for Pi.** It isn't. Pi 5 has 8 GB RAM; the `.pte` is mmapped and only the active working set is paged in (~2 GB during inference). 5.14 GB is the on-disk size.
- **Quality is degraded by quantization.** It isn't. INT4 weights + INT8 embeddings produce bit-exact token sequences vs FP32 reference on the canonical prompt across 9 tokens. Larger-scale evaluation hasn't been done, but the toy test passes cleanly.
- **Pi 5 lacks KleidiAI.** It does not — KleidiAI is a userspace library included in XNNPACK on aarch64. The reason it does not contribute to performance in this repo is the XNNPACK `per_op_mode` workaround (issue #1), not the absence of KleidiAI.

---

## How to use this catalog if you're filing a bug

1. Pick the issue.
2. Reduce to a minimal `.pte` (or `.pt2` for export issues) that reproduces. Often a single-layer or 2-token version suffices.
3. Include: ExecuTorch version, torchao version, torch version, host arch, exact error message + stack trace.
4. Link back to this catalog so the maintainer sees the chain of related findings.

The issues marked "Should upstream report? Yes" are the ones where filing has the highest expected value.

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

`per_op_mode=True` makes XNNPACK delegate each op individually instead of fusing into multi-op subgraphs. **This avoids the rejection but loses the kernel-fusion benefits that give KleidiAI's INT4 path its speedup.** Result: Pi 5 decode at **~0.87 tok/s** vs **6.71 tok/s** for `llama.cpp` + GGUF on the same model on the same hardware ([potato-os/core benchmark](https://github.com/potato-os/core/blob/main/docs/benchmarks/gemma4-pi-benchmark-2026-04-04.md), April 4 2026) — a ~7.7× gap attributable entirely to this issue.

**Should upstream report?** Yes. ExecuTorch repo. Minimal repro: this entire repository — build the `.pte`, attempt to load on ARM with default partitioner.

**Related but distinct upstream issue:** [ExecuTorch #16896](https://github.com/pytorch/executorch/issues/16896) (Jan 27, 2026, open) reports that XNNPACK *silently skips* integer-tensor ops on certain platforms. Our failure mode is the opposite — XNNPACK *delegates* the op but then *actively rejects* a tensor parameter at compile time. Different failure, same surface area; worth cross-referencing if filing.

**Possible recovery path (not attempted in this repo)**

Switch from `torchao.quantize_` to ExecuTorch's PT2E + `XNNPACKQuantizer` flow. PT2E produces XNNPACK-native quantization patterns that the ARM build is more likely to accept. Substantial rework; not validated here.

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

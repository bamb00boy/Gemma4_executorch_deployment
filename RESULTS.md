# Results & Gotchas Log

Running log of findings during the Gemma 4 E2B → ExecuTorch → Pi 5 deployment. This is the raw record — cleaned up versions go in the final writeup.

## Environment

| Package | Version |
|---|---|
| Python | 3.12 |
| executorch | 1.2.0 |
| torch | 2.11.0 |
| torchao | 0.17.0 |
| transformers | 5.5.3 |
| OS (dev) | macOS (Apple Silicon) |
| OS (target) | Pi OS 64-bit Bookworm |
| Hardware (target) | Raspberry Pi 5, 8GB RAM |

## Phase 1 — Environment setup

### Gotcha: Poetry vs conda
Poetry's `poetry env use` command struggles with Python paths containing `@` on zsh (e.g. `python@3.11`). Zsh treats `@` as a glob character. Quoting the path works (`"$(brew --prefix)/opt/python@3.11/bin/python3.11"`) but conda is a cleaner fit for ML projects — it manages the Python interpreter itself, no Homebrew dance needed.

**Resolution:** Switched to Miniforge/conda. Simpler, and it's what ExecuTorch's own docs assume.

### Gotcha: Version pinning
ExecuTorch is tightly coupled to specific torch/torchao versions. Don't pin manually — install `executorch` first via pip and let it pull matching dependencies. Then `pip freeze > requirements.txt` to capture the real working set.

## Phase 2 — Smoke test

### Gotcha: `torchvision` is a hidden dependency
Gemma 4's processor instantiates a `Gemma4VideoProcessor` even for image-only or text-only inputs. This requires `torchvision` to be installed. Not listed in any Gemma 4 quickstart guide.

```
ImportError: Gemma4VideoProcessor requires the Torchvision library
```

**Resolution:** `pip install torchvision`

### Gotcha: `torch_dtype` deprecated
Transformers 5.x renamed `torch_dtype` to `dtype` in `from_pretrained()`. Old tutorials all use `torch_dtype`. Functional but emits a deprecation warning.

```
`torch_dtype` is deprecated! Use `dtype` instead!
```

**Resolution:** Use `dtype=torch.float16` going forward.

### Gotcha: `top_p`/`top_k` generation warning
When using `do_sample=False` (greedy decoding), transformers warns about invalid generation flags. Harmless — the model's `generation_config.json` sets these defaults, but they're ignored during greedy decode.

### Result: Smoke test passing
- Text generation: correct ("The capital of France is **Paris**.")
- Vision + text: correct (described a flower image accurately)
- Model size: single 10.2GB safetensors download
- Device: MPS (Apple Silicon) works for both text and vision paths

### Module tree observations
- `embed_tokens_per_layer`, `per_layer_model_projection`, `per_layer_projection_norm` — new in Gemma 4, not present in Gemma 3. Likely related to the "effective 2B" parameter architecture.
- `rotary_emb` is a sibling of `layers`, not inside each attention block — shared RoPE state.
- `audio_tower` present with subsample conv, relative positional encoding, and output projection.

## Phase 3 — Architecture inspection & export

### Inspection (02_inspect.py)

Full output: [`results/02_inspect.txt`](results/02_inspect.txt). Populated config + parameter tables in [`docs/architecture.md`](docs/architecture.md). Key numbers:

- 35 decoder layers, hidden_size 1536, head_dim 256, 8 attention heads, **1 KV head** (GQA 8:1)
- vocab 262,144 — large; lm_head alone is 403 M params
- sliding_window 512 — confirms per-layer alternation between `sliding_attention` and `full_attention`
- final_logit_softcapping 30.0
- **Total params: 5.51 B**; `embed_tokens_per_layer` alone is **2.35 B** (the "E2B" trick is sparse activation of a large per-layer table)

### Export hazards (from source inspection)

These will need workarounds when exporting:

| # | Hazard | Where | Plan |
|---|---|---|---|
| 1 | `shared_kv_states: dict[int, tuple[Tensor, Tensor]]` arg | `Gemma4TextDecoderLayer.forward` | Flatten into positional tensor args, or skip shared-KV optimization for export |
| 2 | `past_key_values: Cache` | both forwards | Use `StaticCache` (pre-allocated) instead of `DynamicCache` |
| 3 | `getattr(self, f"{layer_type}_inv_freq")` | `Gemma4TextRotaryEmbedding.forward` | Pre-compute both (sliding, full) cos/sin tensors in wrapper, select statically per layer |
| 4 | `@dynamic_rope_update` decorator | rotary forward | Bypass by computing cos/sin outside the graph |
| 5 | `maybe_autocast(device_type=x.device.type, ...)` | rotary forward | Acceptable — device specializes at export time |
| 6 | `gelu_pytorch_tanh` activation | decoder layer MLP | Standard op, should lower fine to ARM/XNNPACK |
| 7 | `final_logit_softcapping = 30.0` | model.forward | Trivial — single `tanh(x/cap)*cap` |

### Export attempt 1 — `ConstraintViolationError` on seq dim

**Setup:** `TextOnlyWrapper` around `model.language_model` + `lm_head`, using `transformers.cache_utils.StaticCache` (max_cache_len=2048). Dynamic dim: `seq = Dim("seq", min=1, max=2048)`.

**Eager:** passed (`logits shape: (1, 16, 262144)`).

**Export:**
```
torch._dynamo.exc.UserError: Constraints violated (seq)!
- Not all values of seq in the specified range seq <= 2048 satisfy
  the generated guard 2 <= L['input_ids'].size()[1]
                  and L['input_ids'].size()[1] <= 511
Suggested fixes:
  seq = Dim('seq', max=511)
```

**Root cause:** Gemma 4's attention generates guards tying `seq` to `sliding_window=512`, and a separate `seq >= 2` guard (prefill-vs-decode branch). The model's internal shape contract is `2 <= seq < sliding_window`.

**Resolution:** Tightened to `Dim("seq", min=2, max=511)` and lowered `MAX_SEQ` to 512 (StaticCache length). Longer contexts will require either disabling sliding window or a paged-cache strategy — out of scope for this first export.

### Export attempt 2 — SUCCESS ✓

Same wrapper, `seq=Dim("seq", min=2, max=511)`, `MAX_SEQ=512`, FP32 on CPU.

```
Eager: logits shape (1, 16, 262144)
Export: SUCCEEDED
Saved: models/gemma4_e2b_text_fp32.pt2  (9.06 GB)
```

Graph signature shows:
- 4 USER_INPUT: `input_ids`, `attention_mask`, `position_ids`, `cache_position`
- 1 USER_OUTPUT: the post-softcap logits
- All 35 layers' parameters lifted (q/k/v/o proj + q_norm/k_norm + gate/up/down + 4 layernorms + 3 per-layer-input modules per layer)
- Constant tensors (`lifted_tensor_*`) carried per layer — likely the per-layer-type attention mask constants

**No warnings, no rewrites, no monkeypatches needed.** The HF Gemma 4 implementation in transformers 5.5.3 + StaticCache exports cleanly out of the box (within the sliding-window seq constraint).

This is a much better Phase 3 outcome than the README/architecture doc anticipated. The export hazards listed from source inspection (dict-typed `shared_kv_states`, dynamic `getattr` in rotary, `@dynamic_rope_update`) all traced through fine — the constants got baked in or the decorator no-ops at export time.

**Caveats / known unknowns:**
- `seq` is capped at 511, so a single forward can't process more than 510 prefill tokens at once. Decode (seq=1) is blocked by the `seq >= 2` guard — needs separate export or a different cache-position scheme.
- FP32 artifact is 9.06 GB — INT4 in Phase 4 should bring this under 2 GB.
- Haven't verified numerical equivalence between the eager and exported graph yet.

### Cross-checks before Phase 4

After the bare export, three things needed verifying: numerical equivalence (does the exported graph produce the same logits as eager?), decode-shape export (can we trace `seq=1`?), and end-to-end greedy generation (does the wrapper drive prefill+decode correctly?). Two bugs surfaced.

#### Gotcha: `.pt2` truncated when saved via `pathlib.Path`

Running `torch.export.save(ep, path)` where `path` is a `pathlib.Path` produced a partial archive: the central directory was missing (`zipfile.is_zipfile` → False) and `torch.export.load` failed with `RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory`. The save reported success silently.

```python
# BAD (8.4 GB, truncated, central directory missing):
torch.export.save(ep, _paths.MODELS_DIR / "...pt2")

# OK (18.5 GB, valid zip, round-trip load passes):
with open(str(path), "wb") as f:
    torch.export.save(ep, f)
```

This is on `torch==2.11.0` and only manifests for >2 GB exports — small ExportedPrograms saved with Path arguments are fine. Workaround is to always pass an open file handle. Verified the new save by immediately reloading inside `03_export.py` so we catch this at write time, not in a downstream script.

#### Cross-check 1: prefill numerical equivalence — **PASS (bit-exact)**

[`scripts/06_verify.py`](scripts/06_verify.py) runs the wrapper eagerly on the chat-templated prompt `"The capital of France is"` (seq=14), then loads the `.pt2` and runs the same inputs through the exported `module()`. Compares last-position logits.

```
next-token match:    True   (eager=818, exported=818)  → "The"
top-5 ids identical: True   [818, 100, 1018, 669, 18574]
max |diff|:          0.0e+00
mean |diff|:         0.0e+00
max relative diff:   0.0e+00
```

Zero numerical drift — the export captured the graph exactly. Eager and exported share parameters byte-for-byte.

#### Cross-check 2: decode (seq=1) export — **PASS**

[`scripts/07_decode_check.py`](scripts/07_decode_check.py) attempts `torch.export(wrapper, example_with_seq=1)` with no `dynamic_shapes`. Hypothesis: the `2 <= seq` guard from the prefill export was generated only because we declared `seq` as a `Dim`. With a fixed `seq=1` example and no dynamic shape, the guard specializes away.

Confirmed: `Export SUCCEEDED, inputs: 4, outputs: 1`. Both prefill (dynamic seq ∈ [2, 511]) and decode (static seq=1) are exportable.

We did **not** save the decode `.pt2` to disk this phase — the prefill `.pt2` is 18.5 GB FP32, and the dev machine has ~26 GB free. Two FP32 `.pt2`s don't fit. Both prefill + decode will be saved after INT4 quantization in Phase 4, which should bring each artifact under 5 GB.

#### Cross-check 3: end-to-end greedy generation — **PASS**

The wrapper's prefill+decode loop must produce the same output as `model.generate(do_sample=False)`. Same prompt, 10 max new tokens.

**Reference** (`model.generate`):
```
ids:  [818, 5279, 529, 7001, 563, 5213, 50429, 84750, 106]
text: "The capital of France is **Paris**."
```

**Wrapper** (prefill seq=14 + decode loop seq=1 × 9):
```
ids:  [818, 5279, 529, 7001, 563, 5213, 50429, 84750, 106, 106]
text: "The capital of France is **Paris**."
```

9/9 token-id match (10th wrapper token is a post-EOS continuation; reference stops early at the `<turn|>` EOS token, wrapper has no early-stop yet).

#### Gotcha: `attention_mask` shape during decode

First decode-loop attempt diverged after token 0 — wrapper produced `"Theodore 2000000"` instead of `" capital of France is **Paris**."`. Root cause: passed `attention_mask = torch.ones((1, 1))` for the seq=1 decode call, meaning *"the new query attends only to itself"*. The new query never saw the prefilled cache.

Fix: pass `attention_mask` of shape `[1, cache_position + 1]` so the model knows positions `[0, cache_position]` in the cache are valid. After the fix, exact 9/9 match.

This will matter for the C++ runner in Phase 6 — it must build a growing attention_mask each decode step, not pass a 1-element mask.

### Phase 3 deliverables

- [`docs/architecture.md`](docs/architecture.md) — populated config + parameter tables + export-hazard analysis
- [`results/02_inspect.txt`](results/02_inspect.txt) — full architecture inspection output
- [`results/03_export.log`](results/03_export.log) — full export run log
- [`results/06_verify_eager.log`](results/06_verify_eager.log) + [`results/06_verify_exported.log`](results/06_verify_exported.log) — bit-exact prefill cross-check
- [`results/07_decode_check.log`](results/07_decode_check.log) — decode export + greedy generation cross-check
- [`scripts/_wrapper.py`](scripts/_wrapper.py) — shared `TextOnlyWrapper` + `build_static_cache` (used by 03 + 07)
- `models/gemma4_e2b_text_fp32.pt2` — exported FP32 prefill program (gitignored, 18.54 GB). **Deleted in Phase 4 prep** to make disk room for the INT4 prefill+decode artifacts; recreate via `scripts/run.sh scripts/03_export.py` if you need to re-run `06_verify.py`.

### Phase 3 closing notes

- The export hazards we anticipated from source inspection (dict-typed `shared_kv_states`, dynamic `getattr` in rotary, `@dynamic_rope_update`) did not actually manifest. transformers 5.5.3's Gemma 4 export-traces cleanly with `StaticCache`.
- The two real Phase 3 gotchas were both downstream of `torch.export`, not inside the model: the `pathlib.Path` save bug, and the decode-loop `attention_mask` shape requirement.
- **Known constraint:** `seq ∈ [2, 511]` for the dynamic prefill graph (sliding_window=512). Longer contexts need a separate paged-cache export strategy — out of scope until after we have working benchmarks.
- **Carried forward to Phase 4:** quantize first, then re-export both prefill (dynamic seq) and decode (static seq=1) graphs to disk.

## Phase 4 — Quantization

### Gotcha: torchao 0.17's `Int4WeightOnlyConfig` is GPU-only

First attempt used `from torchao.quantization import int4_weight_only` (the legacy convenience factory from the README). Doesn't exist in torchao 0.17.0:

```
ImportError: cannot import name 'int4_weight_only' from 'torchao.quantization'
```

The new API is `Int4WeightOnlyConfig`. Tried that:

```
ImportError: Requires mslk >= 1.0.0
```

`mslk` is Meta's CUDA TINYGEMM kernel library — only PyPI version is `0.0.0` (placeholder). Every other `Int4*Tensor` variant in torchao 0.17 also requires CUDA / xpu / npu:

| Format | Requires |
|---|---|
| `PLAIN` (default) | `mslk` (CUDA TINYGEMM) |
| `PRESHUFFLED` | CUDA |
| `PLAIN_INT32` | xpu (Intel GPU) or npu only |
| `TILE_PACKED_TO_4D` | `torch.cuda.get_device_capability() >= 8` |

This is a real torchao limitation, not a config error. We're not running on CUDA.

**Could MLX substitute?** No. MLX is a separate framework with its own tensor type — quantizing with MLX gives MLX-format weights that `transformers` can't ingest, `torch.export` can't trace, and ExecuTorch can't lower. We'd lose the entire pipeline.

### Resolution: `Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4)`

CPU-compatible scheme: INT8 dynamic activations + INT4 weights. This is the standard scheme for CPU LLM inference (xnnpack/KleidiAI both target it). Storage on disk is unpacked INT8 (1 byte per INT4 weight) — no compression vs INT8 yet. True INT4 packing happens at ExecuTorch lowering time (Phase 5) when the ARM backend rewrites for KleidiAI kernels.

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

### Gotcha: peak RAM during FP32 baseline + INT4 generation

Running both `model.generate` calls (FP32 baseline, then INT4) on a 5.5 B model on CPU exhausts a 25 GB Mac. The first attempted re-run hung for 40+ minutes with RAM down to 72 MB free and the swap file eating 23 GB of disk. Killed the process.

**Resolution:** skip the FP32 generation by default — we've already captured the FP32 token sequence three times in prior runs (Phase 2 smoke test on MPS, Phase 3 `06_verify.py` eager phase, Phase 4 v3 before it crashed on decode). All three produced the identical sequence:

```
ids:  [818, 5279, 529, 7001, 563, 5213, 50429, 84750, 106]
text: "The capital of France is **Paris**."
```

`04_quantize.py` now hardcodes this as `FP32_REFERENCE_IDS` and compares INT4 against it. `--with-fp32-baseline` re-enables in-process FP32 gen for verification. Adds `gc.collect()` between heavy steps to free intermediate state.

### Gotcha: decode `mask_len` hit the same sliding-window guard

`Dim("mask_len", min=2, max=MAX_SEQ)` with `MAX_SEQ=512` violated the model's `mask_len < 512` guard (same root cause as Phase 3's `seq < 512`). Fixed: `max=MAX_SEQ - 1 = 511`.

### Results

| Check | Result |
|---|---|
| INT4 vs FP32 token-id sequence | **PASS** — 9/9 identical |
| INT4 vs FP32 generated text | **PASS** — `"The capital of France is **Paris**."` byte-for-byte |
| INT4 prefill export (seq ∈ [2, 511]) | **PASS** — `models/gemma4_e2b_text_int4_prefill.pt2` (13.41 GB) |
| INT4 decode export (seq=1, mask_len ∈ [2, 511]) | **PASS** — `models/gemma4_e2b_text_int4_decode.pt2` (13.41 GB) |
| INT4 exported vs INT4 eager numerical | **PASS** — `max diff = 0.0e+00`, top-5 logits identical to 6 decimals |

### Why .pt2 is 13 GB, not the expected 3 GB

INT4 weights are stored unpacked (1 byte per weight, just with 4 high bits set to 0) inside the `IntxUnpackedToInt8Tensor` class. So the on-disk size is dominated by the same N bytes as INT8 storage, not the true 4-bit-packed N/2.

The actual 8× compression happens at lowering time when ExecuTorch's ARM backend repacks weights into KleidiAI's expected INT4 tile layout. Expected `.pte` size after Phase 5: ~2-3 GB.

In-memory footprint reported by `model_size_bytes`: 22.03 GB (vs 20.42 GB FP32). Slightly bigger because torchao adds per-group scale tensors. The footprint difference is misleading — the SAVING comes from cheaper matmul kernels (8-bit / 4-bit integer fast-paths), not from holding less data in RAM.

### Phase 4 deliverables

- [`scripts/04_quantize.py`](scripts/04_quantize.py) — quantize + INT4 sanity gen + re-export prefill & decode
- [`scripts/08_verify_int4.py`](scripts/08_verify_int4.py) — INT4 eager-vs-exported numerical check (two-phase)
- [`results/04_quantize.log`](results/04_quantize.log) — Phase 4 v5 (passing) full log
- [`results/08_verify_int4_eager.log`](results/08_verify_int4_eager.log) + [`results/08_verify_int4_exported.log`](results/08_verify_int4_exported.log) — bit-exact INT4 cross-check
- `models/gemma4_e2b_text_int4_prefill.pt2` (13.41 GB, gitignored)
- `models/gemma4_e2b_text_int4_decode.pt2` (13.41 GB, gitignored)

### Carry-forward to Phase 5

- Both `.pt2` files exist and round-trip-load cleanly. Phase 5 will load each, run ExecuTorch's `to_edge_transform_and_lower` with the ARM (or XNNPACK) partitioner, capture the fallback table, and serialize to `.pte`.
- The 13 GB `.pt2` size will collapse to ~2-3 GB once KleidiAI repacks the INT4 weights — expect a big disk-size delta in Phase 5, and that's the win we're after.
- Same prompt + reference tokens (`"The capital of France is **Paris**."`) will be the regression check end-to-end.

## Phase 5 — ExecuTorch backend lowering

### Calibration: ARM backend vs XNNPACK

The README originally planned to lower through `executorch.backends.arm.arm_partitioner.ArmPartitioner`. In ExecuTorch 1.2.0 that backend exists, but inspecting its submodules (`ethosu`, `tosa`, `vgf`, `arm_vela`) makes its scope clear: it's an **NPU-only path** — TOSA / Ethos-U / Vela / Corstone targets. The Pi 5 has a Cortex-A76 CPU and no Ethos-U NPU, so this backend doesn't apply.

The right backend for a Cortex-A *CPU* deployment is **XNNPACK**. On ARM platforms XNNPACK dispatches the heavy INT4/INT8 matmul to **KleidiAI kernels automatically** — so we get the same KleidiAI fast-path the README anticipated, just reached through the XNNPACK partitioner rather than the ARM backend.

`from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner` is the partitioner class. `to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()], compile_config=EdgeCompileConfig(_check_ir_validity=False))` is the call shape.

### Blocker: mutable constants from `StaticCache`

First attempt at lowering the INT4 prefill .pt2 failed:

```
RuntimeError: Constant language_model.layers.0.self_attn.lifted_tensor_0
is mutated in the forward method. Pls register it as buffer
```

These `lifted_tensor_*` warnings actually surfaced way back in Phase 3 (`scripts/03_export.py` output) but didn't break anything until now. Calling them "benign" was wrong in hindsight.

#### Root cause

`torch.export` walks the wrapper's forward, encounters tensors that aren't registered as parameters or buffers on any `nn.Module`, and *lifts* them into the ExportedProgram as **constants** (`inputs_to_lifted_tensor_constants`). 45 such constants ended up in our prefill `.pt2`:

| Lifted tensor | Shape | Dtype | What it actually is |
|---|---|---|---|
| `..._lifted_tensor_0`, `_3`, `_6`, … | `(1,)` | int64 | `StaticLayer.cumulative_length` — the per-layer cache write counter |
| `..._lifted_tensor_1`, `_2`, `_4`, `_5`, … | `(1, 1, 512, 256)` | float32 | `StaticLayer.keys` and `StaticLayer.values` — the K/V cache slots |

i.e. **the K/V cache itself**, plus the bookkeeping counter. The shape `(1, 1, 512, 256)` is exactly `(max_batch, num_kv_heads, MAX_SEQ, head_dim)`.

Why are they lifted as constants? Because `transformers.cache_utils.StaticCache` (and its layer types `StaticLayer`, `StaticSlidingWindowLayer`) are **not `nn.Module`s** — they're plain Python objects that hold tensors as bare attributes. When the wrapper does `self.cache = StaticCache(...)`, the cache isn't a child module of the wrapper, so its tensors aren't part of the wrapper's `nn.Module.buffers()`. Export sees raw tensors used in forward, can't tie them to a parameter/buffer registration, and lifts them as constants.

In Phase 3 and Phase 4 this was harmless because we never asked `torch.export` to do any decomposition pass that enforces the "constants must not be mutated" invariant. Both `06_verify.py` and `08_verify_int4.py` simply load and run the `.pt2` directly — no decomposition needed.

**`to_edge_transform_and_lower` is different.** Internally it calls `program.run_decompositions({})`, which calls `_override_graph_signature_for_temp_registered_constants` (`torch/_export/utils.py:137`). That function detects: "this thing is registered as a constant in the signature, but the graph mutates it via `index_copy_` / `add_` etc." — and raises the error.

The StaticLayer source actually documents this:

> `lazy_initialization(...)` — If this is unwanted, one can call `early_initialization(...)` on the Cache directly, which will call this function ahead-of-time (**this is required for `torch.export` for example**).

So HF knows about the export issue. Calling `early_initialization` solves the lazy-allocation half (so the tensors exist at export time, not lazily on first call), but it does **not** make them `nn.Module` buffers. The lifting still happens.

#### Alternatives considered

| Option | What it does | Pros | Cons |
|---|---|---|---|
| **A. nn.Module shim around StaticCache** *(chosen)* | Subclass `StaticLayer` and `StaticCache` to also inherit from `nn.Module`, register K/V/`cumulative_length` as buffers, put the layers in an `nn.ModuleList`. Substitute in `_wrapper.py`. | Single-file fix. Clean: tensors are real buffers, signature shows them in `buffers` + `buffers_to_mutate`, lowering passes the check. Carries no torch-internal-API dependency. | Requires re-exporting (re-run Phase 4, ~25 min). The shim has to mirror enough of `StaticLayer` for the LM's `.update()` to still work. |
| **B. Post-load ExportedProgram surgery** | After `torch.export.load`, walk `ep.graph_signature`, move entries from `inputs_to_lifted_tensor_constants` → `inputs_to_buffers`, move tensors from `ep.constants` → `ep.state_dict`, add to `buffers_to_mutate`, rebuild `ExportGraphSignature` (NamedTuple, so requires full reconstruction), reconstruct the `ExportedProgram`. | No re-export. Saves ~25 min. | Touches torch internals deeply. Multiple immutable NamedTuples to rebuild correctly. `ExportedProgram.__init__` signature varies across versions. High chance of subtle breakage that surfaces only at the next ExecuTorch pass. Brittle for the project's stated goal of being a publishable reproducible recipe. |
| **C. Use older `to_edge` + manual `to_backend`** | Skip `to_edge_transform_and_lower`. Call `to_edge(ep, ...)` (which historically didn't run decompositions in the same way) then call partition + lower in separate steps. | Quick to try. | The 1.2.0 `to_edge` may chain to the same decomposition internally, and even if it doesn't, the next pass (`to_executorch()`) likely has the same constraint. Doesn't fix the root cause — just delays the error. |
| **D. Defer Phase 5, do Phase 6 setup** | Park the lowering, move to Pi-side ExecuTorch runtime build (cross-compile, KleidiAI flags, etc.). | Parallel progress. | We won't know whether the `.pte` lowers cleanly until we come back. If it doesn't, we wasted Pi-side effort. |

**Choice: A.** Re-export cost (~25 min wall time) is small compared to (B)'s engineering risk, and the resulting wrapper is a normal `nn.Module` tree that any downstream ExecuTorch pass can consume.

### Phase 5 — design of the nn.Module cache shim

To make the cache export-friendly:

1. **`BufferStaticLayer(StaticLayer, nn.Module)`** — multiple-inheritance subclass. `nn.Module.__init__` runs first to set up the module machinery. Then we set `max_cache_len` etc., and `register_buffer` the K/V/cumulative_length tensors. Inherits `.update()`, `.get_seq_length()`, `.get_mask_sizes()` from `StaticLayer` unchanged.
2. **`BufferStaticCache(StaticCache, nn.Module)`** — same idea at the cache level. After `StaticCache.__init__` populates `self.layers`, swap in `BufferStaticLayer` instances and wrap them in `nn.ModuleList` so torch.export sees the child-module tree.
3. **`_wrapper.build_static_cache(...)`** — returns `BufferStaticCache` instead of `StaticCache`. The wrapper's `__init__` already assigns `self.cache = cache`, so once `cache` is an `nn.Module`, it becomes a child module of the wrapper, and its buffers are part of the wrapper's `buffers()`.

Result: in the re-exported `.pt2`, the K/V tensors appear in `graph_signature.buffers` and `graph_signature.buffers_to_mutate` — and `run_decompositions` is happy.

### Implementation: `scripts/_buffer_cache.py`

`BufferStaticLayer(StaticLayer, nn.Module)`, `BufferStaticSlidingWindowLayer(StaticSlidingWindowLayer, nn.Module)`, `BufferStaticCache(StaticCache, nn.Module)`. Each layer registers `keys`, `values`, `cumulative_length` as buffers; the cache wraps layers in `nn.ModuleList`. Method dispatch (`update`, `get_seq_length`, etc.) inherits unchanged from the HF parent. `_wrapper.build_static_cache` was switched to return `BufferStaticCache`.

Eager sanity check confirmed 45 cache buffers properly registered (3 per layer × 15 layers; `num_kv_shared_layers=20` means cache holds only the first 15 of 35 decoder layers), and the wrapper's forward produced the right output shape.

### Gotcha: per-layer-type head_dim

First eager run after the shim crashed with `index_copy_(): Source/destination tensor must have same slice shapes. Destination slice shape: 1 1 256 at dimension 2 and source slice shape: 1 1 512 at dimension 0.`

Gemma 4's text config has both `head_dim=256` and `global_head_dim=512`. `Gemma4TextAttention.__init__` picks per-layer:

```python
self.head_dim = config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim
```

So sliding layers use `head_dim=256` and full_attention layers use `global_head_dim=512`. The original `StaticCache` lazily allocates the right shape per layer because it sees the K/V tensor shapes at first call. Our eager-allocated `BufferStaticCache` had to mirror this branch explicitly — sliding layers → `head_dim`, full layers → `global_head_dim` (or `head_dim` if `global_head_dim` not set).

Also, `num_kv_heads` can differ when `attention_k_eq_v=True` (uses `num_global_key_value_heads`); for Gemma 4 E2B `attention_k_eq_v=False`, so this branch is inactive, but the shim covers it for portability.

### Gotcha: `to_executorch()` fires autograd's leaf-with-grad check

After the buffer-cache fix, `to_edge_transform_and_lower` succeeded — XNNPACK accepted 683 subgraphs (the model partitioned cleanly). But `to_executorch()` failed with:

```
RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
While executing %copy__default_1 = call_function[target=torch.ops.aten.copy_.default](
    args = (%b_cache_layers_0_keys, %aten_index_put_default), kwargs = {})
```

Inspecting the `.pt2` confirmed all 87 buffers are `requires_grad=False` at save time. Inspecting `edge_pm.exported_program()` right before `to_executorch()` showed 0 buffers with `requires_grad=True`. So the source state is fine — but `to_executorch()` internally evaluates parts of the graph, and **something inside that evaluation runs under enabled autograd**, which trips on the in-place `copy_` to the buffer.

**Fix:** wrap `to_executorch()` in `torch.inference_mode()`. This disables autograd's tensor-version-counter machinery entirely, so the leaf-with-grad check doesn't fire even if intermediate tensors get marked requires_grad during graph eval.

```python
with torch.inference_mode():
    prog = edge_pm.to_executorch()
```

### Gotcha: `.pte` shape-specializes to the upper bound of the dynamic `Dim`

The exported `.pt2` had `seq` as a dynamic `Dim("seq", min=2, max=511)`. After lowering, the `.pte`'s method metadata reports input shapes as `[1, 511]` — fully specialized to the upper bound, not a dynamic range. ExecuTorch's memory planner pre-allocates buffers for the worst case.

Practical implication: at runtime we have to pad inputs to seq=511 (the actual prompt was 14 tokens). Padding tokens are masked out via `attention_mask=0` on positions [14, 510], so position 13's logits are still numerically correct.

For the Pi 5 C++ runner, the same constraint applies — it'll pad prompts to seq=511 for prefill (wasting compute proportional to `511 - actual_prompt_len`) and run decode at seq=1. Worth a note: future optimization could re-export with multiple `seq` buckets (e.g., 16, 64, 256, 511) and pick the smallest one that fits.

### Phase 5 results

```
running to_edge_transform_and_lower (XnnpackPartitioner)... finished in 648.3s (~11 min)
  partition summary:
    total graph nodes: 4998
    executorch_call_delegate calls: 683   ← XNNPACK-accepted subgraphs
    portable ops (top): getitem (1112), view_copy (388), mean.dim (242),
                        pow.Tensor_Scalar (242), expand_copy (218),
                        unsqueeze_copy (194), slice_copy (100),
                        where (70), scalar_tensor (85)...

  Serializing to .pte (with torch.inference_mode())... finished in 135.3s

prefill .pte: 12.18 GB  (1.10x reduction vs 13.41 GB .pt2)
decode  .pte: 12.18 GB
```

### Why isn't the `.pte` ~3 GB?

The original target ("8× compression via INT4") doesn't materialize on disk because:

1. **`embed_tokens_per_layer` is FP32**: 2.35 B params × 4 bytes ≈ **9.4 GB** of the `.pte`. torchao's filter only quantizes `nn.Linear` modules; this is `Gemma4TextScaledWordEmbedding` (an `nn.Embedding` subclass) and stays FP32. This is the right default — INT4-quantizing embeddings typically hurts quality noticeably.
2. **INT4 weights are still unpacked**: torchao's `IntxUnpackedToInt8Tensor` stores each INT4 in 1 INT8 byte. The XNNPACK lowering on Mac doesn't repack into true 4-bit. **True INT4 packing happens at runtime on ARM** when KleidiAI loads the weights and rewrites them into the tile layout its kernels expect. The `.pte` size on Mac is misleading — actual memory footprint on Pi 5 will be ~30% smaller.

Acceptable for Phase 5. Both `.pte`s are valid ExecuTorch programs.

### Cross-check: `.pte` produces the right token

[`scripts/09_verify_pte.py`](scripts/09_verify_pte.py) loads `gemma4_e2b_text_int4_prefill.pte` via ExecuTorch's Python runtime (`Runtime.get().load_program(...)`), runs prefill on the same chat-templated `"The capital of France is"` prompt (padded to seq=511), and reads logits at position 13.

```
.pte next token:    id=818 text='The'      ← matches FP32/INT4 reference
.pte top-5:         [818, 1018, 236777, 50429, 669]
INT4 eager top-5:   [818, 1018, 100, 236777, 6372]
```

Top-1 matches. Ranks 2–5 shuffle slightly vs INT4 eager — expected for a lowered quantized graph; the XNNPACK INT8 activation quantization rounds differently than torchao's eager `Int8DynamicActivationIntxWeightConfig`. For greedy decode (argmax) this doesn't matter; for sampled decode it would shift sampling probabilities marginally.

### Phase 5 deliverables

- [`scripts/_buffer_cache.py`](scripts/_buffer_cache.py) — nn.Module shim for `StaticCache` so cache K/V are buffers, not lifted constants
- [`scripts/05_lower.py`](scripts/05_lower.py) — XNNPACK lowering + `torch.inference_mode()` wrapper + partition summary
- [`scripts/09_verify_pte.py`](scripts/09_verify_pte.py) — load `.pte` via ExecuTorch Python runtime, run prefill, compare to reference
- [`results/05_lower_prefill.log`](results/05_lower_prefill.log) + [`results/05_lower_decode.log`](results/05_lower_decode.log) — full lowering logs with partition summaries
- [`results/09_verify_pte.log`](results/09_verify_pte.log) — `.pte` cross-check log
- `models/gemma4_e2b_text_int4_prefill.pte` (12.18 GB, gitignored)
- `models/gemma4_e2b_text_int4_decode.pte` (12.18 GB, gitignored)

### Carry-forward to Phase 6

- Both `.pte`s are valid and load cleanly in Mac-side Python runtime. They should load on Pi 5 too via the C++ runtime once we build ExecuTorch with KleidiAI enabled.
- Prefill `.pte` specializes to seq=511 → runner must pad prompts. Decode `.pte` specializes to seq=1 with `mask_len` ranging [2, 511] → runner grows the mask each step.
- KleidiAI weight-repack happens at first inference on Pi (or via an offline conversion step in the build). Watch for the first-load delay; benchmark separately.

## Phase 6 — Deployment prep + size reduction

### Architecture pivot: cache externalization

Phase 5 produced two `.pte` files (prefill, decode) but each ran as an independent ExecuTorch program with **its own internal buffer state**. After running prefill, the populated KV cache lives inside prefill's program memory; running decode (a separate program) starts with its own empty cache. Two `.pte`s can't share state, and ExecuTorch's Python runtime doesn't expose buffer reset. End-to-end generation across two programs was therefore broken.

Verified the bug empirically — `scripts/10_decode_only_check.py` running decode `.pte` token-by-token produced gibberish (`"## The user is a user is a user seems"` instead of `"The capital..."`).

**Resolution: externalize the cache.** Wrote [`scripts/_external_cache.py`](scripts/_external_cache.py) with `TransientCache` / `TransientCacheLayer` — *not* `nn.Module`s, they just hold references to externally-provided tensors. Wrote `TextWrapperExternal` in `_wrapper.py` that takes cache tensors as forward inputs and returns updated cache as outputs. Result: **stateless `.pte`** — runner allocates cache once, threads it across calls, single `.pte` handles both prompt token-by-token and decode.

Sanity-checked in eager via [`scripts/11_external_cache_check.py`](scripts/11_external_cache_check.py): 9/9 token match.

### The size problem

After re-exporting with the external-cache wrapper (still FP32, INT4-as-INT8 Linears only), the `.pte` was **12.18 GB** — *larger than the original FP16 HuggingFace download (~10 GB)*. Not a Pi-deployable artifact.

Breakdown of where the 12 GB went:

| Component | Params | Treatment | Size |
|---|---|---|---|
| `embed_tokens_per_layer` | 2.35 B | FP32 (torchao's `_is_linear` filter skips `nn.Embedding`) | **9.4 GB** |
| All `nn.Linear` weights | 3.1 B | INT4 logical, stored as INT8 (`IntxUnpackedToInt8Tensor`) | 3.1 GB |
| Per-group scales | — | FP32 | 0.1 GB |
| Other (`embed_tokens`, norms, biases, RoPE) | 0.06 B | FP32 | 0.25 GB |

The 2.35 B per-layer embedding (Gemma 4's "E2B" trick) dominates. Linear INT4 quant alone could not fix this.

### Why we couldn't just use `Int8WeightOnlyConfig` on embeds

First attempt: `quantize_(model, Int8WeightOnlyConfig(), filter_fn=lambda m, fqn: isinstance(m, nn.Embedding))`. Hit two roadblocks:

1. **`padding_idx`**: Gemma 4's embeddings have `padding_idx=0`. torchao's quantized embedding op asserts `padding_idx is None` at inference time (it only matters for gradients). Fix: null it before quantize.
2. **Direct 2D weight slicing**: `modeling_gemma4.py:2219` does `embed_tokens.weight[pad_token_id, :]` — `aten.select.int` on a 2D tensor. torchao's quantized tensor only implements `select` on 3D. Fix: skip `embed_tokens` (the small 0.4 GB one) and only quantize `embed_tokens_per_layer` (the 2.35 B monster).

With those fixes, Phase 4 succeeded eager (9/9 quality) and `.pt2` size dropped from 13 GB → 5.5 GB. But **Phase 5 lowering FAILED**:

```
RuntimeError: Missing out variants: {
    'torchao::quantize_affine',
    'torchao::choose_qparams_affine',
    'torchao::dequantize_affine',
}
```

torchao's `Int8WeightOnlyConfig` uses `AffineQuantizedTensor`, whose `quantize_affine` / `dequantize_affine` ops don't have ExecuTorch out-variants in 1.2.0. `to_executorch()` can't serialize them.

Tried `IntxUnpackedToInt8Tensor` (the same tensor type the Linear path uses successfully). Same error — the embedding lookup pattern doesn't get fused into the XNNPACK delegate the way the Linear matmul does, so the torchao::*affine ops stay in the portable graph.

### Resolution: hand-rolled `Int8Embedding`

[`scripts/_int8_embedding.py`](scripts/_int8_embedding.py) — replaces `nn.Embedding` with a tiny module that uses **only standard aten ops** (`index_select` + `to(dtype)` + `mul`) — all have ExecuTorch out-variants, all lower cleanly through XNNPACK.

- Per-row INT8 symmetric quantization (one scale per token row, FP32)
- Preserves Gemma's `embed_scale` attribute
- Replaces the module directly (`setattr(parent, name, Int8Embedding.from_embedding(old))`) instead of swapping the `.weight` tensor — sidesteps the torchao tensor-subclass issue entirely

**Final size:**

| Stage | `.pt2` | `.pte` |
|---|---|---|
| Phase 5 v1 (prefill+decode split, FP32, no embed quant) | 13.4 GB × 2 | 12.2 GB × 2 |
| Phase 6 external-cache (FP32, no embed quant) | 13.4 GB | 12.2 GB |
| Phase 6 + `Int8Embedding` (FP32) | **6.4 GB** | **5.14 GB** ✓ |

**Inference speedup on Mac (bonus):** 14-token prompt + 9 decode tokens, before vs after `Int8Embedding`:

| Stage | Before (FP32 BF16 embeds via torchao) | After (FP32 + Int8Embedding) | Speedup |
|---|---|---|---|
| Prompt feed | 1.71 tok/s | 7.73 tok/s | **4.5×** |
| Decode | 3.33 tok/s | 9.01 tok/s | **2.7×** |
| Total (23 tokens) | 10.88 s | 2.81 s | **3.9×** |

The FP32 → INT8 embed swap reduced memory bandwidth pressure dramatically. The previous version was likely RAM-bound during embed lookup.

### Why FP32 baseline (not BF16)

Tried BF16 baseline for another ~250 MB savings. Lowering failed with the same `torchao::quantize_affine` out-variant error — appears specific to BF16 + INT4 dynamic-activation Linears. FP32 + Int8Embedding lowers cleanly; BF16 + Int8Embedding doesn't. Parked the BF16 path; FP32 + Int8Embedding fits in Pi 5 RAM with room to spare, so the marginal BF16 win isn't worth chasing right now.

### Quality vs FP32 reference

9/9 token match preserved end-to-end through all of: Linear INT4 quant, `Int8Embedding` substitution, `torch.export`, XNNPACK lowering, ExecuTorch runtime. Same `"The capital of France is **Paris**."` from every stage.

### Phase 6 deliverables

- [`scripts/_external_cache.py`](scripts/_external_cache.py) — `TransientCache` / `TransientCacheLayer` (cache as inputs, not internal state)
- [`scripts/_int8_embedding.py`](scripts/_int8_embedding.py) — hand-rolled INT8 embedding using only ExecuTorch-compatible aten ops
- [`scripts/04_quantize.py`](scripts/04_quantize.py) — rewritten: external-cache wrapper + Int8Embedding + INT4 Linears; single `.pt2` output
- [`scripts/05_lower.py`](scripts/05_lower.py) — single `.pt2` → `.pte` lowering
- [`runner/pi_runner.py`](runner/pi_runner.py) — self-contained Pi inference runner (~250 lines, only depends on `executorch` + `transformers`)
- [`scripts/deploy_pi.sh`](scripts/deploy_pi.sh) — rsyncs minimum bundle (~5 GB) to Pi
- [`docs/pi5_setup.md`](docs/pi5_setup.md) — Pi prep guide (OS, deps, SSH, deployment flow)
- [`scripts/10_decode_only_check.py`](scripts/10_decode_only_check.py), [`scripts/11_external_cache_check.py`](scripts/11_external_cache_check.py) — bug-hunt + eager-mode cross-check scripts
- `models/gemma4_e2b_text_int4_extcache.pte` — **5.14 GB** (gitignored)

### Carry-forward to actual deployment

- Bundle to ship: `.pte` (5.14 GB) + `tokenizer/` (32 MB) + `pi_runner.py` (9 KB) + `requirements_pi.txt` (150 B). Total ~5.2 GB transfer.
- Pi-side: `pip install executorch transformers` (~2 GB deps), then `python pi_runner.py --verify`.
- Expected on Pi 5: decode ~3–8 tok/s depending on KleidiAI repacking behavior. Prompt feed slower (no batched prefill in this design — token-by-token).

## Phase 7 — Benchmarks

*(Pending)*

### Benchmark protocol
- 5 prompts × 3 runs each, report median
- CPU governor: `performance`
- Active cooling on
- Metrics: tokens/sec, time-to-first-token, peak RSS, Joules/token
- Same model, same quantization, same hardware for both ExecuTorch and llama.cpp

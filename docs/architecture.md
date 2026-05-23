# Gemma 4 E2B Architecture — An Exporter's Perspective

Notes on Gemma 4 E2B's architecture as it relates to `torch.export` and ExecuTorch deployment. This is not a general model card — it focuses on what matters for getting a `.pte` out the other end.

## Overview

Gemma 4 E2B is a multimodal model (text + vision + audio) with ~2B effective parameters. "E2B" refers to effective parameter count, not total — a significant portion of the parameter budget lives in embedding machinery rather than transformer blocks.

**Model ID:** `google/gemma-4-e2b-it`
**Total download:** ~10.2 GB (FP16 safetensors)
**Modalities:** Text, image (via SigLIP-style vision tower), audio (via subsample conv encoder)

## Module hierarchy

```
model
├── language_model
│   ├── embed_tokens               # Standard token embedding
│   ├── layers                     # Transformer decoder blocks
│   ├── norm                       # Final RMSNorm
│   ├── rotary_emb                 # Shared RoPE (sibling of layers, not per-block)
│   ├── embed_tokens_per_layer     # NEW in Gemma 4: per-layer embeddings
│   ├── per_layer_model_projection # NEW in Gemma 4: per-layer projection
│   └── per_layer_projection_norm  # NEW in Gemma 4: per-layer projection norm
├── vision_tower
│   ├── patch_embedder
│   ├── encoder
│   └── pooler
├── embed_vision
│   ├── embedding_projection
│   └── embedding_pre_projection_norm
├── audio_tower
│   ├── subsample_conv_projection
│   ├── rel_pos_enc
│   ├── layers
│   └── output_proj
├── embed_audio
│   ├── embedding_projection
│   └── embedding_pre_projection_norm
└── lm_head                        # Output projection to vocab
```

## Export-relevant observations

### 1. Per-layer embedding machinery (Gemma 4 specific)

The three new modules (`embed_tokens_per_layer`, `per_layer_model_projection`, `per_layer_projection_norm`) are the most likely source of export issues. If they use data-dependent indexing (e.g. `torch.index_select` with a dynamic layer index) or Python control flow over tensor values, `torch.export` will reject them.

**Action:** Inspect source code via `02_inspect.py`. If problematic, refactor to use static indexing in the wrapper.

### 2. Shared RoPE

`rotary_emb` lives as a sibling of `layers`, not inside each attention block. This means:
- RoPE is computed once and passed to all layers (good for export — no repeated computation)
- But the forward signature may expect `position_ids` to be passed through, which adds a required input to the wrapper

If RoPE uses `complex64` internally (as Gemma 3 did), this will need patching — ExecuTorch's portable ops don't fully support complex types.

### 3. Sliding window attention

Gemma models use sliding window attention on some layers and global attention on others. This creates:
- Variable `attention_mask` shapes depending on layer type
- Potential Python `if` branches in the attention implementation

**Action:** Check `sliding_window` config field. If present, the attention mask construction will need careful handling in the export wrapper.

### 4. Attention logit softcapping

Gemma uses logit softcapping (`tanh(logits / cap) * cap`) instead of standard attention scaling. This is a single extra op and should export cleanly, but verify it doesn't use in-place operations that trip the exporter.

### 5. Vision and audio towers

These are the Phase 9 stretch goal. For initial export (Phase 3), skip them entirely — export only the text path:
`embed_tokens → layers → norm → lm_head`

The vision tower's variable resolution handling (70/140/280/560/1120 image tokens) will be the hardest part of multimodal export.

## Export strategy

### Phase 3 target: text-only path

```
ExportableGemma4(nn.Module):
    forward(input_ids, attention_mask, position_ids, kv_cache) → logits
```

- Static KV cache (pre-allocated, max_seq=2048)
- Symbolic seq dimension only
- No vision/audio towers
- Handle per-layer embedding modules explicitly

### Phase 9 target: add vision

```
ExportableGemma4Multimodal(nn.Module):
    forward(input_ids, attention_mask, position_ids, kv_cache, pixel_values) → logits
```

Audio is Phase 9+ or a separate writeup.

## Config fields (from `results/02_inspect.txt`)

| Field | Value |
|---|---|
| hidden_size | 1536 |
| num_hidden_layers | 35 |
| num_attention_heads | 8 |
| num_key_value_heads | 1 |
| head_dim | 256 |
| intermediate_size | 6144 |
| vocab_size | 262,144 |
| max_position_embeddings | 131,072 |
| sliding_window | 512 |
| final_logit_softcapping | 30.0 |
| hidden_activation | gelu_pytorch_tanh |

GQA ratio is 8:1 (8 attention heads, 1 KV head). Sliding window is 512 — confirms the per-layer `sliding_attention` vs `full_attention` split, which the rotary embedding uses to select the correct `inv_freq`/`attention_scaling` tensor.

## Parameter distribution

| Component | Params |
|---|---|
| model.language_model | 4,628,569,344 |
| model.vision_tower | 167,364,608 |
| model.audio_tower | 304,824,608 |
| model.embed_vision | 1,179,648 |
| model.embed_audio | 2,359,296 |
| lm_head | 402,653,184 |
| **TOTAL** | **5,506,950,688** |

Of the 4.63 B in `language_model`, **2.35 B sits in `embed_tokens_per_layer`** — the Gemma 4 "effective 2B" trick is per-token-sparse activation of a large per-layer embedding table. Active params per forward pass are much smaller than total; the name `E2B` refers to effective active params.

## Forward signatures (export-critical)

```
language_model.forward(
    input_ids, attention_mask, position_ids,
    past_key_values: Cache | None,
    inputs_embeds, per_layer_inputs,
    use_cache, **kwargs
) -> BaseModelOutputWithPast
```

```
Gemma4TextDecoderLayer.forward(
    hidden_states,
    per_layer_input: Tensor = None,
    shared_kv_states: dict[int, tuple[Tensor, Tensor]] | None = None,  # export hazard
    position_embeddings: Tensor = None,
    attention_mask, position_ids,
    past_key_values: Cache | None,                                     # export hazard
    **kwargs
)
```

```
Gemma4TextRotaryEmbedding.forward(x, position_ids, layer_type=None)
  -> inv_freq = getattr(self, f"{layer_type}_inv_freq")               # export hazard
  -> attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
```

## Export hazards (confirmed from source)

1. **`shared_kv_states: dict[int, tuple[Tensor, Tensor]]`** — dict inputs are not first-class in `torch.export`. The wrapper must flatten this into positional tensor args, one per layer that needs shared KV, or skip the shared-KV optimization for export.
2. **`Cache` subclass** (`past_key_values`) — transformers' `DynamicCache` uses Python list mutation. Use `StaticCache` (pre-allocated) or replace entirely with raw KV tensors.
3. **`getattr(self, f"{layer_type}_inv_freq")`** — dynamic attribute lookup based on a string keyed by layer type. `torch.export` won't trace this. Either (a) bind two parallel RoPE modules and select statically in the wrapper, or (b) pre-compute cos/sin tensors outside the graph and pass them in.
4. **`@dynamic_rope_update`** decorator — wraps the rotary forward with logic that may mutate state across calls; needs inspection for Python branches on tensor values.
5. **`maybe_autocast(device_type=x.device.type ...)`** — branching on `x.device.type` is a Python conditional that the tracer will specialize on. Acceptable if the device is fixed at export time (it will be).
6. **`@dynamic_rope_update` + per-call inv_freq selection** combined: the rotary module is essentially a polymorphic dispatch over layer type. Cleanest fix is to pre-compute (cos_full, sin_full, cos_sliding, sin_sliding) once at the start of the wrapper's forward and pass the appropriate pair to each layer.
7. **`final_logit_softcapping = 30.0`** — single `tanh(x / 30.0) * 30.0`. Trivial, should export cleanly.
8. **Sliding vs full attention mask** — layer_type is a static config field, so an `if`-branch in the wrapper that constructs both masks once and selects per-layer should trace fine.

## Export reality (Phase 3)

None of hazards 1–6 above actually fired during export. transformers 5.5.3's Gemma 4 implementation + `StaticCache` traces through `torch.export.export` cleanly with the `TextOnlyWrapper` defined in `scripts/_wrapper.py`. The HF internal code paths that *use* the hazardous patterns (dict-typed `shared_kv_states`, dynamic `getattr` in rotary, `@dynamic_rope_update`) either take the inactive branch under `StaticCache` or get specialized away by the tracer.

The only real constraint encountered: `sliding_window=512` generates the symbolic-shape guard `seq < 512`. Two consequences:

- Prefill graph: `Dim("seq", min=2, max=511)` — single forward processes ≤ 510 prompt tokens.
- Decode graph: `seq=1` (static), `Dim("mask_len", min=2, max=511)` — `attention_mask` grows by one each decode step to cover all valid cache slots.

Both shapes were verified bit-exact against eager (see `scripts/06_verify.py`, `scripts/08_verify_int4.py`).

## Quantization layout (Phase 4)

Scheme: **INT8 dynamic activations + INT4 weights**, per-group scales with `group_size=128`. Applied via:

```python
quantize_(
    model,
    Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int4,
        weight_granularity=PerGroup(group_size=128),
    ),
)
```

`torchao 0.17`'s default `Int4WeightOnlyConfig` is CUDA-only (requires Meta's `mslk` TINYGEMM kernel library, not available on CPU). The `Int8DynamicActivationIntxWeightConfig` path is the CPU-supported one and matches what ExecuTorch's ARM backend (KleidiAI) expects downstream — same scheme XNNPACK uses on mobile.

### What gets quantized
torchao's default filter is `_is_linear` — only `nn.Linear` modules are touched. For Gemma 4 E2B this means:

| Module class | Action | Approx params |
|---|---|---|
| `nn.Linear` (q/k/v/o proj, gate/up/down MLP, lm_head, per_layer_model_projection) | INT4 weight | ~2.7 B |
| `Gemma4TextScaledWordEmbedding` (`embed_tokens`, `embed_tokens_per_layer`) | left FP32 | ~2.5 B |
| `Gemma4RMSNorm` (all `*_layernorm`, `q_norm`, `k_norm`) | left FP32 | < 1 M |
| Layer biases, rotary buffers | left FP32 | small |

The large embedding tables remain FP32 by default. This is typically the appropriate choice — INT4 quantization of embeddings can noticeably shift token-id distributions. For Gemma 4 E2B's 2.35 B `embed_tokens_per_layer`, this decision could be revisited if disk or RAM pressure later requires it.

### Storage layout
Quantized weights become `IntxUnpackedToInt8Tensor` instances. The 4-bit values are stored unpacked — one INT4 weight occupies one INT8 byte (high 4 bits zero). No compression vs INT8 storage. The real 4-bit packing is deferred to ExecuTorch lowering (Phase 5), where the ARM backend repacks weights into KleidiAI's tile layout.

Per-quantized-Linear, torchao adds:
- INT8 weight tensor of shape `[out, in]` (1 byte/elem)
- FP32 scale tensor of shape `[out, in/group_size]`
- INT8 zero-point tensor of shape `[out, in/group_size]`

Counter-intuitive result from `model_size_bytes()`: the in-memory footprint *grows* slightly post-quantization (22.0 GB vs 20.4 GB FP32). The scales/zeros overhead per group plus the unpacked INT8 storage cancels the per-weight savings. The win comes from cheaper INT8×INT4 matmul kernels, not from holding less data in RAM. The on-disk `.pt2` is smaller (13.4 GB vs 18.5 GB FP32) because the saved tensors compress better.

### Saved artifacts
- `models/gemma4_e2b_text_int4_prefill.pt2` — dynamic `seq ∈ [2, 511]`, 13.41 GB
- `models/gemma4_e2b_text_int4_decode.pt2` — static `seq=1`, dynamic `mask_len ∈ [2, 511]`, 13.41 GB

Both round-trip through `torch.export.save` / `load`. Both have been numerically verified against the INT4 eager wrapper (`scripts/08_verify_int4.py`): bit-exact (max diff = 0.0).

## ExecuTorch lowering (Phase 5)

Both `.pt2`s are lowered through XNNPACK to portable `.pte` programs that run in the ExecuTorch runtime.

**Backend choice.** ExecuTorch 1.2.0's `executorch.backends.arm` is **Ethos-U NPU**-only (TOSA, Vela, Corstone). The Pi 5 has a Cortex-A76 *CPU* and no NPU, so we use `XnnpackPartitioner` instead. On ARM, XNNPACK auto-dispatches INT4/INT8 matmul to **KleidiAI** kernels — same fast path the README originally anticipated, reached through XNNPACK rather than the ARM backend.

### Cache contract for export-then-lower
`transformers.cache_utils.StaticCache` is not an `nn.Module`. Its K/V/cumulative_length tensors get lifted as `inputs_to_lifted_tensor_constants` by `torch.export`. Subsequent `run_decompositions` (called inside `to_edge_transform_and_lower`) rejects mutated constants.

`scripts/_buffer_cache.py` provides `BufferStaticCache(StaticCache, nn.Module)` plus `BufferStaticLayer` / `BufferStaticSlidingWindowLayer` variants. Each:
- Calls `nn.Module.__init__` first, skips parent `__init__` (which would set bare-tensor attrs).
- Eagerly allocates `keys`, `values`, `cumulative_length` and registers them as buffers (`requires_grad=False`).
- Inherits `update`, `get_seq_length`, etc. from the HF parent — only the registration mechanics change.
- Cache wraps layers in `nn.ModuleList` so torch.export sees the child-module tree.

Per-layer-type subtlety: Gemma 4 uses `head_dim=256` on sliding layers and `global_head_dim=512` on full_attention layers. The HF `StaticCache` lazy-allocates from the first call's shapes; the buffer cache mirrors this branch explicitly in `BufferStaticCache.__init__`.

### Lowering quirks
- `to_executorch()` triggers autograd's leaf-with-grad check on cache buffer copy_ ops, even though the buffers themselves are `requires_grad=False`. Wrap the call in `torch.inference_mode()`.
- The `.pte` shape-specializes to the upper bound of the dynamic `Dim` — `seq=511` for prefill. Runtime must pad shorter prompts and `attention_mask=0` the padding.
- 683 subgraphs delegate to XNNPACK on Mac. Unquantized weights (embeddings, layernorms, RoPE buffers) stay portable. True 4-bit weight packing happens at runtime on ARM via KleidiAI's load-time repack — Mac `.pte` size doesn't reflect Pi 5 in-memory footprint.

### `.pte` artifacts
- `models/gemma4_e2b_text_int4_prefill.pte` — 12.18 GB, specialized to seq=511
- `models/gemma4_e2b_text_int4_decode.pte` — 12.18 GB, static seq=1, dynamic `mask_len` collapsed to 511

Both verified via `scripts/09_verify_pte.py`: load with `Runtime.get().load_program(path, verification=Verification.Minimal)`, run prefill on the chat-templated reference prompt, get `id=818 "The"` as next token — matches FP32/INT4 references.

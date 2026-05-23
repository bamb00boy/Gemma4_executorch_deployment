"""
Hand-rolled INT8 embedding that lowers through ExecuTorch.

WHY THIS EXISTS
---------------
torchao's quantized embedding paths (Int8WeightOnlyConfig and
IntxUnpackedToInt8Tensor swapped onto the Embedding's .weight) both
work in eager mode but FAIL at `to_executorch()` with:

    RuntimeError: Missing out variants: {
        'torchao::quantize_affine',
        'torchao::choose_qparams_affine',
        'torchao::dequantize_affine',
    }

torchao's quantized-tensor ops don't have ExecuTorch out-variant
implementations registered. We need a quantized embedding that uses
only standard aten ops (index_select, cast, mul) — which all have
out-variants and lower cleanly through XNNPACK.

Int8Embedding stores INT8 row-quantized weights + per-row BF16 scales
as buffers. Forward: gather INT8 rows by input_ids, cast to BF16,
multiply by per-row scales. Optionally apply Gemma's `embed_scale`.

The big win: embed_tokens_per_layer is 2.35 B params. Per-row INT8 with
BF16 scale = 2.35 GB + ~0.5 MB scales, vs 4.7 GB BF16 or 9.4 GB FP32.
"""

import torch
import torch.nn as nn


class Int8Embedding(nn.Module):
    """Drop-in replacement for nn.Embedding with INT8 per-row quantized weight.

    Uses only standard aten ops in forward — no torchao custom ops, so
    ExecuTorch's `to_executorch()` accepts it.
    """

    def __init__(self, num_embeddings, embedding_dim, weight_int8, scale,
                 compute_dtype=torch.bfloat16, embed_scale=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compute_dtype = compute_dtype
        # INT8 weight: (vocab, hidden)
        self.register_buffer("weight_int8", weight_int8.contiguous())
        # Per-row scale: (vocab,) — one scale per token row
        self.register_buffer("scale", scale.to(compute_dtype).contiguous())
        # Optional outer scale (Gemma's `embed_scale`); applied at the end.
        if embed_scale is not None:
            self.register_buffer("embed_scale", embed_scale.to(compute_dtype).contiguous())
            self._has_embed_scale = True
        else:
            self._has_embed_scale = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (batch, seq) long -> (batch, seq, hidden) compute_dtype."""
        # All standard ops: index_select (via fancy indexing), to (cast), mul.
        q_rows = self.weight_int8[input_ids]                # int8 (batch, seq, hidden)
        row_scales = self.scale[input_ids]                  # bf16 (batch, seq)
        out = q_rows.to(self.compute_dtype) * row_scales.unsqueeze(-1)
        if self._has_embed_scale:
            out = out * self.embed_scale
        return out

    @classmethod
    def from_embedding(cls, embed_module: nn.Embedding,
                       compute_dtype=torch.bfloat16) -> "Int8Embedding":
        """Quantize an existing nn.Embedding (or subclass) to per-row INT8.

        If the source is a Gemma-style `*ScaledWordEmbedding` with an
        `embed_scale` attribute, the scale is preserved.
        """
        weight = embed_module.weight.data.to(torch.float32)
        vocab, hidden = weight.shape
        # Per-row symmetric absmax quantization to INT8 ([-127, 127] for safety)
        abs_max = weight.abs().amax(dim=1)                  # (vocab,)
        scale = (abs_max / 127.0).clamp(min=1e-8)           # avoid div-by-zero
        weight_int8 = (weight / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
        embed_scale = getattr(embed_module, "embed_scale", None)
        if embed_scale is not None:
            embed_scale = embed_scale.detach().to(torch.float32)
        return cls(
            num_embeddings=vocab,
            embedding_dim=hidden,
            weight_int8=weight_int8,
            scale=scale,
            compute_dtype=compute_dtype,
            embed_scale=embed_scale,
        )


def replace_embedding(model: nn.Module, fqn: str, new_module: nn.Module) -> None:
    """Replace a module by fully-qualified name (e.g. 'language_model.embed_tokens_per_layer')."""
    parts = fqn.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)

"""
nn.Module-backed replacement for transformers.cache_utils.StaticCache.

WHY THIS EXISTS
---------------
`transformers.cache_utils.StaticCache` and its layer types (`StaticLayer`,
`StaticSlidingWindowLayer`) are plain Python objects, not `nn.Module`s.
Their `keys` / `values` / `cumulative_length` tensors are bare attributes.

When `torch.export` traces a wrapper that uses `StaticCache`, those tensors
get *lifted* as constants (entries in `inputs_to_lifted_tensor_constants`)
because the tracer can't tie them to any registered parameter or buffer.
ExecuTorch's `to_edge_transform_and_lower` later calls `run_decompositions`,
which errors out:

    RuntimeError: Constant ...self_attn.lifted_tensor_0 is mutated in the
    forward method. Pls register it as buffer

This module wraps each layer (and the cache itself) as `nn.Module`s with
properly registered buffers. The buffers behave the same way the bare
attributes did — only the registration metadata changes, which is exactly
what `torch.export` needs to record the K/V tensors as mutable buffers
(in `buffers_to_mutate`) instead of constants.

USAGE
-----
Drop-in replacement for StaticCache:

    from _buffer_cache import BufferStaticCache
    cache = BufferStaticCache(config=model.config.text_config,
                              max_batch_size=1, max_cache_len=512,
                              device='cpu', dtype=torch.float32)
"""

import torch
import torch.nn as nn
from transformers.cache_utils import StaticCache, StaticLayer, StaticSlidingWindowLayer


class BufferStaticLayer(StaticLayer, nn.Module):
    """nn.Module variant of StaticLayer with K/V/cumulative_length as buffers.

    Skips the parent's lazy initialization (which would set bare-tensor
    attrs); allocates the cache tensors eagerly via `register_buffer` so
    they show up in the parent module's `buffers()` and the resulting
    ExportedProgram records them in `graph_signature.buffers`.
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len, max_batch_size, num_kv_heads, head_dim, dtype, device="cpu"):
        nn.Module.__init__(self)
        # Set StaticLayer state manually — skip its __init__ so we don't
        # get a bare-tensor `cumulative_length`.
        self.max_cache_len = max_cache_len
        self.max_batch_size = max_batch_size
        self.num_heads = num_kv_heads
        self.k_head_dim = head_dim
        self.v_head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.is_initialized = True  # eager init below, no lazy path needed

        # Explicit requires_grad_(False): defensive. torch.zeros already
        # returns a no-grad tensor, but ExecuTorch's to_executorch() pass
        # complains about leaf buffers with requires_grad=True being used
        # in in-place ops if anything upstream flipped this flag.
        self.register_buffer(
            "cumulative_length",
            torch.zeros(1, dtype=torch.int64, device=self.device).requires_grad_(False),
        )
        self.register_buffer(
            "keys",
            torch.zeros(
                (max_batch_size, num_kv_heads, max_cache_len, head_dim),
                dtype=dtype, device=self.device,
            ).requires_grad_(False),
        )
        self.register_buffer(
            "values",
            torch.zeros(
                (max_batch_size, num_kv_heads, max_cache_len, head_dim),
                dtype=dtype, device=self.device,
            ).requires_grad_(False),
        )

    def lazy_initialization(self, key_states, value_states):
        # Already initialized eagerly — no-op.
        return


class BufferStaticSlidingWindowLayer(StaticSlidingWindowLayer, nn.Module):
    """nn.Module variant of StaticSlidingWindowLayer with buffers."""

    is_compileable = True
    is_sliding = True

    def __init__(self, max_cache_len, sliding_window, max_batch_size, num_kv_heads, head_dim, dtype, device="cpu"):
        nn.Module.__init__(self)
        effective_max_cache_len = min(sliding_window, max_cache_len)
        # StaticLayer state
        self.max_cache_len = effective_max_cache_len
        self.max_batch_size = max_batch_size
        self.num_heads = num_kv_heads
        self.k_head_dim = head_dim
        self.v_head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.is_initialized = True
        # Sliding-window state (Python int — not exported, doesn't need to be a buffer)
        self.cumulative_length_int = 0

        self.register_buffer(
            "cumulative_length",
            torch.zeros(1, dtype=torch.int64, device=self.device),
        )
        self.register_buffer(
            "keys",
            torch.zeros(
                (max_batch_size, num_kv_heads, effective_max_cache_len, head_dim),
                dtype=dtype, device=self.device,
            ),
        )
        self.register_buffer(
            "values",
            torch.zeros(
                (max_batch_size, num_kv_heads, effective_max_cache_len, head_dim),
                dtype=dtype, device=self.device,
            ),
        )

    def lazy_initialization(self, key_states, value_states):
        return


class BufferStaticCache(StaticCache, nn.Module):
    """
    Drop-in replacement for `StaticCache` that's an `nn.Module` with
    child-module layers and buffer-backed K/V state.

    Mirrors `StaticCache.__init__`'s layer-type dispatch (sliding vs full)
    but builds `BufferStaticLayer` / `BufferStaticSlidingWindowLayer`
    instead of the bare-attribute originals. Layers go into an
    `nn.ModuleList` so `torch.export` sees the child-module tree.
    """

    def __init__(self, config, max_batch_size=1, max_cache_len=512, device="cpu", dtype=torch.float32, **kwargs):
        nn.Module.__init__(self)

        cfg = config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config

        layer_types = getattr(cfg, "layer_types", None)
        if layer_types is None:
            if getattr(cfg, "sliding_window", None) is not None:
                layer_types = ["sliding_attention"] * cfg.num_hidden_layers
            elif getattr(cfg, "attention_chunk_size", None) is not None:
                layer_types = ["chunked_attention"] * cfg.num_hidden_layers
            else:
                layer_types = ["full_attention"] * cfg.num_hidden_layers

        n_shared = getattr(cfg, "num_kv_shared_layers", 0)
        if n_shared and n_shared > 0:
            layer_types = layer_types[:-n_shared]

        # Gemma 4 (and similar) use *different head_dim per layer type*:
        #   sliding layers -> head_dim
        #   full layers    -> global_head_dim (if set, else head_dim)
        # Same logic as Gemma4TextAttention.__init__.
        sliding_head_dim = cfg.head_dim
        full_head_dim = getattr(cfg, "global_head_dim", None) or cfg.head_dim

        # num_kv_heads can also differ per layer type when attention_k_eq_v is True.
        attention_k_eq_v = getattr(cfg, "attention_k_eq_v", False)
        sliding_num_kv = cfg.num_key_value_heads
        full_num_kv = (
            getattr(cfg, "num_global_key_value_heads", None) or cfg.num_key_value_heads
        ) if attention_k_eq_v else cfg.num_key_value_heads

        layers = []
        for lt in layer_types:
            if lt == "sliding_attention":
                layer = BufferStaticSlidingWindowLayer(
                    max_cache_len=max_cache_len,
                    sliding_window=cfg.sliding_window,
                    max_batch_size=max_batch_size,
                    num_kv_heads=sliding_num_kv,
                    head_dim=sliding_head_dim,
                    dtype=dtype,
                    device=device,
                )
            elif lt == "chunked_attention":
                layer = BufferStaticSlidingWindowLayer(
                    max_cache_len=max_cache_len,
                    sliding_window=cfg.attention_chunk_size,
                    max_batch_size=max_batch_size,
                    num_kv_heads=sliding_num_kv,
                    head_dim=sliding_head_dim,
                    dtype=dtype,
                    device=device,
                )
            else:
                layer = BufferStaticLayer(
                    max_cache_len=max_cache_len,
                    max_batch_size=max_batch_size,
                    num_kv_heads=full_num_kv,
                    head_dim=full_head_dim,
                    dtype=dtype,
                    device=device,
                )
            layers.append(layer)

        # ModuleList so torch.export sees each layer as a child module
        self.layers = nn.ModuleList(layers)

        # Cache-base attrs that Cache.__init__ would normally set
        self.layer_class_to_replicate = None
        self.offloading = False
        self.offload_only_non_sliding = True

"""
Externalized KV cache for stateless ExecuTorch deployment.

WHY THIS EXISTS
---------------
Phase 5's `BufferStaticCache` (`_buffer_cache.py`) made the K/V tensors
proper `nn.Module` buffers — necessary for `to_edge_transform_and_lower`
to lower the graph without complaining about mutated constants. But it
left a different problem: those buffers are *part of the .pte's saved
state*. Once a .pte is loaded, its cache buffers start at whatever values
they had at `torch.export.save` time, and ExecuTorch's Python runtime
doesn't expose a way to reset them. Two .pte files can't share state
either, so a prefill→decode pipeline across two .pte files can't work.

`TransientCache` and `TransientCacheLayer` push the cache state *out*
of the model. They hold references to externally-provided tensors (graph
inputs) instead of registering buffers. The wrapper that uses them
becomes a pure function:

    forward(input_ids, ..., k_caches, v_caches, cumlen_caches)
        -> logits, k_caches', v_caches', cumlen_caches'

The runner allocates cache tensors once, then threads them across calls.
A single .pte (seq=1) is used for both prompt token-by-token feed
("prefill, slow") and generation ("decode, normal"). Same .pte handles
the entire generation; no state leaks between calls because there IS no
saved state.

This pattern is sometimes called "stateless KV-cache" or "external KV".
"""

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


class TransientCacheLayer(CacheLayerMixin):
    """A cache layer that points to externally-provided tensors.

    Deliberately NOT an nn.Module — its tensors come in as graph inputs
    (via the wrapper's forward), not as registered buffers. torch.export
    treats them as `user_inputs_to_mutate`.

    Mimics enough of `transformers.cache_utils.StaticLayer` /
    `StaticSlidingWindowLayer` that the language model's attention code
    (which calls `cache.layers[layer_idx].update(...)`) just works.
    """

    is_compileable = True

    def __init__(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        cumulative_length: torch.Tensor,
        max_cache_len: int,
        max_batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device,
        is_sliding: bool,
    ):
        self.keys = keys
        self.values = values
        self.cumulative_length = cumulative_length
        self.max_cache_len = max_cache_len
        self.max_batch_size = max_batch_size
        self.num_heads = num_kv_heads
        self.k_head_dim = head_dim
        self.v_head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.is_initialized = True  # tensors are real, no lazy alloc needed
        self.is_sliding = is_sliding
        # For sliding-window layers, transformers' StaticSlidingWindowLayer
        # tracks a Python int as well. We always run with cumulative_length
        # < max_cache_len (we cap seq < 512), so the "not full" branch is
        # static at trace time.
        self.cumulative_length_int = 0

    def lazy_initialization(self, key_states, value_states):
        # Tensors already exist — no-op.
        return

    def update(self, key_states, value_states, *args, **kwargs):
        """Mirrors StaticLayer.update — in-place index_copy_ + cumulative
        length increment. The mutations are visible to torch.export and
        get recorded as input mutations on the external tensors."""
        kv_length = key_states.shape[-2]
        cache_position = torch.arange(kv_length, device=self.device) + self.cumulative_length
        self.cumulative_length.add_(kv_length)
        self.keys.index_copy_(2, cache_position, key_states)
        self.values.index_copy_(2, cache_position, value_states)
        return self.keys, self.values

    def get_seq_length(self):
        return self.cumulative_length

    def get_mask_sizes(self, query_length: int):
        return self.max_cache_len, 0

    def get_max_cache_shape(self):
        return self.max_cache_len


class TransientCache(Cache):
    """Cache shim that holds externally-provided per-layer tensors.

    Construct fresh inside the wrapper's forward each call. Cheap (just
    Python object construction; the tensors are not copied).
    """

    def __init__(self, layer_specs, k_caches, v_caches, cumlen_caches):
        # Don't call Cache.__init__ — it would try to set self.layers
        # before we're ready. Build our layers list directly.
        self.layers = [
            TransientCacheLayer(
                keys=k_caches[i],
                values=v_caches[i],
                cumulative_length=cumlen_caches[i],
                **layer_specs[i],
            )
            for i in range(len(layer_specs))
        ]
        self.layer_class_to_replicate = None
        self.offloading = False


def compute_layer_specs(model, max_batch_size=1, max_cache_len=512, dtype=torch.float32, device="cpu"):
    """Walk the model's config and return per-cache-layer metadata.

    Mirrors the layer-type dispatch in `_buffer_cache.BufferStaticCache.__init__`:
      - 15 cache layers (35 decoder layers, minus 20 kv-shared)
      - sliding layers: head_dim
      - full layers: global_head_dim (or head_dim if not set)
    """
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config

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

    sliding_head_dim = cfg.head_dim
    full_head_dim = getattr(cfg, "global_head_dim", None) or cfg.head_dim
    sliding_num_kv = cfg.num_key_value_heads
    full_num_kv = cfg.num_key_value_heads  # attention_k_eq_v=False for Gemma 4 E2B

    specs = []
    for lt in layer_types:
        if lt in ("sliding_attention", "chunked_attention"):
            specs.append(dict(
                max_cache_len=max_cache_len,
                max_batch_size=max_batch_size,
                num_kv_heads=sliding_num_kv,
                head_dim=sliding_head_dim,
                dtype=dtype,
                device=device,
                is_sliding=True,
            ))
        else:
            specs.append(dict(
                max_cache_len=max_cache_len,
                max_batch_size=max_batch_size,
                num_kv_heads=full_num_kv,
                head_dim=full_head_dim,
                dtype=dtype,
                device=device,
                is_sliding=False,
            ))
    return specs


def allocate_cache_tensors(layer_specs):
    """Allocate zero-initialized cache tensors for all layers.

    Returns three parallel lists (k, v, cumlen) of length len(layer_specs).
    Each runner-side call passes these in; the wrapper returns mutated copies.
    """
    k_caches, v_caches, cumlen_caches = [], [], []
    for spec in layer_specs:
        shape = (
            spec["max_batch_size"], spec["num_kv_heads"],
            spec["max_cache_len"], spec["head_dim"],
        )
        k_caches.append(torch.zeros(shape, dtype=spec["dtype"], device=spec["device"]))
        v_caches.append(torch.zeros(shape, dtype=spec["dtype"], device=spec["device"]))
        cumlen_caches.append(torch.zeros(1, dtype=torch.int64, device=spec["device"]))
    return k_caches, v_caches, cumlen_caches

"""
Shared text-only wrapper for export + verification.

`import _paths` must run BEFORE this module is imported (it sets the
HF/torch cache env vars). Callers handle that.
"""

import torch
import torch.nn as nn
from transformers.cache_utils import StaticCache

from _buffer_cache import BufferStaticCache
from _external_cache import TransientCache

MAX_SEQ = 512  # capped by sliding_window=512; see RESULTS.md


class TextOnlyWrapper(nn.Module):
    """Text-only wrapper around model.language_model + lm_head.

    Forward takes plain tensors only:
        input_ids       [batch, seq]      long
        attention_mask  [batch, seq]      long  (1 = attend, 0 = pad)
        position_ids    [batch, seq]      long
        cache_position  [seq]             long  (cache slots to write)
    Returns:
        logits          [batch, seq, vocab]
    """

    def __init__(self, model, cache: StaticCache):
        super().__init__()
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head
        self.cache = cache
        cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
        cap = getattr(cfg, "final_logit_softcapping", None)
        self.register_buffer(
            "_softcap",
            torch.tensor(cap if cap is not None else 0.0, dtype=torch.float32),
        )
        self._has_softcap = cap is not None and cap > 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        out = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=self.cache,
            cache_position=cache_position,
            use_cache=True,
        )
        hidden = out.last_hidden_state
        logits = self.lm_head(hidden)
        if self._has_softcap:
            cap = self._softcap
            logits = torch.tanh(logits / cap) * cap
        return logits


class TextWrapperExternal(nn.Module):
    """Text-only wrapper with externalized KV cache.

    Cache state is passed in as forward inputs and updated cache is
    returned as forward outputs — no buffers, no state lives inside the
    module/.pte. The runner allocates cache tensors once and threads
    them across calls. Same .pte handles both token-by-token prompt
    feed AND generation; no two-program state-sync problem.

    Forward inputs:
        input_ids       [1, 1]                                long
        attention_mask  [1, mask_len]                         long  (1 = attend, 0 = pad)
        position_ids    [1, 1]                                long
        cache_position  [1]                                   long
        k_caches        list[Tensor]  (one per cache layer)
        v_caches        list[Tensor]  (one per cache layer)
        cumlen_caches   list[Tensor]  (one per cache layer, shape [1] int64)
    Returns:
        logits          [1, 1, vocab]
        k_caches'       (mutated in place; returned to thread to next call)
        v_caches'       same
        cumlen_caches'  same
    """

    def __init__(self, model, layer_specs):
        super().__init__()
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head
        self.layer_specs = layer_specs  # list of dicts, static metadata per cache layer
        cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
        cap = getattr(cfg, "final_logit_softcapping", None)
        self.register_buffer(
            "_softcap",
            torch.tensor(cap if cap is not None else 0.0, dtype=torch.float32),
        )
        self._has_softcap = cap is not None and cap > 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        k_caches: list,
        v_caches: list,
        cumlen_caches: list,
    ):
        # Wrap the input tensors in a transient cache the LM can call .update() on.
        cache = TransientCache(self.layer_specs, k_caches, v_caches, cumlen_caches)
        out = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
        )
        hidden = out.last_hidden_state
        logits = self.lm_head(hidden)
        if self._has_softcap:
            cap = self._softcap
            logits = torch.tanh(logits / cap) * cap
        # Return updated cache tensors so the runner can thread them.
        return logits, k_caches, v_caches, cumlen_caches


def build_static_cache(model, batch: int, max_seq: int, device, dtype) -> StaticCache:
    """Build a `BufferStaticCache` (nn.Module-backed) so torch.export sees
    the K/V tensors as buffers rather than lifted constants.

    Returns a `StaticCache` for type compatibility (BufferStaticCache subclasses
    it) — interface and behavior match `transformers.cache_utils.StaticCache`.
    """
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    return BufferStaticCache(
        config=cfg,
        max_batch_size=batch,
        max_cache_len=max_seq,
        device=device,
        dtype=dtype,
    )

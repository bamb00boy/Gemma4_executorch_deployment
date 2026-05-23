"""
Gemma 4 E2B — interactive terminal chat.

Self-contained chat REPL that runs the same .pte as pi_runner.py but in
multi-turn mode. KV-cache is reused across turns — only the *new* tokens
in each chat round are fed to the model, so latency stays bounded by the
size of each new user message + the response, not by the cumulative
conversation length (until cache fills).

Dependencies (install once on the deployment host):
    pip install torch==2.11.0 executorch==1.2.0 transformers==5.5.3

Expected files in the same directory (or pass --pte / --tokenizer paths):
    gemma4_e2b_text_int4_extcache.pte
    tokenizer/                          # tokenizer.json + tokenizer_config.json + chat_template.jinja

Usage:
    python gemma4_terminal_chat.py
    python gemma4_terminal_chat.py --max-new-tokens 200
    python gemma4_terminal_chat.py --pte /path/to/.pte --tokenizer /path/to/tokenizer/

Controls:
    Type a message + Enter        → model replies
    Ctrl+C  or  Ctrl+D            → exit
    /reset                         → wipe cache + history, start fresh
    /help                          → show controls
"""

import argparse
import os
import signal
import sys
import time

import torch
from transformers import AutoTokenizer
from executorch.runtime import Runtime, Verification

# -------------------- paths + model layout (must match pi_runner.py) --------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PTE = os.path.join(HERE, "gemma4_e2b_text_int4_extcache.pte")
DEFAULT_TOK = os.path.join(HERE, "tokenizer")

MAX_CACHE_LEN = 512
MASK_LEN = MAX_CACHE_LEN - 1            # 511; .pte specialized to this upper bound
DTYPE = torch.float32

# Same hardcoded layout as pi_runner.py — must match what 04_quantize.py exported.
GEMMA4_E2B_LAYER_SHAPES = [
    # (head_dim, is_sliding)
    (256, True),  (256, True),  (256, True),  (256, True),  (512, False),   # 0..4
    (256, True),  (256, True),  (256, True),  (256, True),  (512, False),   # 5..9
    (256, True),  (256, True),  (256, True),  (256, True),  (512, False),   # 10..14
]
NUM_KV_HEADS = 1
BATCH = 1

# Gemma 4 end-of-turn / EOS tokens — stop generation when we see any of these
EOS_TOKEN_IDS = {106, 1, 2}  # <end_of_turn>, <eos>, <bos>-as-sentinel


# -------------------- ExecuTorch-side helpers --------------------

def allocate_cache_tensors():
    """One set of K, V, cumulative_length tensors per cache layer (zero-filled)."""
    k_caches, v_caches, cumlen_caches = [], [], []
    for head_dim, _is_sliding in GEMMA4_E2B_LAYER_SHAPES:
        shape = (BATCH, NUM_KV_HEADS, MAX_CACHE_LEN, head_dim)
        k_caches.append(torch.zeros(shape, dtype=DTYPE))
        v_caches.append(torch.zeros(shape, dtype=DTYPE))
        cumlen_caches.append(torch.zeros(1, dtype=torch.int64))
    return k_caches, v_caches, cumlen_caches


def step(method, token_id, pos, k_caches, v_caches, cumlen_caches):
    """One forward call: feed `token_id` at position `pos`, return (logits, updated caches)."""
    input_ids = torch.tensor([[token_id]], dtype=torch.long)
    attention_mask = torch.zeros(1, MASK_LEN, dtype=torch.long)
    attention_mask[:, :pos + 1] = 1
    position_ids = torch.tensor([[pos]], dtype=torch.long)
    cache_position = torch.tensor([pos], dtype=torch.long)

    args = (input_ids, attention_mask, position_ids, cache_position,
            *k_caches, *v_caches, *cumlen_caches)
    outputs = method.execute(args)

    # See pi_runner.py for the .pte's 91-output layout. We use indices [46..90].
    n = len(GEMMA4_E2B_LAYER_SHAPES)
    logits = outputs[45]
    base = 46
    k_new = list(outputs[base:base + n])
    v_new = list(outputs[base + n:base + 2 * n])
    cumlen_new = list(outputs[base + 2 * n:base + 3 * n])
    return logits, k_new, v_new, cumlen_new


# -------------------- chat session state --------------------

class ChatSession:
    """Tracks conversation history + the tokens already fed to the .pte cache.

    Key trick: every turn, we re-render the full conversation via the chat
    template, find the longest common prefix with `fed_ids` (what's already
    in the cache), and only feed the new tail. Avoids paying token-by-token
    cost for the same context twice.
    """

    def __init__(self, tokenizer, method, max_new_tokens):
        self.tokenizer = tokenizer
        self.method = method
        self.max_new_tokens = max_new_tokens
        self.history = []           # list of {"role": "user"/"model", "content": "..."}
        self.fed_ids = []           # tokens already in cache
        self.k, self.v, self.cumlen = allocate_cache_tensors()

    def reset(self):
        """Wipe cache and history."""
        self.history = []
        self.fed_ids = []
        self.k, self.v, self.cumlen = allocate_cache_tensors()

    def _render(self):
        """Render the full conversation (with chat template + generation prompt)."""
        messages = [
            {"role": h["role"], "content": [{"type": "text", "text": h["content"]}]}
            for h in self.history
        ]
        enc = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        return enc["input_ids"][0].tolist()

    def _feed(self, token_ids, start_pos):
        """Feed a sequence of tokens to the model, updating cache as we go.
        Returns the logits from the LAST token (for next-token prediction)."""
        last_logits = None
        for i, tok in enumerate(token_ids):
            last_logits, self.k, self.v, self.cumlen = step(
                self.method, tok, start_pos + i,
                self.k, self.v, self.cumlen,
            )
        return last_logits

    def turn(self, user_text):
        """Process one user message → assistant response. Returns (response_text, timing_dict)."""
        self.history.append({"role": "user", "content": user_text})

        # Render full conversation + find what's new
        full_ids = self._render()
        # Verify the existing cache is still a prefix of the new render
        # (chat template should always extend, not modify earlier tokens).
        n_existing = len(self.fed_ids)
        prefix_match = (n_existing <= len(full_ids)
                        and full_ids[:n_existing] == self.fed_ids)
        if not prefix_match:
            # Shouldn't happen with normal chat use, but if it does, reset.
            print("\n  [warn] chat template re-rendered prefix differently; resetting cache.",
                  file=sys.stderr)
            self.k, self.v, self.cumlen = allocate_cache_tensors()
            self.fed_ids = []
            n_existing = 0

        new_tail = full_ids[n_existing:]

        # Cache overflow check
        if n_existing + len(new_tail) + self.max_new_tokens > MASK_LEN:
            tokens_left = MASK_LEN - (n_existing + len(new_tail))
            if tokens_left < 4:
                raise RuntimeError(
                    f"context window full (cache holds {n_existing + len(new_tail)}/{MASK_LEN}). "
                    f"Type /reset to start a new conversation."
                )

        # Feed the new tokens
        t_prefill = time.time()
        last_logits = self._feed(new_tail, start_pos=n_existing)
        self.fed_ids.extend(new_tail)
        t_prefill = time.time() - t_prefill
        n_prefill = len(new_tail)

        # Greedy decode loop
        next_id = int(last_logits[0, -1].argmax())
        generated = []
        t_decode = time.time()
        for step_idx in range(self.max_new_tokens):
            if next_id in EOS_TOKEN_IDS:
                # Don't add the EOS to history's text, but include in fed_ids
                # so cache_position stays aligned.
                self.fed_ids.append(next_id)
                # Feed the EOS so the cache reflects model's own output marker
                _, self.k, self.v, self.cumlen = step(
                    self.method, next_id, len(self.fed_ids) - 1,
                    self.k, self.v, self.cumlen,
                )
                break
            generated.append(next_id)
            pos = len(self.fed_ids)
            last_logits, self.k, self.v, self.cumlen = step(
                self.method, next_id, pos,
                self.k, self.v, self.cumlen,
            )
            self.fed_ids.append(next_id)
            next_id = int(last_logits[0, -1].argmax())
        t_decode = time.time() - t_decode

        response_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        self.history.append({"role": "model", "content": response_text})

        return response_text, {
            "prefill_ms": t_prefill * 1000,
            "prefill_tokens": n_prefill,
            "prefill_tok_s": (n_prefill / t_prefill) if t_prefill > 0 else 0,
            "decode_ms": t_decode * 1000,
            "decode_tokens": len(generated),
            "decode_tok_s": (len(generated) / t_decode) if t_decode > 0 else 0,
            "context_used": len(self.fed_ids),
            "context_max": MASK_LEN,
        }


# -------------------- terminal UI --------------------

HELP_TEXT = """
Commands:
  (just type)   send a message to the model
  /reset        wipe conversation history + cache, start fresh
  /stats        show timing for the last turn
  /help         show this help
  Ctrl+C/D      exit
"""


def main():
    parser = argparse.ArgumentParser(
        description="Interactive terminal chat with Gemma 4 E2B (ExecuTorch .pte runtime)",
    )
    parser.add_argument("--pte", default=DEFAULT_PTE, help="path to .pte (default: alongside this script)")
    parser.add_argument("--tokenizer", default=DEFAULT_TOK, help="path to tokenizer dir")
    parser.add_argument("--max-new-tokens", type=int, default=200,
                        help="max tokens to generate per response (default 200)")
    parser.add_argument("--quiet", action="store_true",
                        help="don't print per-turn timing")
    args = parser.parse_args()

    # Validate files
    for path, label in [(args.pte, ".pte"), (args.tokenizer, "tokenizer dir")]:
        if not os.path.exists(path):
            print(f"error: {label} not found at {path}", file=sys.stderr)
            sys.exit(1)

    print(f"loading tokenizer from {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"loading .pte from {args.pte}...")
    t0 = time.time()
    rt = Runtime.get()
    program = rt.load_program(args.pte, verification=Verification.Minimal)
    method = program.load_method("forward")
    print(f"  loaded in {time.time() - t0:.1f}s")

    session = ChatSession(tokenizer, method, args.max_new_tokens)
    last_stats = None

    print()
    print("=" * 60)
    print("  Gemma 4 E2B — terminal chat")
    print("  /help for commands · Ctrl+C or Ctrl+D to exit")
    print("=" * 60)

    def goodbye(*_args):
        print("\nbye.")
        sys.exit(0)
    signal.signal(signal.SIGINT, goodbye)

    while True:
        try:
            user = input("\nyou> ").strip()
        except EOFError:
            goodbye()

        if not user:
            continue

        # Commands
        if user.startswith("/"):
            cmd = user.lower()
            if cmd == "/help":
                print(HELP_TEXT)
            elif cmd == "/reset":
                session.reset()
                print("  (history + cache reset)")
            elif cmd == "/stats":
                if last_stats is None:
                    print("  (no turn yet)")
                else:
                    s = last_stats
                    print(f"  prefill: {s['prefill_ms']:.0f} ms / {s['prefill_tokens']} tok "
                          f"= {s['prefill_tok_s']:.2f} tok/s")
                    print(f"  decode:  {s['decode_ms']:.0f} ms / {s['decode_tokens']} tok "
                          f"= {s['decode_tok_s']:.2f} tok/s")
                    print(f"  context: {s['context_used']}/{s['context_max']} tokens used")
            else:
                print(f"  unknown command: {user}. type /help for the list.")
            continue

        # Normal turn
        try:
            response, stats = session.turn(user)
        except RuntimeError as e:
            print(f"  [error] {e}")
            continue
        except KeyboardInterrupt:
            goodbye()

        last_stats = stats
        print(f"\nmodel> {response}")
        if not args.quiet:
            print(f"  [{stats['decode_tokens']} tok @ {stats['decode_tok_s']:.2f} tok/s · "
                  f"context {stats['context_used']}/{stats['context_max']}]")


if __name__ == "__main__":
    main()

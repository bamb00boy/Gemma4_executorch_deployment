#!/usr/bin/env bash
# Deploy the inference bundle to the Pi.
#
# Pushes ONLY what's needed to run inference — no codebase, no dev artifacts,
# no .pt2 intermediates. Total transfer: ~12 GB.
#
# What goes over:
#   ~/gemma4/                           on the Pi
#     gemma4_e2b_text_int4_extcache.pte ~5 GB
#     pi_runner.py                      one-shot prompt runner (self-contained)
#     gemma4_terminal_chat.py           interactive multi-turn chat REPL
#     requirements_pi.txt               pip deps
#     tokenizer/
#       tokenizer.json                  ~32 MB
#       tokenizer_config.json
#       chat_template.jinja
#       special_tokens_map.json (if present)
#
# Usage:
#   PI_USER=<your-pi-user> PI_HOST=<your-pi-host>.local scripts/deploy_pi.sh
#   scripts/deploy_pi.sh --user <pi-user> --host <pi-host>.local
#   scripts/deploy_pi.sh --dest /mnt/nvme/gemma4     # custom remote dir
#
# Requires: ssh + rsync already working to the Pi (key-based auth).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# No defaults — must be supplied via env var or CLI flag. Fails loudly if missing.
PI_USER="${PI_USER:-}"
PI_HOST="${PI_HOST:-}"
PI_DEST="${PI_DEST:-}"

# Parse simple flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest) PI_DEST="$2"; shift 2 ;;
        --host) PI_HOST="$2"; shift 2 ;;
        --user) PI_USER="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set/p' "${BASH_SOURCE[0]}" | head -25
            exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

# Hard-fail with a helpful error if user didn't set the target.
if [[ -z "$PI_USER" || -z "$PI_HOST" ]]; then
    echo "error: PI_USER and PI_HOST must be set." >&2
    echo "  PI_USER=<your-pi-user> PI_HOST=<your-pi-host>.local scripts/deploy_pi.sh" >&2
    echo "  or: scripts/deploy_pi.sh --user <pi-user> --host <pi-host>.local" >&2
    exit 1
fi
PI_DEST="${PI_DEST:-/home/${PI_USER}/gemma4}"

PTE="$REPO_ROOT/models/gemma4_e2b_text_int4_extcache.pte"
RUNNER="$REPO_ROOT/runner/pi_runner.py"
CHAT="$REPO_ROOT/runner/gemma4_terminal_chat.py"
SNAP_DIR=$(ls -d "$REPO_ROOT"/cache/huggingface/hub/models--google--gemma-4-e2b-it/snapshots/*/ 2>/dev/null | head -1 || true)

if [[ ! -f "$PTE" ]]; then
    echo "error: $PTE not found. Run scripts/04_quantize.py + scripts/05_lower.py first." >&2
    exit 1
fi
if [[ -z "$SNAP_DIR" ]]; then
    echo "error: no HF snapshot for gemma-4-e2b-it in cache/. Run scripts/01_smoke_test.py to download." >&2
    exit 1
fi

# Build a tiny requirements file for the Pi
REQS=$(mktemp -t gemma4_pi_reqs.XXXXXX)
trap "rm -f $REQS" EXIT
cat > "$REQS" <<'EOF'
# Minimal Python deps to run pi_runner.py on the Pi.
# torch MUST be pinned to the same version used at Phase 5 lowering
# (this Mac). Otherwise XNNPACK on the Pi may reject the .pte:
#   `XNNCompiler::compileModel failed: 0x1` / tensor parameter rejected.
torch==2.11.0
executorch==1.2.0
transformers==5.5.3
EOF

echo "=== Deploy plan ==="
echo "  source repo:   $REPO_ROOT"
echo "  target:        ${PI_USER}@${PI_HOST}:${PI_DEST}"
echo "  .pte:          $(ls -lh "$PTE" | awk '{print $5}') $PTE"
echo "  runner:        $(wc -l <"$RUNNER") lines  $RUNNER"
echo "  tokenizer dir: $SNAP_DIR"
echo "  reqs:          $(wc -l <"$REQS") lines"
echo

echo "=== Ensuring remote dir exists ==="
ssh "${PI_USER}@${PI_HOST}" "mkdir -p ${PI_DEST}/tokenizer"
ssh "${PI_USER}@${PI_HOST}" "df -h ${PI_DEST} | tail -1"
echo

echo "=== Rsyncing .pte (~5 GB; resumable if connection drops) ==="
# --partial --append-verify so a dropped SSH session can resume from
# where it left off instead of restarting the multi-GB transfer.
# -W disables the rsync delta algorithm (pointless for large opaque
# binary diffs); --inplace writes directly to the target file so the
# resume position is preserved.
rsync -avh --progress \
    --partial --append-verify --inplace -W \
    "$PTE" \
    "${PI_USER}@${PI_HOST}:${PI_DEST}/"

echo "=== Rsyncing runner + reqs ==="
rsync -avh "$RUNNER" "$CHAT" "${PI_USER}@${PI_HOST}:${PI_DEST}/"
rsync -avh "$REQS"  "${PI_USER}@${PI_HOST}:${PI_DEST}/requirements_pi.txt"

echo "=== Rsyncing tokenizer files (~32 MB) ==="
# Only the files transformers' AutoTokenizer actually needs. Use --copy-links
# because the snapshot dir contains symlinks into HF's blob store.
rsync -avhL --include='tokenizer.json' \
            --include='tokenizer_config.json' \
            --include='chat_template.jinja' \
            --include='special_tokens_map.json' \
            --exclude='*' \
            "$SNAP_DIR" "${PI_USER}@${PI_HOST}:${PI_DEST}/tokenizer/"

echo
echo "=== Done. Next steps on the Pi: ==="
cat <<EOF

ssh ${PI_USER}@${PI_HOST}
cd ${PI_DEST}

# One-time install (~2 GB of deps, ~5 min on Pi 5):
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_pi.txt

# Run inference (verify mode first to confirm the .pte produces the right
# output for our canonical prompt):
python pi_runner.py --verify

# Then any prompt you like:
python pi_runner.py "Why is the sky blue?" --max-new-tokens 50

# Or open an interactive chat REPL (multi-turn, with KV-cache reuse across turns):
python gemma4_terminal_chat.py
# /help inside chat for commands. Ctrl+C or Ctrl+D to exit.
EOF

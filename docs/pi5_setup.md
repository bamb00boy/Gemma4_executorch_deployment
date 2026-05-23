# Raspberry Pi 5 — Phase 6 setup guide

What the Pi needs **before** we can deploy and run the `.pte`. Aimed at the person physically holding the Pi (i.e. you).

**Deployment-only flow:** the Pi is a pure inference target — no source code, no compilation, no ExecuTorch build from source. We just `pip install executorch` + `transformers` and run a ~250-line Python runner against the pre-built `.pte`. Total bundle pushed to Pi: ~5.2 GB.

Once steps 1–5 are done and the Pi is reachable via SSH, share the credentials (see §6) and I'll take it from there.

---

## 1. Hardware checklist

| Item | Spec | Why |
|---|---|---|
| **Raspberry Pi 5** | Model B, **8 GB RAM** | The `.pte` is 5.14 GB; mmap'd at load + activations + Python deps ≈ 6–7 GB peak. 4 GB Pi 5 will OOM. |
| **Active cooling** | Official Pi 5 Active Cooler, or equivalent fan + heatsink | Sustained LLM inference will pin all 4 Cortex-A76 cores at 100%. Passive cooling thermal-throttles within ~30 s and ruins benchmark numbers. |
| **Power supply** | Official 27 W USB-C PD adapter | Cheap phone chargers under-volt the Pi 5 → it'll show `Throttled: 0x50000` under load and clamp to 1.5 GHz. Throws benchmarks badly. |
| **Storage** | **NVMe via M.2 HAT** (preferred) OR fast microSD (A2 rated, ≥ 32 GB; 64 GB comfortable) | We need ~10 GB free (5.2 GB bundle + ~2 GB Python deps + headroom). microSD is workable but slow — `.pte` load adds ~10–20 s vs NVMe. |
| **Network** | Wired Ethernet (preferred) or 2.4/5 GHz WiFi | Wired is more reliable for the SSH session we'll be running. |
| **For initial setup only** | HDMI cable + monitor + USB keyboard, OR headless setup via Raspberry Pi Imager (no monitor needed) | Headless is fine if you trust the Imager's WiFi configuration |

---

## 2. OS install — pick one

Two reasonable choices. Both ARM64. Use **Raspberry Pi Imager** (https://www.raspberrypi.com/software/) for either.

### Option A (recommended for this project): Ubuntu Server 24.04 LTS

**Why:** It's what most published LLM-on-Cortex-A benchmarks use (geerlingguy / Phoronix / llama.cpp issues), so our Phase 7 numbers will be directly comparable. Newer toolchain (gcc 13.2, cmake 3.28, Python 3.12). Primary ExecuTorch CI target.

**Tradeoff:** Slightly less Pi-specific tooling out of the box — `vcgencmd` needs `apt install libraspberrypi-bin`.

In Imager: **"Other general-purpose OS"** → **"Ubuntu"** → **"Ubuntu Server 24.04 LTS (64-bit)"** (the one **without** "Desktop" in the name).

### Option B: Raspberry Pi OS Lite (Bookworm) 64-bit

**Why:** Smaller install (~700 MB), Pi-Foundation tuning, `vcgencmd` and `raspi-config` pre-installed, friendlier if you're going to poke at the hardware directly.

**Tradeoff:** Slightly older toolchain (gcc 12.2, Python 3.11.2). Less common in published LLM benchmarks.

In Imager: **"Raspberry Pi OS (other)"** → **"Raspberry Pi OS Lite (64-bit)"**.

### Imager advanced options (either OS)

Click the gear icon (⚙️) for advanced options. Set:

- **Hostname:** pick something memorable (e.g., `my-pi`) — this becomes `<hostname>.local` on the LAN.
- **Enable SSH:** ✅ Yes. **Use public-key auth** (paste the contents of `~/.ssh/id_ed25519.pub` from your laptop). Avoid password auth.
- **Username / password:** create a user. Don't use the default (`pi` on Pi OS, `ubuntu` on Ubuntu) — pick anything unique.
- **WiFi:** if going wireless, fill SSID + password and pick your country.
- **Locale / keyboard:** set as needed.

Write the image. Eject. Insert into Pi. Connect power.

First boot takes ~60 s (Pi OS) to ~2 min (Ubuntu Server — runs cloud-init).

Quick test from your laptop:
```bash
ping YOUR_PI_HOSTNAME.local
ssh pi-user@YOUR_PI_HOSTNAME.local
```

If `.local` doesn't resolve on Ubuntu, install `avahi-daemon` (usually already there in 24.04) or just use the IP from your router's DHCP table.

---

## 3. Base software install (run on the Pi)

Once SSH'd in. **No build toolchain needed** — we install ExecuTorch via `pip` (pre-built wheel for ARM64), not from source.

```bash
# Update first — fresh images are usually a few weeks behind
sudo apt update && sudo apt upgrade -y

# Python runtime (Ubuntu 24.04 ships 3.12, Pi OS Bookworm ships 3.11 — both fine)
sudo apt install -y \
    python3-pip python3-venv python3-dev \
    htop tmux

# Benchmarking + monitoring
sudo apt install -y cpufrequtils sysstat ncdu

# Ubuntu Server only — install Pi-specific tools (vcgencmd, etc.)
# Skip this block on Pi OS (already installed).
if ! command -v vcgencmd >/dev/null 2>&1; then
    sudo apt install -y libraspberrypi-bin
fi

# Reboot to pick up kernel updates if any
sudo reboot
```

`sudo apt upgrade` will pull the latest kernel and firmware — important for stable thermal behavior on Pi 5.

Python pip deps (`executorch` ~600 MB, `transformers` + `torch` ~1.5 GB) install later via `requirements_pi.txt`, in a venv on the Pi. Total Python install ~2 GB.

---

## 4. Performance config (for benchmark sanity)

```bash
# Set CPU governor to performance (don't let the kernel down-clock under load)
sudo cpufreq-set -g performance

# Persist across reboots
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils

# Disable WiFi power management during benchmarks (if on WiFi)
sudo iwconfig wlan0 power off 2>/dev/null || true

# Verify
vcgencmd get_throttled    # should print "throttled=0x0"; anything else means power/thermal issue
cpufreq-info | grep 'current CPU frequency'   # should show 2.4 GHz on Pi 5
```

**Don't disable swap.** ExecuTorch's first-load + KleidiAI weight repack will briefly spike memory — better to swap a little than OOM-kill the runner.

If you want clean benchmark numbers, you can stop background services for the duration:
```bash
sudo systemctl stop bluetooth avahi-daemon
# (re-enable later: sudo systemctl start bluetooth avahi-daemon)
```

---

## 5. Disk space check

```bash
df -h /
```

Need at least **10 GB free**. Breakdown:
- `.pte` model: 5.14 GB
- Tokenizer + runner + reqs: ~35 MB
- Python venv (executorch + transformers + torch + numpy + …): ~2 GB
- pip download cache + apt cache: ~1 GB transient
- Working space + headroom: ~2 GB

A 32 GB microSD with the default install leaves approximately 25 GB free — sufficient. A 64 GB+ NVMe provides ample headroom.

---

## 6. What I'll need from you to connect

Once steps 1–5 are done, share **all of these** in one message:

| What | Example | Notes |
|---|---|---|
| **Host** | `YOUR_PI_HOSTNAME.local` or `192.168.1.42` | mDNS hostname or fixed LAN IP. If the Pi is on a different network than your laptop, you'll need to set up port forwarding or a reverse SSH tunnel. |
| **Port** | `22` | Default; if you changed it in `/etc/ssh/sshd_config` say so. |
| **Username** | `pi-user` | Whatever you set in step 2. |
| **Auth** | Public key (preferred): set up so `ssh user@host` from this Mac works passwordless. Or password (last resort). | If using key auth: `ssh-copy-id user@host` from this Mac to install the Mac's public key into `~/.ssh/authorized_keys` on the Pi. |
| **sudo** | Yes / no | Needed only if there's a missing apt package; the deploy + run flow itself doesn't need sudo (pip into a user venv). |
| **Reverse-tunnel / VPN?** | If the Pi isn't on the same network as my laptop, we need one of: (a) port-forward 22 on your router (insecure unless behind fail2ban), (b) Tailscale / Cloudflare Tunnel (recommended), (c) reverse SSH tunnel to a jump host. | Most home networks work fine with (b). |

**Security note:** treat any SSH access you grant as "anyone with the credentials can read every file the user can read and run commands as them". Don't share creds for a Pi that has anything sensitive on it. The Pi for this project should be a clean install used only for this work.

---

## 7. Quick "is the Pi ready?" sanity check

Run this on the Pi before you message me — pastes a one-line summary:

```bash
echo "$(uname -m) | $(lsb_release -ds) | python=$(python3 --version) | gcc=$(gcc --version | head -1) | cmake=$(cmake --version | head -1) | free $(df -h / | awk 'NR==2 {print $4}')"
```

Expected output looks like:
```
aarch64 | Debian GNU/Linux 12 (bookworm) | python=Python 3.11.2 | gcc=gcc (Debian 12.2.0-14) 12.2.0 | cmake=cmake version 3.25.1 | free 55G
```

If `aarch64` (NOT `armv7l`) and Python ≥ 3.11 and free ≥ 10G — you're good to go.

---

## 8. What Phase 6 actually does (deployment-only)

The Pi is a pure inference target — no repo, no model dev, no compilation. From the Mac side, `scripts/deploy_pi.sh` rsyncs just what's needed:

```
~/gemma4/                              on the Pi
  gemma4_e2b_text_int4_extcache.pte    5.14 GB — the single lowered INT4 program
  pi_runner.py                         ~9 KB — self-contained runner
  requirements_pi.txt                  ~150 B — pip deps
  tokenizer/                           ~32 MB
    tokenizer.json
    tokenizer_config.json
    chat_template.jinja
    special_tokens_map.json (if present)
```

**Total transfer: ~5.2 GB.** One `.pte`, not two — Phase 6 cache externalization made prefill and decode shareable in one program; the runner threads cache tensors across calls in Python. Full design rationale in `RESULTS.md` Phase 6 section.

Then on the Pi:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements_pi.txt    # executorch + transformers + torch, ~5 min
python pi_runner.py --verify          # confirms .pte produces "The capital of France is **Paris**."
python pi_runner.py "your prompt"
```

### What's inside `pi_runner.py`

Fully self-contained — no imports from this repo, just `torch`, `transformers`, `executorch`. ~250 lines. It:
1. Loads the tokenizer (apply_chat_template for the Gemma chat format)
2. Loads the `.pte` via `executorch.runtime.Runtime`
3. **Allocates KV cache tensors externally** (~19 MB across 15 cache layers — 12 sliding @ head_dim=256, 3 full @ head_dim=512)
4. **Token-by-token forward** for both prompt feed AND decode (same `.pte` for both phases — no batched prefill in this design)
5. Threads cache tensors across all 23+ forward calls
6. Stops at EOS (`<end_of_turn>` token, id 106) or `--max-new-tokens`

### Measured performance

Identical 14-token chat-formatted prompt + decode until EOS (9 tokens). Both columns produce the same text bit-for-bit. Dev box is a MacBook Pro 14" with **Apple M1 Pro (8 cores: 6 performance + 2 efficiency), 16 GB unified memory, macOS 26.3.1**.

| Stage | Mac (M1 Pro, this repo) | Pi 5 (8 GB, A76, this repo) |
|---|---|---|
| `.pte` load | 4.0 s | 5.1 s (microSD) |
| Prompt feed (token-by-token) | 1946 ms for 14 tok → **7.20 tok/s** | 18.2 s for 14 tok → **0.77 tok/s** |
| Decode | 1039 ms for 9 tok → **8.66 tok/s** | 10.4 s for 9 tok → **0.87 tok/s** |
| Total (23 tok) | **2.99 s** | **28.6 s** |
| Quality vs FP32 reference | 9/9 token match | 9/9 token match |

The ~10× Mac→Pi gap exceeds the underlying CPU-generation difference (M1 Pro vs Cortex-A76 should be ~4–6×). The extra cost is the `XnnpackPartitioner(per_op_mode=True)` workaround for the ARM XNNPACK rejection (see `KNOWN_ISSUES.md` #1) — it neutralizes kernel fusion that XNNPACK on Mac DOES apply.

### Comparison target for Phase 7

`llama.cpp` running Gemma 4 E2B (GGUF) on the same Pi 5 hardware. Real benchmark from [potato-os/core](https://github.com/potato-os/core/blob/main/docs/benchmarks/gemma4-pi-benchmark-2026-04-04.md) (April 4 2026): **6.71 tok/s** decode on Pi 5 16 GB. That's the target.

If the ExecuTorch + KleidiAI decode rate falls in the same range or higher, the project achieves its primary goal. If it falls meaningfully below, the result is reported as such — still a publishable first-Gemma-4-on-Pi ExecuTorch deployment, qualified by the observation that ExecuTorch currently trails hand-tuned ggml on this hardware.

---

## 9. Architecture decisions worth knowing (for context)

The following are the non-obvious decisions embedded in the shipped artifact. Full reasoning is in `RESULTS.md`.

| Decision | Why |
|---|---|
| **XNNPACK, not ARM backend** | ExecuTorch 1.2's `executorch.backends.arm` is Ethos-U **NPU**-only. Pi 5 is Cortex-A *CPU*. XNNPACK on ARM auto-dispatches to KleidiAI INT4/INT8 kernels — same fast-path, reached through XNNPACK. |
| **External KV cache** | Two-`.pte` (prefill + decode) couldn't share buffer state in ExecuTorch. Externalizing the cache → one stateless `.pte`, runner threads cache across calls. |
| **Token-by-token "slow prefill"** | Same `.pte` does both phases (single program, single graph). Trade: TTFT scales linearly with prompt length, but no second program + no shape buckets. |
| **Hand-rolled `Int8Embedding`** | torchao's quantized embedding paths lower to `torchao::quantize_affine` etc., which have no ExecuTorch out-variants. Our `_int8_embedding.py` uses only standard aten ops (`index_select`, cast, mul) — all have out-variants, all lower through XNNPACK. **Big win:** `.pte` dropped from 12.18 GB → 5.14 GB. |
| **FP32 baseline (not BF16)** | BF16 + INT4 Linears + Int8Embedding fails at `to_executorch()` with `torchao::quantize_affine` missing out-variants — appears specific to BF16 dynamic activation quant. FP32 works and fits in Pi 5 RAM, so we shipped FP32. BF16 chase remains parked for later (would save ~250 MB more). |
| **INT4 stored as INT8 bytes** | torchao 0.17's CPU INT4 path stores each 4-bit weight in one INT8 byte (4 useful bits + 4 zero bits). No real disk savings vs INT8. KleidiAI's load-time repack on ARM should convert to true 4-bit tile layout for matmul, but Mac sees the unpacked form. |

Phase 7 will be the llama.cpp baseline on the same hardware for comparison.

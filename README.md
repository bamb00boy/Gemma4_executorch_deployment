# Gemma 4 E2B → ExecuTorch → Raspberry Pi 5

An end-to-end deployment of Google's **Gemma 4 E2B** (5.5 B parameters, multimodal) on a **Raspberry Pi 5 (8 GB)** through PyTorch's **ExecuTorch 1.2.0** runtime with the **XNNPACK** backend. This is the first publicly documented Gemma 4 deployment via the **ExecuTorch** stack; Gemma 4 has previously been deployed on Pi 5 via `llama.cpp` / GGUF (see [Performance](#performance) for the comparison). The pipeline is documented at a level intended to be reproducible by anyone with the listed hardware.

## Status

The pipeline runs end-to-end: the `.pte` loads, executes, and produces bit-exact text matching the FP32 reference on Pi 5 hardware. Two qualifications apply:

- **Decode throughput is approximately 7.7× lower than `llama.cpp`** on the same hardware. This repo measures **0.72–0.87 tok/s** decode across sessions; the published [potato-os/core Pi 5 benchmark](https://github.com/potato-os/core/blob/main/docs/benchmarks/gemma4-pi-benchmark-2026-04-04.md) (Gemma 4 E2B, llama_cpp + GGUF, Pi 5 16 GB, April 4 2026) measures **6.71 tok/s** decode. The shipped `.pte` uses `XnnpackPartitioner(per_op_mode=True)` to work around an ARM XNNPACK rejection bug in ExecuTorch 1.2.0 — initially believed to be the root cause of the entire gap. **Subsequent experiments (PT2E, `config_precisions=DYNAMIC_QUANT`, ExecuTorch nightly 1.4.0.dev with default fused partitioner) show the partitioner mode is NOT the bottleneck:** even after recovering 508 fused subgraphs on ARM the decode rate does not improve. Full diagnosis, measured three-way Pi benchmark, and updated remaining-candidate analysis in [KNOWN_ISSUES.md #1](KNOWN_ISSUES.md).
- If maximum decode throughput on Pi 5 is the only requirement, `llama.cpp` is the appropriate tool. This project provides a reproducible end-to-end pipeline and a catalog of documented issues for ExecuTorch's official deployment path on a non-trivial LLM.

## What this is

| | |
|---|---|
| **Model** | `google/gemma-4-e2b-it` (5.5 B params, 35 decoder layers, GQA 8:1, sliding_window=512) |
| **Quantization** | INT4 weight (torchao `Int8DynamicActivationIntxWeightConfig`) on `nn.Linear`; INT8 per-row custom `Int8Embedding` on the 2.35 B `embed_tokens_per_layer` |
| **Runtime** | ExecuTorch 1.2.0 with XNNPACK partitioner (`per_op_mode=True`) |
| **Target** | Raspberry Pi 5 (Cortex-A76, 8 GB RAM, aarch64 Ubuntu Server 24.04) |
| **Output `.pte` size** | 5.14 GB (fits in Pi 5 RAM with ~3 GB headroom) |
| **Quality** | bit-exact 9/9 token match vs FP32 reference on the canonical prompt |
| **Decode speed** | see [Performance](#performance) section below |
| **Baseline FP32** | recommended (BF16 lowering fails — torchao's `quantize_affine` op has no ExecuTorch out-variants when chained with INT4 dynamic-act on BF16) |

### Performance

Measured on identical 14-prompt + 9-decode tokens for `"The capital of France is"`, bit-exact output across all rows.

| Host | Role | Prompt feed | Decode | Total wall |
|---|---|---|---|---|
| **Raspberry Pi 5** — 8 GB RAM, Cortex-A76 @ 2.4 GHz, Ubuntu Server 24.04 LTS, kernel 6.8, microSD storage | deployment target | 0.77 tok/s (18.2 s for 14 tok) | **0.87 tok/s** | 28.6 s |
| **MacBook Pro 14"** — Apple M1 Pro (8 cores: 6P + 2E), 16 GB unified memory, macOS 26.3.1 | development / iteration reference | 7.20 tok/s (1946 ms for 14 tok) | **8.66 tok/s** (1039 ms for 9 tok) | 2.99 s |
| [potato-os/core llama.cpp Gemma 4 E2B on Pi 5](https://github.com/potato-os/core/blob/main/docs/benchmarks/gemma4-pi-benchmark-2026-04-04.md) (Pi 5 16 GB, SD card, April 4 2026) | external reference (different runtime) | n/a | **6.71 tok/s** | n/a |

`.pte` load time: 4.0 s on Mac (NVMe), 5.1 s on Pi 5 (microSD). Cache footprint at runtime: 18.9 MB across 15 layers (12 sliding @ head_dim=256, 3 full @ head_dim=512).

**Notes on the numbers:**

- The Mac CPU decode rate and the published `llama.cpp` Pi 5 rate fall in the same range (~7–9 tok/s decode), indicating that the `.pte` itself runs at competitive speed when the underlying matmul fast-path is reached.
- The Pi 5 result in this repo is approximately 10× slower than the Mac measurement and approximately 7.7× slower than `llama.cpp` on identical Pi 5 hardware. **The partitioner mode is not the bottleneck** — a three-way Pi 5 benchmark across `per_op_mode=True` (49 unfused subgraphs), `config_precisions=DYNAMIC_QUANT` (211 fused), and ExecuTorch nightly's default partitioner (508 fused) produced 0.72 / 0.70 / 0.64 tok/s decode respectively, all within sampling noise. The ARM XNNPACK rejection (workaround #14 below) is real and is fixed upstream in nightly, but recovering fused subgraphs alone does not close the gap. Full diagnosis and remaining-candidate analysis in [KNOWN_ISSUES.md #1](KNOWN_ISSUES.md).
- Mac numbers are listed as a **development reference**, not a competitive benchmark. The M1 Pro and Cortex-A76 represent different CPU generations, ISAs, and process nodes, and Mac is not a deployment target. The figure is included so that those iterating on the scripts on a Mac have a local expectation.

## What this is not

- Not the fastest LLM runtime on Pi 5 — see [Performance](#performance) for the measured comparison.
- Not a turn-key inference server. The runners process one prompt at a time (one-shot) or one chat session (REPL). No batching, no streaming, no web UI.
- Not a port of Gemma 4 to GGUF (`llama.cpp` will likely add Gemma 4 support natively at some point; this repo does not compete with that).
- Not a tutorial on ExecuTorch in general — this is a *specific deployment pipeline* and a *catalog of issues encountered on a non-trivial model*.

## Who would actually use this

| Audience | Why this matters to them |
|---|---|
| Someone deploying a *custom PyTorch model* (not in any model zoo) on Pi-class hardware | This is a worked example of the toolchain — export → quantize → lower → runtime — applied to a non-trivial model. Substitute your model for Gemma 4 and most of the pipeline transfers. |
| Someone wanting an **ExecuTorch CPU/ARM** Gemma deployment | The [official ExecuTorch Gemma 3 example](https://github.com/pytorch/executorch/tree/main/examples/models/gemma3) is CUDA-focused (uses `tile_packed_to_4d` packing, recommends NVIDIA GPUs). This repo fills the CPU + ARM gap for a newer model (Gemma 4) with the issues documented. |
| ExecuTorch contributors / maintainers | [KNOWN_ISSUES.md](KNOWN_ISSUES.md) is a catalog of upstream bugs identified during this work — each with a minimal repro path. |
| Anyone evaluating whether ExecuTorch is production-ready for a new LLM | A concrete-evidence answer: *almost — the specific gaps are documented*. |
| Anyone retrying this with a newer ExecuTorch / torchao | The pipeline + scripts are in place; `pip install --upgrade executorch`, re-lower, and (if the ARM XNNPACK gap has been fixed upstream) the fast path may light up. |

## Repository layout

```
.
├── README.md                  # this file
├── LICENSE                    # MIT for the code in this repo
├── NOTICE-GEMMA.md            # Gemma 4 attribution + Apache 2.0 + Prohibited Use Policy notice
├── KNOWN_ISSUES.md            # upstream bug catalog + workarounds applied
├── RESULTS.md                 # engineering log: every bug, every fix, every benchmark
├── requirements.txt           # pip-pinned versions that work (Mac dev side)
├── environment.yml            # conda env export (alternative)
├── .gitignore
├── docs/
│   ├── architecture.md        # Gemma 4 from an exporter's perspective (config, hazards, what mattered)
│   └── pi5_setup.md           # Pi 5 prep guide (OS, deps, SSH, deployment flow)
├── scripts/
│   ├── _paths.py              # repo-local HF/torch cache routing (sets HF_HOME etc.)
│   ├── _wrapper.py            # TextOnlyWrapper + TextWrapperExternal (export-friendly)
│   ├── _buffer_cache.py       # nn.Module shim for transformers' StaticCache (Phase 5 fix)
│   ├── _external_cache.py     # TransientCache: cache as graph inputs, not module state (Phase 6 fix)
│   ├── _int8_embedding.py     # custom INT8 Embedding (Phase 6 size reduction)
│   ├── run.sh                 # activates conda env + sources .env + runs a script
│   ├── 01_smoke_test.py       # Phase 2: load model, verify text + vision generation
│   ├── 02_inspect.py          # Phase 3: architecture introspection
│   ├── 03_export.py           # Phase 3: torch.export of the text path (FP32, internal cache)
│   ├── 04_quantize.py         # Phase 4 + 6: quantize + re-export (external cache + Int8Embedding)
│   ├── 05_lower.py            # Phase 5: lower .pt2 → .pte via XNNPACK (per_op_mode workaround)
│   ├── 06_verify.py           # Phase 3: FP32 exported vs eager numerical equivalence
│   ├── 07_decode_check.py     # Phase 3: wrapper prefill+decode loop vs model.generate
│   ├── 08_verify_int4.py      # Phase 4: INT4 exported vs INT4 eager numerical equivalence
│   ├── 09_verify_pte.py       # Phase 5: .pte loads + runs in ExecuTorch runtime
│   ├── 10_decode_only_check.py    # Phase 6 diagnosis: two-.pte cache state problem
│   ├── 11_external_cache_check.py # Phase 6 sanity: external-cache wrapper eager
│   └── deploy_pi.sh           # rsync minimum bundle to a remote Pi over SSH
├── runner/
│   ├── pi_runner.py           # one-shot runner: tokenize prompt → generate → exit (250 lines)
│   └── gemma4_terminal_chat.py # interactive multi-turn chat REPL with KV-cache reuse (~325 lines)
└── results/                   # per-phase logs from the runs that produced this deployment
    ├── 01_smoke_test.log
    ├── 02_inspect.txt
    ├── 03_export.log
    ├── 04_quantize.log
    ├── 05_lower.log
    └── ...                    # full chronology
```

## Quick start

### To run the existing `.pte` on a Pi 5

The `.pte` (5.14 GB) is too large for GitHub and is hosted on HuggingFace:

> **HuggingFace model repo:** [`bamb00boy/gemma4-e2b-int4-executorch-pi5`](https://huggingface.co/bamb00boy/gemma4-e2b-int4-executorch-pi5)
> *(If the link 404s, the `.pte` upload is still in progress — rebuild locally with the pipeline in [§ To re-build the `.pte` yourself](#to-re-build-the-pte-yourself-on-a-mac-or-other-dev-machine) in the meantime.)*

The HuggingFace repo contains the `.pte`, the tokenizer files, and a copy of `pi_runner.py` and `gemma4_terminal_chat.py`. Total download is approximately 5.2 GB.

On the Pi:

```bash
# 1. Install the HuggingFace CLI (one-time). The `hf` command ships with
#    huggingface_hub >= 0.30 and is the current canonical CLI; the older
#    `huggingface-cli` name is still aliased for backward compatibility.
pip install --user --upgrade huggingface_hub

# 2. Download the model bundle into ~/gemma4
hf download bamb00boy/gemma4-e2b-int4-executorch-pi5 \
    --local-dir ~/gemma4

# 3. Set up the runtime environment
cd ~/gemma4
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.11.0 executorch==1.2.0 transformers==5.5.3

# 4. Verify
python pi_runner.py --verify
# Expected: "RESULT: PASS" — "The capital of France is **Paris**."

# 5. Use
python pi_runner.py "Your prompt here" --max-new-tokens 50

# Or open an interactive chat (multi-turn, KV-cache reused across turns):
python gemma4_terminal_chat.py
# Type a message + Enter. /help for commands. Ctrl+C or Ctrl+D to exit.
```

The Pi prep guide ([docs/pi5_setup.md](docs/pi5_setup.md)) covers OS install (Ubuntu Server 24.04 LTS recommended), required apt packages, performance tuning (CPU governor, cooling), and SSH setup.

### To re-build the `.pte` yourself (on a Mac or other dev machine)

You'll need: HuggingFace access to `google/gemma-4-e2b-it` (gated — accept terms first), Python 3.12, ~22 GB of free disk, ~16 GB of free RAM.

```bash
# Setup
brew install miniforge   # or use your system conda
conda create -n gemma4 python=3.12 -y
conda activate gemma4
pip install executorch==1.2.0 transformers==5.5.3 torchvision

# Put your HF token in <repo>/.env (chmod 600)
echo "HF_TOKEN=hf_xxx" > .env && chmod 600 .env

# Run the pipeline (each script writes a log into results/)
scripts/run.sh scripts/01_smoke_test.py        # ~30 s on MPS; first run downloads 10 GB
scripts/run.sh scripts/02_inspect.py           # architecture inspection
scripts/run.sh scripts/04_quantize.py          # Phase 4 + 6: quantize + export single .pt2 (~15 min)
scripts/run.sh scripts/05_lower.py             # Phase 5: lower to .pte (~15 min)
scripts/run.sh scripts/09_verify_pte.py        # confirm the .pte produces correct output

# Deploy to a remote Pi
scripts/deploy_pi.sh    # rsyncs models/ + tokenizer + runner to PI_USER@PI_HOST
```

See [docs/pi5_setup.md](docs/pi5_setup.md) for full Pi prep (hardware, OS, SSH, performance tuning).

## Issues encountered

The following issues were encountered and documented during this work (full chronology and fixes in [RESULTS.md](RESULTS.md)):

| # | Issue | Where | Workaround applied |
|---|---|---|---|
| 1 | `torch_dtype` deprecated in transformers 5.x | Phase 2 | Use `dtype=` |
| 2 | `torchvision` is a hidden Gemma 4 dep | Phase 2 | `pip install torchvision` |
| 3 | `torch.export.save` truncates `.pt2` if given a `pathlib.Path` (>2 GB) | Phase 3 | Pass an open file handle |
| 4 | Decode `attention_mask` must be `[1, cache_position+1]`, not `[1, 1]` | Phase 3 | Document correct shape |
| 5 | `torchao.int4_weight_only` deleted; new `Int4WeightOnlyConfig` is CUDA-only | Phase 4 | Use `Int8DynamicActivationIntxWeightConfig(weight_dtype=int4)` |
| 6 | `StaticCache` is not an `nn.Module` → K/V lifted as constants → fails `run_decompositions` | Phase 5 | Subclass to add `nn.Module` + `register_buffer` (see `scripts/_buffer_cache.py`) |
| 7 | Per-layer-type `head_dim` (256 sliding, 512 full) in Gemma 4 | Phase 5 | Mirror the branch in our buffer cache |
| 8 | `to_executorch()` fires autograd leaf-with-grad check | Phase 5 | Wrap in `torch.inference_mode()` |
| 9 | `.pte` shape-specializes to upper bound of dynamic `Dim` (no actual runtime dynamism) | Phase 5 | Pad runtime inputs to seq=511 |
| 10 | Two `.pte`s (prefill + decode) can't share cache state | Phase 6 | Externalize cache as graph inputs/outputs (see `scripts/_external_cache.py`) |
| 11 | `Int8WeightOnlyConfig` for embeddings has no ExecuTorch out-variants | Phase 6 | Custom `Int8Embedding` using standard aten ops (see `scripts/_int8_embedding.py`) |
| 12 | BF16 + INT4 Linears + INT8 embed → `torchao::quantize_affine` missing out-variants at `to_executorch()` | Phase 6 | Use FP32 baseline (~250 MB regression vs BF16, acceptable) |
| 13 | torch version mismatch between Mac (lowering) and Pi (runtime) silently breaks XNNPACK | Phase 6 | Pin `torch==2.11.0` in `requirements_pi.txt` |
| 14 | ARM XNNPACK in `executorch==1.2.0` rejects fused INT4 subgraphs (`xnn_status_invalid_parameter` on tensor 6) | Phase 6 | `XnnpackPartitioner(per_op_mode=True)` — costs us the fast path but unblocks deployment |

Each of these is a real upstream bug-report-grade finding. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for which ones should be filed against ExecuTorch / torchao.

## Design FAQ

### Why a Python runner on the Pi and not C++?

A C++ runner is commonly assumed to be substantially faster than Python. For this workload, that assumption does not hold.

The time breakdown of a Pi inference run (approximately 28 s total for a 14-token prompt and 9-token generation) is:

| | Time | Fraction |
|---|---|---|
| Matmul / attention compute (in C++ XNNPACK kernels) | ~27.9 s | **99.5%** |
| Python interpreter overhead per `method.execute()` call | ~200 µs × 23 calls ≈ 5 ms | 0.02% |
| Tensor construction / wrapping per call | ~1–2 ms × 23 ≈ 30 ms | 0.1% |
| Tokenizer | ~50 ms (one-shot) | 0.2% |
| **Python orchestration total** | **~85 ms** | **0.3%** |

ExecuTorch is implemented in C++; XNNPACK is C++; KleidiAI is C. Python is the orchestration layer that invokes them. The ~10× gap to `llama.cpp` is not attributable to Python overhead — it is caused by the XNNPACK ARM bug ([KNOWN_ISSUES.md #1](KNOWN_ISSUES.md)) that required `per_op_mode=True` and disabled the fused-subgraph fast path. A C++ runner would not change this.

**What a C++ runner would provide (for this workload):**

| Benefit | Magnitude |
|---|---|
| Startup time | ~5–8 s saved (Python import eliminated) |
| Memory baseline | ~400 MB saved (Python + torch + transformers no longer resident) |
| Deploy bundle size | ~1.5 GB smaller (no `.venv/`) |
| Per-call inference speedup | ~5–10% (within run-to-run noise) |
| Integration ergonomics | A binary artifact, not a Python venv |

**What a C++ runner would cost:**

- Approximately 500–1000 lines of C++ (load `.pte`, manage 49 inputs / 91 outputs per call, run tokenizer, decode loop, sampling)
- A C++ tokenizer (SentencePiece, the HF `tokenizers` Rust crate, or a hand-port)
- Building ExecuTorch from source for static linking (~30–60 min on Pi 5)
- Maintenance overhead of two runners

**When C++ would be appropriate:**

- Once the underlying compute path has been restored (XNNPACK fused subgraphs accepted on ARM, ~5–10× speedup recovered), a C++ runner becomes worthwhile polish — startup time and memory baseline become more significant when inference itself is fast.
- For deployment as a standalone binary or systemd service (no Python environment management).
- For embedded targets without Python available.

The Python runner is shipped here because it is a ~250-line self-contained script that can be read, modified, or substituted with a different model. For an artifact at the current speed bottleneck, this is the appropriate tradeoff. A C++ runner is a reasonable next step once the compute path is faster.

## License

Two distinct licenses are in play:

| Component | License |
|---|---|
| Code in this repository (scripts, runners, documentation) | **MIT** — see [LICENSE](LICENSE) |
| Gemma 4 weights and derivatives (the `.pte`) | **Apache 2.0** + Gemma Prohibited Use Policy — see [NOTICE-GEMMA.md](NOTICE-GEMMA.md) for the notice and links to Google's authoritative documents |

If you redistribute the `.pte` (e.g., via HuggingFace):

1. Include the Apache 2.0 license text alongside the file (standard Apache requirement).
2. Retain copyright notices and clearly mark modifications (Apache §4).
3. Pass through the Gemma Prohibited Use Policy to downstream users (link to it in your model card).
4. You **do NOT** need to gate the download or require recipient agreement — Apache 2.0 does not impose either. (Gating is optional if you want to track usage.)

For the full license text and authoritative sources (Apache 2.0 page, Prohibited Use Policy, model card), see [NOTICE-GEMMA.md](NOTICE-GEMMA.md).

## Acknowledgments

- Google for releasing Gemma 4 with open weights.
- The PyTorch + ExecuTorch teams for the on-device runtime + `torch.export` stack.
- The torchao maintainers for the quantization primitives.
- The Raspberry Pi Foundation for the affordable ARM hardware on which this work was conducted.


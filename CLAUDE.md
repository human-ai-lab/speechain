# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpeeChain is a PyTorch-based machine speech chain toolkit for ASR and TTS, developed at NAIST's AHC lab. It supports joint ASR+TTS (machine speech chain) pipelines, multi-GPU training via DDP, and configuration-driven experiment management.

## Environment Setup

```bash
uv pip install -e .
```

Two environment variables must be set (typically via `source envir_preparation.sh`):
- `SPEECHAIN_ROOT` - absolute path to the repository root
- `SPEECHAIN_PYTHON` - path to the Python interpreter in the speechain conda env

## Linting (run before every commit — enforced by CI)

```bash
black speechain/                                    # format
ruff check --select I speechain/ --fix              # fix import sorting
```

CI checks: `black --check speechain/` and `ruff check --select I speechain/`.

## Testing

```bash
python -m pytest speechain/utilbox/test_humanfriendly.py -v
python -m pytest speechain/module/encoder/test_speaker.py -v
python -m pytest speechain/module/vocoder/test_hifigan.py -v
```

Test files are named `test_*.py` co-located with the source they test.

## Running Experiments

All experiments are launched via `recipes/run.sh`:

```bash
bash recipes/run.sh \
  --task asr \
  --dataset librispeech \
  --subset train-clean-100 \
  --exp_cfg 100-bpe5k_transformer-large_lr2e-3.yaml \
  --ngpu 2 \
  --train true \
  --test true
```

Tasks: `asr`, `tts`, `lm`, `offline_tts2asr`. Experiment configs live in `recipes/{task}/{dataset}/{subset}/exp_cfg/`.

Useful flags: `--resume true`, `--dry_run true`, `--accum_grad 4`, `--ft_factor 0.1`, `--test_model 10_valid_accuracy_average`.

Results are saved under `recipes/{task}/{dataset}/{subset}/exp/{exp_name}/`.

## Architecture

The toolkit uses a **configuration-driven, composable** design. All components are instantiated from YAML configs.

```
Runner (speechain/runner.py)          ← main entry point
  ├── Model (speechain/model/)        ← ar_asr.py, nar_tts.py, lm.py
  │     ├── Module (speechain/module/) ← encoder/, decoder/, frontend/, vocoder/, etc.
  │     └── Criterion (speechain/criterion/) ← loss functions, metrics
  ├── Iterator (speechain/iterator/)  ← batching by sequence length
  │     └── Dataset (speechain/dataset/)
  ├── OptimScheduler (speechain/optim_sche/) ← Noam, exponential decay
  └── Monitor (speechain/monitor.py)  ← TensorBoard, checkpointing, reports
```

**Key abstractions:**
- `speechain/model/abs.py` — abstract `Model` base class; all models inherit from it
- `speechain/module/abs.py` — abstract `Module` base class; all neural components inherit from it
- `speechain/runner.py` — orchestrates training/testing; call with `Runner.run()`

**Models:**
- `ar_asr.py` — attention-based autoregressive ASR (Transformer/Conformer encoder + decoder)
- `nar_tts.py` — non-autoregressive FastSpeech2 TTS
- `lm.py` — language model (used for LM fusion in ASR)

**Module subdirectories:** `frontend/` (feature extraction), `encoder/`, `decoder/`, `prenet/`, `postnet/`, `vocoder/` (HiFi-GAN), `transformer/`, `conformer/`, `norm/`, `augment/`, `standalone/`

**Tokenizers** (`speechain/tokenizer/`): character, SentencePiece/BPE, G2P phoneme.

**Inference functions** (`speechain/infer_func/`): beam search for ASR, autoregressive decoding for TTS.

## Code Style

- Formatter: Black (default settings)
- Import sorting: Ruff (`--select I`)
- Docstrings: Google-style
- Classes: CamelCase; functions/variables: snake_case
- New modules go in the appropriate `speechain/` subdirectory, following the `abs.py` abstract-class pattern

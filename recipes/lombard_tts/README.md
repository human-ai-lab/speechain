# Lombard TTS: a machine speech chain for noise-adaptive synthesis

Implementation of:

> S. Novitasari, S. Sakti, S. Nakamura, "A Machine Speech Chain Approach for
> Dynamically Adaptive Lombard TTS in Static and Dynamic Noise Environments,"
> *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2022.

A FastSpeech2 backbone is fine-tuned with a closed feedback loop: a frozen ASR
"listener" hears the synthesized speech mixed with noise, and an SNR predictor
+ ASR-loss embedding is fed back into the decoder so the model can adapt its
next attempt. At inference time this repeats for up to `max_loops` rounds,
picking the best-scoring one.

## 0. Prerequisites

- A pretrained (non-Lombard) FastSpeech2 checkpoint to fine-tune from, e.g.
  `recipes/tts/ljspeech/exp/22.05khz_mfa_fastspeech2` (see `recipes/tts/README.md`).
- A pretrained ASR checkpoint to act as the frozen listener, e.g.
  `recipes/asr/librispeech/train-clean-100/exp/100-bpe5k_conformer-small_lr2e-3`
  (see `recipes/asr/README.md`).
- [SoX](http://sox.sourceforge.net/) on `PATH` (used by the data-prep script
  below to synthesize the Lombard-effect vocal changes).
- A GPU with enough memory for FastSpeech2 + HiFi-GAN + the ASR listener
  loaded simultaneously (~26 GB was observed for the default config on a
  single GPU).

**Environment gotcha:** if `torch` and `torchvision` were installed from
different index URLs (e.g. plain PyPI vs. the PyTorch CUDA wheel index), their
compiled custom ops can silently mismatch and `import torchvision` (used by
`speechain/snapshooter.py` for TensorBoard image logging) raises
`RuntimeError: operator torchvision::nms does not exist` at the very first
`runner.py` import — this looks unrelated to Lombard TTS but breaks training
entirely. Reinstall a matching build, e.g.:

```bash
uv pip install --python .venv/bin/python "torchvision==<version matching your torch build>" \
  --reinstall --index-url https://download.pytorch.org/whl/cu128   # match your CUDA tag
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

## 1. Data preparation

The paper's Lombard effect is simulated rather than recorded: clean speech is
turned into "Lombard-like" speech per noise condition by shifting gain, pitch
and tempo with SoX (`speechain/datasets/pyscripts/lombard_synth.py`,
Section IV-A of the paper), and separately mixed with noise at training/test
time by `speechain/module/augment/noise_mixer.py`.

Starting from an already-dumped, MFA-aligned SpeeChain dataset (see
`data/README.md` and `data/mfa_preparation.sh`):

```bash
python speechain/datasets/pyscripts/lombard_synth.py \
  --wav_path data/ljspeech/data/wav \
  --mfa_path data/ljspeech/data/mfa/acoustic=english_us_arpa_lexicon=english_us_arpa \
  --mfa_subdir stress/no-punc \
  --asr_text_file idx2no-punc_text \
  --output_path data/ljspeech/data/lombard \
  --subsets train valid test \
  --conditions clean snr0 snr-10 \
  --babble_idx2wav data/librispeech/data/wav/dev-clean/idx2wav \
  --ncpu 8
```

This produces, per condition (`clean`, `snr0`, `snr-10`) and subset
(`train`/`valid`/`test`), an `idx2wav` / `idx2wav_ref` / `idx2text` /
`idx2duration` / `idx2text_asr` / `idx2snr_cond` set under
`data/ljspeech/data/lombard/{condition}/{subset}/`, plus
`data/ljspeech/data/lombard/noise/{white,babble}.wav` for the noise mixer.
Utterance IDs are suffixed with `-{condition}` (e.g. `LJ001-0004-clean`) so
the three conditions can be concatenated directly in `data_cfg` (see the
`exp_cfg`'s `data_cfg.train.conf.dataset_conf.main_data` lists).

Use `--max_utts N` for a quick smoke-test run over only the first `N`
utterances of each subset before committing to the full corpus.

## 2. Training

```bash
bash recipes/run.sh \
  --task lombard_tts --dataset ljspeech \
  --exp_cfg 22.05khz_mfa_lombard_fastspeech2.yaml \
  --ngpu 1 --train true --test false
```

This fine-tunes from `pretrained_tts` and freezes the ASR listener
(`asr_exp_path`/`asr_test_model` in the `exp_cfg`). With the default config
(30 epochs, `early_stopping_patience: 10`), expect early stopping well before
epoch 30 on a single fine-tuning run — the general TTS losses (mel, pitch,
energy, duration) and the SNR-classification accuracy converge quickly since
the backbone is warm-started; `fb_asr_loss` is a **diagnostic** metric (the
frozen ASR's confusion on real, noise-mixed reference audio used as feedback
*input*) rather than a training objective, so it staying flat across epochs
is expected, not a sign that training has stalled.

Checkpoints, TensorBoard logs and mel/attention snapshots are written to
`recipes/lombard_tts/ljspeech/exp/22.05khz_mfa_lombard_fastspeech2/`.

## 3. Evaluation

The `exp_cfg`'s built-in `infer_cfg` runs the test set through five
conditions — clean/no-adaptation, and `{snr0, snr-10} x {white, babble}` with
the closed loop enabled — and reports both the closed-loop result
(`cer`/`wer`) and the same model's single-pass, no-feedback result
(`cer_loop0`/`wer_loop0`) for a same-model, apples-to-apples comparison:

```bash
bash recipes/run.sh \
  --task lombard_tts --dataset ljspeech \
  --exp_cfg 22.05khz_mfa_lombard_fastspeech2.yaml \
  --ngpu 1 --train false --test true
```

Per-condition results land under
`recipes/lombard_tts/ljspeech/exp/22.05khz_mfa_lombard_fastspeech2/<condition>/<test_model>/test/`,
including `overall_results.md`, per-utterance `idx2cer` / `idx2cer_loop0` /
`idx2wer` / `idx2wer_loop0` / `idx2loops_used`, and the synthesized `wav/` and
noise-mixed `noisy_wav/` audio for the *closed-loop* result only (loop 0's
audio isn't saved separately — see below to get it explicitly).

The full 523-utterance test set across all five conditions is slow (each
noisy condition re-synthesizes + re-listens up to `max_loops` times per
utterance); for a quick check, point `--data_cfg` at a small custom
`idx2text`/`idx2text_asr`/`idx2wav_len` subset instead (any absolute path
works, no need to place it under `recipes/lombard_tts/ljspeech/data_cfg/`):

```bash
bash recipes/run.sh \
  --task lombard_tts --dataset ljspeech \
  --exp_cfg 22.05khz_mfa_lombard_fastspeech2.yaml \
  --data_cfg /absolute/path/to/subset_data_cfg.yaml \
  --ngpu 1 --train false --test true
```

To get the standard-TTS (no adaptation) audio explicitly, in the same noise
condition, for a direct listening comparison against the closed-loop result,
run a second pass with a single flat `infer_cfg` (see
`config/infer/lombard_tts/standard_tts.yaml` for the template) setting
`max_loops: 0` and the matching `snr`/`noise_type` — pass it via
`--infer_cfg /path/to/your.yaml`.

## Known gotchas

- **`infer_cfg` shared vs. exclusive args**: `runner.py` rejects any key that
  appears in both `infer_cfg.shared_args` and an `infer_cfg.exclu_args` entry
  (`ValueError: Find a duplicate argument ... in both 'shared_args' and
  'exclu_args'`). Since the "clean" condition needs `max_loops: 0` while the
  noisy conditions need `max_loops: 4`, `max_loops` must live in each
  `exclu_args` entry, never in `shared_args` — see how this `exp_cfg` does it.
- **`recipes/` is gitignored by default** (`.gitignore` has a blanket
  `recipes/` rule; only `config/feat/` and `config/infer/` are carved back out
  under `config/`). New files under `recipes/lombard_tts/` (this README,
  `exp_cfg/`, `data_cfg/`) need `git add -f`, or they will silently never be
  committed even by `git add -A`.
- Only run one training/testing job against the same experiment directory at
  a time — a second, concurrent `run.sh` invocation against the same
  `--train_result_path` will race on the same result files.

# Changelog

All notable changes to this project will be documented in this file.

## [unreleased]

### ✨ New Features

- Standalone inference engine `speechain/inference.py`: apply a trained ASR/TTS model directly to your own inputs (audio files for ASR, raw sentences for TTS) with only the experiment folder of the model — no dataset metadata is required. See `docs/inference.md` for the full document.
- Off-the-shelf inference configurations in `config/infer/`: `asr/greedy_decoding.yaml`, `asr/beam_search.yaml`, `asr/beam_search_lm.yaml`, and `tts/default.yaml`. The `config/feat/` and `config/infer/` folders are no longer ignored by git so that they can be shared across machines.
- Safe checkpoint loading utility `load_model_state_dict()` in `speechain/utilbox/data_loading_util.py`: checkpoints are loaded in the safe mode of `torch.load` (`weights_only=True`) by default, with an opt-in fallback (`--trust_checkpoint` for `speechain/inference.py`) for legacy checkpoints.

### 🐛 Bug Fixes

- Skip the unnecessary resampling for the dumped acoustic features whose sampling rate is unknown, cache the resampler of each encountered sampling rate, and keep the sampling rate consistent after the on-the-fly resampling (`speechain/dataset/speech_text.py`)
- Bugs in metadata_generator.py librispeech
- Conflict in lm_text/exp_cfg/100-*
- Linting
- Linting

### 💼 Other

- The built-in LM copies (`lm_model_cfg.yaml` & `lm_model.pth`) and the tokenizer backups (`token_vocab` & `token_model`) are only (re)written when they are absent or outdated, so inference is no longer interrupted by a read-only or shared experiment folder
- Vocoder failures during TTS inference now raise a warning instead of failing silently (`speechain/model/nar_tts.py`)
- __init__.py

### 📚 Documentation

- Add the document of the standalone inference engine (`docs/inference.md`) and update the handbook, index, README, and ASR/TTS recipe READMEs for it
- Fix format
- Update
- Update tts

### 🎨 Styling

- Formatted with black

### ⚙️ Miscellaneous Tasks

- Add inverted logo
- Update gitignore
- Update gitignore
- Clean repo
- Clean repo
- Replace humanfriendly package with python code, add test
- Clean requirements
- Update CHANGELOG for version 0.1.2, add new requirements.txt and humanfriendly.py
- Update docs

## [0.1.2] - 2024-12-11 

### Added
- a new requirements.txt file
- `humanfriendly.py` in `utilbox` along with its test file


## [0.1.1] - 2024-09-30

### Added
- yaml file is added `recipes/asr/librispeech/train-960/exp_cfg/960-bpe5k_transformer-wide_ctc_perturb.yaml`
- train-clean-5 recipe

### Changed
- repo name from SpeeChain to speechain
- bacth size to 2.4e7 (from 2.4e7) in yaml file above.
- `envir_preparation.sh` to `create_env.sh`

### Fixed
- Error in `meta_generator.py` for librispeech dataset


## [0.1.0] - 2024-09-26 
- Forked from `heli-qi/SpeeChain
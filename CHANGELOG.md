# Changelog

All notable changes to this project will be documented in this file.

## [unreleased]

### ✨ New Features

- Standalone inference engine `speechain/inference.py`: apply a trained ASR/TTS model directly to your own inputs (audio files for ASR, raw sentences for TTS) with only the experiment folder of the model — no dataset metadata is required. See `docs/inference.md` for the full document.
- Off-the-shelf inference configurations in `config/infer/`: `asr/greedy_decoding.yaml`, `asr/beam_search.yaml`, `asr/beam_search_lm.yaml`, and `tts/default.yaml`. The `config/feat/` and `config/infer/` folders are no longer ignored by git so that they can be shared across machines.
- Safe checkpoint loading utility `load_model_state_dict()` in `speechain/utilbox/data_loading_util.py`: checkpoints are loaded in the safe mode of `torch.load` (`weights_only=True`) by default, with an opt-in fallback (`--trust_checkpoint` for `speechain/inference.py`) for legacy checkpoints.

### 🐛 Bug Fixes

- Skip the unnecessary resampling for the dumped acoustic features whose sampling rate is unknown, cache the resampler of each encountered sampling rate, and keep the sampling rate consistent after the on-the-fly resampling (`speechain/dataset/speech_text.py`)
- `setup.py` only installed the top-level `speechain` package without any of its sub-packages (`find_packages(include=['speechain'])` does not match sub-packages) and shipped a top-level `datasets` package that could shadow the HuggingFace `datasets` package after installation
- The bare `datasets` pattern in `.gitignore` also ignored the shared dataset-dumping code after it was moved into `speechain/datasets/`; negation rules are added so that the moved code is always tracked by git
- `lab_file_generator.py` aborted the whole multiprocessing chunk when meeting an existing `.lab` file (`return` instead of `continue`), so re-running MFA preparation on a partially-generated corpus silently skipped most of the remaining utterances
- `np.loadtxt` in `data_packager.py` and `vocab_generator.py` crashed with `IndexError` when the loaded metadata file contains only one line (`ndmin=2` is now given)
- `vocab_generator.py` crashed with `IndexError` when a transcript starts with a punctuation mark (the phoneme list is empty at that point)
- Correct the wrong default value of `txt_format` in the help messages of `meta_generator.py` and `data_dumping.sh` (`normal` → `no-punc`)
- Bugs in metadata_generator.py librispeech
- Conflict in lm_text/exp_cfg/100-*
- Linting
- Linting

### ♻️ Code Refactoring

- Move all the shared dataset-dumping Python code from the top-level `datasets/` folder into `speechain/datasets/` (issue #2): the abstract base classes (`meta_generator.py`, `meta_post_processor.py`) and the fixed executable scripts (`pyscripts/`). This resolves the import ambiguity with the HuggingFace `datasets` package and brings the code under the CI checks (Black & Ruff) that only scan the `speechain` directory. Per-dataset scripts under `datasets/{dataset_name}/` now import the base classes from the `speechain` package, and `datasets/data_dumping.sh` & `datasets/mfa_preparation.sh` point to the new script locations

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
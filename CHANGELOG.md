Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Types of changes: Added, Changed, Deprecated, Removed, Fixed, Security

## [Unreleased]
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
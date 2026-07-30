"""Everything about datasets in the SpeeChain toolkit.

This sub-package is the single place for all dataset-related code:

- ``abs.py``: the abstract ``Dataset`` base class that reads data instances
  from the disk into memory and packages them into batches.
- ``speech_text.py``: the built-in ``SpeechTextDataset`` implementation used
  by speech-text tasks (ASR, TTS, etc.).
- ``meta_generator.py``: the abstract base class for per-dataset metadata
  generation scripts (``datasets/{dataset_name}/meta_generator.py``).
- ``meta_post_processor.py``: the abstract base class for per-dataset
  metadata post-processing scripts
  (``datasets/{dataset_name}/meta_post_processor.py``).
- ``pyscripts/``: the fixed executable scripts used by
  ``datasets/data_dumping.sh`` and ``datasets/mfa_preparation.sh``
  (feature extraction, waveform downsampling, vocabulary generation, etc.).

The dataset-dumping code lives inside the ``speechain`` package (instead of
the top-level ``datasets`` folder) so that:

1. it does not shadow the HuggingFace ``datasets`` package, and
2. it is covered by the CI checks (Black & Ruff) that only scan the
   ``speechain`` directory.
"""

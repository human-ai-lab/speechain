"""Shared dataset-dumping code of the SpeeChain toolkit.

This sub-package hosts the Python code that is shared by all the dataset
folders under ``${SPEECHAIN_ROOT}/datasets/``:

- ``meta_generator.py``: the abstract base class for per-dataset metadata
  generation scripts (``datasets/{dataset_name}/meta_generator.py``).
- ``meta_post_processor.py``: the abstract base class for per-dataset
  metadata post-processing scripts
  (``datasets/{dataset_name}/meta_post_processor.py``).
- ``pyscripts/``: the fixed executable scripts used by
  ``datasets/data_dumping.sh`` and ``datasets/mfa_preparation.sh``
  (feature extraction, waveform downsampling, vocabulary generation, etc.).

The code lives inside the ``speechain`` package (instead of the top-level
``datasets`` folder) so that:

1. it does not shadow the HuggingFace ``datasets`` package, and
2. it is covered by the CI checks (Black & Ruff) that only scan the
   ``speechain`` directory.
"""

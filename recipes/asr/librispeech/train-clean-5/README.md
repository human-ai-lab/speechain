# ASR Recipe: LibriSpeech `train-clean-5`

Conformer-based Automatic Speech Recognition (ASR) trained on the `train-clean-5`
subset of [LibriSpeech](https://www.openslr.org/12). Uses a small Conformer encoder +
Transformer decoder with BPE-1k tokenization.

## Model Overview

| Property | Value |
|---|---|
| Model | Conformer-small (13.79 M parameters) |
| Tokenizer | SentencePiece BPE-1k |
| Training set | `train-clean-5` |
| Validation set | `dev-clean-2` |
| Test set | `test-clean` |
| Required GPU | 1 × NVIDIA RTX 3090 (24 GB) |
| Training time | ~4 hours |
| Expected WER (test-clean) | ~42 % (ASR-only) |

## Available Experiment Configs (`exp_cfg/`)

| Config | Description |
|---|---|
| `5-bpe1k_conformer-small_lr2e-3.yaml` | Baseline – no speed perturbation, no SpecAugment, ASR-only decoding |
| `5-bpe1k_conformer-small_lr2e-3b.yaml` | Enhanced – speed perturbation + SpecAugment, ASR+LM decoding |

## Steps to Run

### 1. Set Up the Environment

Run the following **once** from the repository root:

```bash
source envir_preparation.sh
```

This exports two required variables:
- `SPEECHAIN_ROOT` – absolute path to the repository root
- `SPEECHAIN_PYTHON` – Python interpreter for the speechain environment

### 2. Prepare the Dataset

Use the dataset scripts under `datasets/librispeech/` to download and dump the data.
After preparation the following paths must exist:

```
datasets/librispeech/data/wav/train-clean-5/
datasets/librispeech/data/wav/dev-clean-2/
datasets/librispeech/data/wav/test-clean/
```

### 3. Train

```bash
cd recipes/asr/librispeech/train-clean-5

bash run.sh --exp_cfg 5-bpe1k_conformer-small_lr2e-3.yaml \
            --train true --test false
```

### 4. Test

```bash
bash run.sh --exp_cfg 5-bpe1k_conformer-small_lr2e-3.yaml \
            --train false --test true
```

Results and checkpoints are saved under `exp/<exp_name>/`.

---

## Common Variants

### Train + Test in one command

```bash
# do not specify --train and --test to run both by default, it cannot be set both true on the same command line 
# set it through the config file instead, or just run without these arguments to use the default values  
bash run.sh --exp_cfg 5-bpe1k_conformer-small_lr2e-3.yaml 
```

### Resume interrupted training

```bash
bash run.sh --exp_cfg 5-bpe1k_conformer-small_lr2e-3.yaml \
            --train true --test false --resume true
```

### Multi-GPU training

```bash
bash run.sh --exp_cfg 5-bpe1k_conformer-small_lr2e-3.yaml \
            --train true --test false --ngpu 2
```

### Test a specific averaged checkpoint

```bash
bash run.sh --exp_cfg 5-bpe1k_conformer-small_lr2e-3.yaml \
            --train false --test true \
            --test_model 10_valid_accuracy_average
```

### Evaluate on `dev-clean-2` (inference tuning)

```bash
bash run.sh --exp_cfg 5-bpe1k_conformer-small_lr2e-3.yaml \
            --train false --test true \
            --data_cfg test_dev-clean.yaml
```

### Evaluate on `test-clean` explicitly

```bash
bash run.sh --exp_cfg 5-bpe1k_conformer-small_lr2e-3.yaml \
            --train false --test true \
            --data_cfg test_test-clean.yaml
```

---

## Script Arguments Reference

| Argument | Default | Description |
|---|---|---|
| `--exp_cfg` | *(required)* | Config file in `exp_cfg/` |
| `--train` | `true` | Run training |
| `--test` | `true` | Run testing |
| `--resume` | `false` | Resume from last checkpoint |
| `--ngpu` | — | Number of GPUs |
| `--gpus` | — | Specific GPU IDs (`null` = auto) |
| `--accum_grad` | — | Gradient accumulation steps |
| `--data_cfg` | — | Override data config from `data_cfg/` |
| `--infer_cfg` | — | Override inference config |
| `--test_model` | — | Checkpoint name to test |
| `--dry_run` | `false` | Print commands without running |

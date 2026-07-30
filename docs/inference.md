# Standalone Inference

SpeeChain provides a standalone inference engine, `speechain/inference.py`, that applies a trained ASR or TTS model directly to **your own inputs** — no dataset dumping or metadata preparation is required:

* **ASR:** give one or more audio files (wav/flac) and get their transcripts.
* **TTS:** give one or more raw text sentences and get the synthetic waveforms (a HiFi-GAN vocoder is automatically downloaded on the first use).

This is different from the standard testing branch of `runner.py` (launched by `recipes/run.sh --train false`), which evaluates the model on the test sets described by the dumped dataset metadata (`idx2wav`, `idx2text`, `data_cfg`, ...) and produces the multi-level evaluation reports inside the experiment folder.

## Table of Contents
1. [Requirements](#requirements)
2. [Quick Start](#quick-start)
3. [Command-Line Arguments](#command-line-arguments)
4. [Inference Configurations](#inference-configurations)
5. [Outputs](#outputs)
6. [Notes and Tips](#notes-and-tips)

## Requirements

Only the experiment folder of a trained model is required, which must contain:

1. `exp_cfg.yaml` (or `train_cfg.yaml`): the configuration files saved during training.
2. `models/{test_model}.pth`: the checkpoint of the model you want to use for inference.

The script registers the toolkit root into both the environmental variables and `sys.path` by itself, so it can be executed from anywhere (not necessarily the toolkit root), and the in-toolkit relative paths (e.g., `recipes/...`, `config/...`) are resolved from `SPEECHAIN_ROOT`.

👆[Back to the table of contents](#table-of-contents)

## Quick Start

### ASR: transcribe your own audio files

```bash
# greedy decoding is used by default
python speechain/inference.py \
    --exp_path recipes/asr/librispeech/train-clean-100/exp/100-bpe5k_conformer-small_lr2e-3 \
    --test_model latest \
    --audio /path/to/utterance1.wav /path/to/utterance2.flac

# beam-search decoding with the hyperparameters given by an inline Dict
python speechain/inference.py \
    --exp_path recipes/asr/librispeech/train-clean-100/exp/100-bpe5k_conformer-small_lr2e-3 \
    --test_model 10_valid_accuracy_average \
    --infer_cfg "beam_size:16,ctc_weight:0.3" \
    --audio /path/to/utterance.wav

# the decoding hyperparameters can also be given by an off-the-shelf configuration file
python speechain/inference.py \
    --exp_path recipes/asr/librispeech/train-clean-100/exp/100-bpe5k_conformer-small_lr2e-3 \
    --test_model 10_valid_accuracy_average \
    --infer_cfg config/infer/asr/beam_search_lm.yaml \
    --audio /path/to/utterance.wav --output_path ./transcripts
```

### TTS: synthesize your own sentences

```bash
# one or more sentences can be given by --text
python speechain/inference.py \
    --exp_path recipes/tts/ljspeech/exp/22.05khz_mfa_fastspeech2 \
    --test_model latest \
    --text "This is a test of the SpeeChain toolkit." \
    --output_path ./syn_wavs

# or give a text file where each line is a sentence to be synthesized
python speechain/inference.py \
    --exp_path recipes/tts/ljspeech/exp/22.05khz_mfa_fastspeech2 \
    --test_model latest \
    --text_file /path/to/sentences.txt \
    --output_path ./syn_wavs
```

👆[Back to the table of contents](#table-of-contents)

## Command-Line Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--exp_path` | *(required)* | The path of the experiment folder of your trained model. There must be `exp_cfg.yaml` (or `train_cfg.yaml`) and `models/` in this folder. |
| `--test_model` | `latest` | The name of the checkpoint in `{exp_path}/models/` used for inference, e.g., `latest`, `10_valid_accuracy_average`, `epoch_100`. Both `.pth` and `.mdl` files are accepted. |
| `--infer_cfg` | `None` | The inference (decoding) configuration, given either as an inline Dict string (e.g., `"beam_size:16,ctc_weight:0.3"`) or as the path of a .yaml configuration file (e.g., `config/infer/asr/beam_search.yaml`). See [Inference Configurations](#inference-configurations) for the resolution priority. |
| `--audio` | `None` | *(ASR only)* One or more paths of the audio files (wav/flac) to be transcribed. |
| `--text` | `None` | *(TTS only)* One or more raw text sentences to be synthesized. |
| `--text_file` | `None` | *(TTS only)* The path of a text file where each line is a sentence to be synthesized. Ignored if `--text` is given. |
| `--output_path` | `None` | The path where the inference results are saved. A relative path is interpreted from your current working directory. If not given, ASR results are only printed while TTS waveforms are saved to `{exp_path}/standalone_inference/`. |
| `--device` | `None` | The device used for inference, e.g., `cpu`, `cuda`, `cuda:1`. It has the higher priority than `--gpu`. |
| `--gpu` | `-1` | The GPU id used for inference. `-1` means using the first GPU if CUDA is available, otherwise using the CPU. Please give `--device cpu` if you want to use the CPU on a machine where CUDA is available. |
| `--trust_checkpoint` | `False` | Whether to load the checkpoint in the unsafe mode of `torch.load()` if the safe mode fails. Only give this argument for the checkpoints whose source you trust (see [Notes and Tips](#notes-and-tips)). |

The task type (ASR or TTS) is automatically decided by the `model_type` recorded in the experiment configuration, so `--audio` is ignored for a TTS model and `--text`/`--text_file` are ignored for an ASR model.

👆[Back to the table of contents](#table-of-contents)

## Inference Configurations

The inference (decoding) configuration is resolved with the following priority:

1. `--infer_cfg` given in the terminal (an inline Dict string or the path of a .yaml file);
2. `infer_cfg` recorded in `exp_cfg.yaml` of your experiment;
3. the default decoding configuration of the model.

If multiple configurations are given (a _List_ or the `shared_args`/`exclu_args` pair, see [the handbook](./handbook.md#inference-configuration-for-hyperparameter-adjustment)), only the first one is used because this script produces only one decoding result for each input.

SpeeChain provides the following off-the-shelf inference configurations in `config/infer/`, which are shared by both the standalone inference engine and the standard testing branch of `runner.py`:

| Configuration file | Task | Content |
| --- | --- | --- |
| `config/infer/asr/greedy_decoding.yaml` | ASR | Greedy decoding (`beam_size: 1`). |
| `config/infer/asr/beam_search.yaml` | ASR | Beam-search decoding with CTC joint scoring (`beam_size: 16, ctc_weight: 0.2`). `ctc_weight` is silently ignored if your ASR model was trained without the CTC loss. |
| `config/infer/asr/beam_search_lm.yaml` | ASR | Beam-search decoding with CTC & LM joint scoring (`beam_size: 16, ctc_weight: 0.3, lm_weight: 0.6`). Requires the language model of your ASR model to be available (i.e., `lm_model_cfg` and `lm_model_path` in `model['customize_conf']` of your exp_cfg, or the built-in `lm_model_cfg.yaml` and `lm_model.pth` in your train_result_path). |
| `config/infer/tts/default.yaml` | TTS | Default generation with the HiFi-GAN neural vocoder (`vocoder: hifigan, return_wav: true, return_feat: false`). |

For the details of all available decoding arguments, please refer to the docstring of `speechain/infer_func/beam_search.py` for ASR models and the docstring of `inference()` in `speechain/model/nar_tts.py` and `speechain/model/ar_tts.py` for TTS models.

👆[Back to the table of contents](#table-of-contents)

## Outputs

* **ASR:** the transcript of each input audio file is printed in the terminal. If `--output_path` is given, a `transcripts.txt` is also saved there where each line is `{utterance_idx} {transcript}`. The utterance indices are made from the file names of your audio files (the parent folder names and the occurrence order are attached when some files share the same name).
* **TTS:** the synthetic waveforms are saved as `syn_001.wav`, `syn_002.wav`, ... in `--output_path` (`{exp_path}/standalone_inference/` by default), together with an `idx2text.txt` that records the mapping from the waveform indices to your input sentences.

👆[Back to the table of contents](#table-of-contents)

## Notes and Tips

* **Audio preprocessing (ASR):** multi-channel waveforms are averaged into a single channel before inference, and the input audio is resampled on the fly if its sampling rate is different from the one of the model (the resampler of each encountered sampling rate is cached to avoid re-creating it for every utterance).
* **Vocoder (TTS):** the non-autoregressive TTS models (`speechain/model/nar_tts.py`) convert the generated acoustic features into waveforms by the HiFi-GAN vocoder, which is downloaded from SpeechBrain on the first use and is only available for the models working on 16kHz or 22.05kHz. Please give `vocoder: gl` in your `infer_cfg` for the other sampling rates. The autoregressive TTS models (`speechain/model/ar_tts.py`) only support Griffin-Lim vocoding and take `return_gl_wav` instead of `vocoder` & `return_wav`.
* **Multi-speaker TTS:** no reference speaker can be given to this script. A random reference speaker is picked up by the model for each sentence, so the voice of the synthetic waveforms is neither controllable nor reproducible.
* **Checkpoint safety:** the checkpoints dumped by SpeeChain only contain the `state_dict()` of a Model object, so they are loaded in the safe mode of `torch.load()` (`weights_only=True`) where no pickled Python object is executed. If your checkpoint is in an old format that cannot be loaded in the safe mode, please make sure that you trust where it comes from and then give `--trust_checkpoint` to load it in the unsafe mode.
* **Read-only experiment folders:** the standalone inference engine never rewrites the files in your experiment folder (e.g., the built-in LM copies `lm_model_cfg.yaml` & `lm_model.pth` and the tokenizer backups `token_vocab` & `token_model` are only created when they are absent or outdated), so it works properly even on a read-only or shared experiment folder.

👆[Back to the table of contents](#table-of-contents)

"""
    Standalone inference script for trained ASR and TTS models.

    Different from the standard testing branch of `runner.py` which is driven by the dumped dataset metadata
    (idx2wav, idx2text, data_cfg, ...), this script performs free-form inference directly on your own input:
        1. ASR: give one or more audio files (wav/flac) and get their transcripts.
        2. TTS: give one or more raw text sentences and get the synthetic waveforms (a HiFi-GAN vocoder is
           automatically downloaded on the first use).

    Only the experiment folder of a trained model is required, which must contain:
        1. `exp_cfg.yaml` (or `train_cfg.yaml`): the configuration files saved during training.
        2. `models/{test_model}.pth`: the checkpoint of the model you want to use for inference.

    Usage examples:
        # ASR: transcribe audio files with a trained Conformer ASR model (greedy decoding by default)
        python speechain/inference.py \\
            --exp_path recipes/asr/librispeech/train-clean-100/exp/100-bpe5k_conformer-small_lr2e-3 \\
            --test_model latest \\
            --audio /path/to/utterance1.wav /path/to/utterance2.flac

        # ASR with beam-search decoding hyperparameters given by an inline Dict or a .yaml file
        python speechain/inference.py \\
            --exp_path recipes/asr/librispeech/train-clean-100/exp/100-bpe5k_conformer-small_lr2e-3 \\
            --test_model 10_valid_accuracy_average \\
            --infer_cfg "beam_size:16,ctc_weight:0.3" \\
            --audio /path/to/utterance.wav

        # TTS: synthesize speech from raw sentences with a trained FastSpeech2 model
        python speechain/inference.py \\
            --exp_path recipes/tts/ljspeech/exp/22.05khz_mfa_fastspeech2 \\
            --test_model latest \\
            --text "This is a test of the SpeeChain toolkit." \\
            --output_path ./syn_wavs
"""

import argparse
import copy
import os
import sys
from typing import Dict, List

# register the toolkit root into both the environmental variables and sys.path so that
# 1. the speechain package can be imported no matter where this script is executed from;
# 2. in-toolkit relative paths in the configuration files (e.g., data/..., recipes/...) can be resolved
_TOOLKIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SPEECHAIN_ROOT", _TOOLKIT_ROOT)
sys.path.insert(0, _TOOLKIT_ROOT)

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

from speechain.utilbox.data_loading_util import load_model_state_dict  # noqa: E402
from speechain.utilbox.import_util import import_class, parse_path_args  # noqa: E402
from speechain.utilbox.type_util import str2dict  # noqa: E402
from speechain.utilbox.yaml_util import load_yaml  # noqa: E402


def parse():
    parser = argparse.ArgumentParser(
        description="Standalone inference for trained ASR/TTS models.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        required=True,
        help="The path of the experiment folder of your trained model. "
        "There must be 'exp_cfg.yaml' (or 'train_cfg.yaml') and 'models/' in this folder.",
    )
    parser.add_argument(
        "--test_model",
        type=str,
        default="latest",
        help="The name of the checkpoint in {exp_path}/models/ used for inference, "
        "e.g., latest, 10_valid_accuracy_average, epoch_100. (default: latest)",
    )
    parser.add_argument(
        "--infer_cfg",
        type=str,
        default=None,
        help="The inference (decoding) configuration. It can be given either as an inline Dict string "
        '(e.g., "beam_size:16,ctc_weight:0.3") or as the path of a .yaml configuration file '
        "(e.g., config/infer/asr/beam_search.yaml). "
        "If not given, the 'infer_cfg' recorded in exp_cfg.yaml will be used; "
        "if it is also not given or not usable, the default decoding configuration of the model is used.",
    )
    parser.add_argument(
        "--audio",
        type=str,
        nargs="+",
        default=None,
        help="(ASR only) One or more paths of the audio files (wav/flac) to be transcribed.",
    )
    parser.add_argument(
        "--text",
        type=str,
        nargs="+",
        default=None,
        help="(TTS only) One or more raw text sentences to be synthesized.",
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="(TTS only) The path of a text file where each line is a sentence to be synthesized. "
        "Ignored if --text is given.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="The path where the inference results are saved. "
        "For ASR, a 'transcripts.txt' will be saved; for TTS, the synthetic waveforms will be saved. "
        "A relative path is interpreted from your current working directory. "
        "If not given, ASR results are only printed while TTS waveforms are saved to "
        "'{exp_path}/standalone_inference/'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="The device used for inference, e.g., 'cpu', 'cuda', 'cuda:1'. "
        "It has the higher priority than --gpu. (default: None)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="The GPU id used for inference. -1 means using the first GPU if CUDA is available, "
        "otherwise using the CPU. Please give --device cpu if you want to use the CPU on a machine "
        "where CUDA is available. (default: -1)",
    )
    parser.add_argument(
        "--trust_checkpoint",
        action="store_true",
        help="Whether to load the checkpoint in the unsafe mode of torch.load() if the safe mode fails. "
        "Loading a checkpoint in the unsafe mode executes the pickled Python objects inside it, "
        "so please only give this argument for the checkpoints whose source you trust. (default: False)",
    )
    return parser.parse_args()


def resolve_input_path(input_path: str) -> str:
    """Resolve the path of an existing input file or folder given in the terminal.

    The given path is first interpreted from your current working directory like a normal terminal
    argument. If there is nothing there, it is interpreted as an in-toolkit relative path (i.e., a
    relative path from SPEECHAIN_ROOT) so that the in-toolkit paths in the usage examples
    (e.g., recipes/..., config/...) also work outside the toolkit root.
    """
    if os.path.exists(input_path):
        return os.path.abspath(input_path)

    toolkit_path = parse_path_args(input_path)
    assert os.path.exists(
        toolkit_path
    ), f"Your given path {input_path} doesn't exist! Please check your input arguments."
    return toolkit_path


def get_device(device: str = None, gpu: int = -1) -> torch.device:
    """Decide the computational device used for inference by your given arguments."""
    # the explicitly-given device has the highest priority
    if device is not None:
        return torch.device(device)

    # a specific GPU is given by its id
    if gpu >= 0:
        assert torch.cuda.is_available(), "CUDA is not available on your machine!"
        assert gpu < torch.cuda.device_count(), (
            f"Your machine only has {torch.cuda.device_count()} GPUs, "
            f"but got --gpu {gpu}! Please give a GPU id smaller than {torch.cuda.device_count()}."
        )
        return torch.device(f"cuda:{gpu}")

    # automatic decision: the first GPU if CUDA is available, otherwise the CPU
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_exp_cfg(exp_path: str) -> Dict:
    """Load the configuration of the experiment placed in `{exp_path}`.

    `exp_cfg.yaml` is dumped from the arguments of `runner.py`, so its 'train_cfg' may be either the
    content of the train configuration (when it is inlined in your exp_cfg) or merely the path of the
    train configuration file (when it is given by `--train_cfg`). In the latter case, the resolved
    `train_cfg.yaml` dumped by `runner.py` into the same folder is used instead.

    Returns:
        The experiment configuration Dict whose 'train_cfg' is guaranteed to be a Dict containing 'model'.
    """
    exp_cfg_path = os.path.join(exp_path, "exp_cfg.yaml")
    train_cfg_path = os.path.join(exp_path, "train_cfg.yaml")
    assert os.path.exists(exp_cfg_path) or os.path.exists(train_cfg_path), (
        f"Neither {exp_cfg_path} nor {train_cfg_path} exists! "
        f"Please make sure that --exp_path points to the folder of a trained model."
    )

    # an empty .yaml file is parsed into None, so `or dict()` is attached here for safety
    exp_cfg = (
        (load_yaml(exp_cfg_path) or dict()) if os.path.exists(exp_cfg_path) else dict()
    )
    train_cfg = exp_cfg.get("train_cfg", None)

    # fall back to the dumped train_cfg.yaml if exp_cfg.yaml doesn't contain the train configuration itself
    if not isinstance(train_cfg, Dict):
        assert os.path.exists(train_cfg_path), (
            f"The 'train_cfg' in {exp_cfg_path} is not a Dict ({train_cfg}) and {train_cfg_path} doesn't "
            f"exist, so the configuration of your model cannot be obtained!"
        )
        train_cfg = load_yaml(train_cfg_path)

    assert isinstance(train_cfg, Dict) and "model" in train_cfg.keys(), (
        "Please make sure that either the 'train_cfg' in your exp_cfg.yaml or your train_cfg.yaml "
        "contains a 'model' tag!"
    )
    exp_cfg["train_cfg"] = train_cfg
    return exp_cfg


def build_model_from_exp(
    exp_cfg: Dict, exp_path: str, device: torch.device
) -> torch.nn.Module:
    """Build the model recorded in the given experiment configuration without the
    Runner/Iterator pipeline.

    Returns:
        The constructed Model object placed on the target device.
    """
    model_cfg = exp_cfg["train_cfg"]["model"]
    assert (
        "model_type" in model_cfg.keys()
    ), "Please specify the model_type in your configuration!"
    assert (
        "module_conf" in model_cfg.keys()
    ), "Please specify the module_conf in your configuration!"

    # the pretrained models used for parameter initialization during training are unnecessary for inference
    # (the final checkpoint will be loaded later), so they are removed to avoid unexpected loading failures
    # when the pretrained model files don't exist on the current machine
    model_conf = copy.deepcopy(model_cfg.get("model_conf", dict()))
    model_conf.pop("pretrained_model", None)

    model_class = import_class("speechain.model." + model_cfg["model_type"])
    model = model_class(
        model_conf=model_conf,
        module_conf=model_cfg["module_conf"],
        criterion_conf=model_cfg.get("criterion_conf", None),
        device=device,
        result_path=exp_path,
        non_blocking=False,
        distributed=False,
    )
    return model.to(device)


def load_checkpoint(
    model: torch.nn.Module,
    exp_path: str,
    test_model: str,
    device: torch.device,
    trust_checkpoint: bool = False,
):
    """Load `{exp_path}/models/{test_model}.pth` (or `.mdl` for older versions) into
    the model."""
    models_path = os.path.join(exp_path, "models")
    if os.path.exists(os.path.join(models_path, f"{test_model}.pth")):
        model_path = os.path.join(models_path, f"{test_model}.pth")
    elif os.path.exists(os.path.join(models_path, f"{test_model}.mdl")):
        model_path = os.path.join(models_path, f"{test_model}.mdl")
    else:
        raise RuntimeError(
            f"{os.path.join(models_path, test_model + '.pth')} is not found! "
            f"Please check your --test_model {test_model}."
        )
    model.load_state_dict(
        load_model_state_dict(
            model_path, map_location=device, trust_checkpoint=trust_checkpoint
        )
    )
    print(f"Checkpoint {model_path} is loaded.")


def resolve_infer_cfg(infer_cfg: str, exp_cfg: Dict) -> Dict:
    """Resolve the inference configuration from the user input and the experiment
    configuration.

    Priority: --infer_cfg in the terminal > 'infer_cfg' in exp_cfg.yaml > the default decoding configuration
    of the model (an empty Dict).
    """
    # use the one recorded in exp_cfg.yaml if the user doesn't give one in the terminal
    if infer_cfg is None and "infer_cfg" in exp_cfg.keys():
        infer_cfg = exp_cfg["infer_cfg"]

    # no inference configuration is given at all: use the default decoding configuration of the model
    if infer_cfg is None:
        return dict()

    # the configuration is given as the path of a .yaml file
    if isinstance(infer_cfg, str):
        infer_cfg = str2dict(infer_cfg)
        if isinstance(infer_cfg, str):
            infer_cfg = load_yaml(resolve_input_path(infer_cfg))

    # the configuration is given as a List of configurations: only the first one is used
    # because this script produces only one decoding result for each input
    if isinstance(infer_cfg, List):
        assert (
            len(infer_cfg) > 0
        ), "Your given infer_cfg is an empty List! Please give at least one configuration in it."
        print(
            f"Note: multiple inference configurations are given ({infer_cfg}). "
            "Only the first one is used."
        )
        infer_cfg = infer_cfg[0]
        if isinstance(infer_cfg, str):
            infer_cfg = load_yaml(resolve_input_path(infer_cfg))

    assert isinstance(
        infer_cfg, Dict
    ), f"infer_cfg should be resolved into a Dict, but got {infer_cfg}!"

    # the configuration is given by 'shared_args' & 'exclu_args':
    # the first exclusive configuration merged with the shared arguments is used
    if "shared_args" in infer_cfg.keys() or "exclu_args" in infer_cfg.keys():
        shared_args = infer_cfg.get("shared_args", dict())
        exclu_args = infer_cfg.get("exclu_args", [])
        assert isinstance(shared_args, Dict) and isinstance(exclu_args, List), (
            "If infer_cfg is given by 'shared_args' and 'exclu_args', "
            "infer_cfg['shared_args'] must be a Dict and infer_cfg['exclu_args'] must be a List."
        )

        chosen_cfg = dict(exclu_args[0]) if len(exclu_args) > 0 else dict()
        for cfg_key in chosen_cfg.keys():
            assert cfg_key not in shared_args.keys(), (
                f"Find a duplicate argument {cfg_key} in both 'shared_args' and 'exclu_args' of your "
                f"infer_cfg! Please only give it in one of them."
            )
        chosen_cfg.update(shared_args)
        print(
            f"Note: the inference configuration given by 'shared_args/exclu_args' is used: {chosen_cfg}"
            + (
                f" (the first one of the {len(exclu_args)} exclusive configurations)"
                if len(exclu_args) > 1
                else ""
            )
        )
        infer_cfg = chosen_cfg
    return infer_cfg


def get_utterance_ids(audio_paths: List[str]) -> List[str]:
    """Turn the paths of the input audio files into their unique utterance indices.

    The file name is used as the index. If some files share the same name (they are placed in different
    folders), the names of their parent folders are attached to distinguish them from each other.
    """
    idx_list = [os.path.splitext(os.path.basename(path))[0] for path in audio_paths]
    if len(set(idx_list)) == len(idx_list):
        return idx_list

    # disambiguate the duplicated file names by the folders where the files are placed
    idx_list = [
        "-".join(
            os.path.splitext(os.path.normpath(path))[0].strip(os.sep).split(os.sep)[-2:]
        )
        for path in audio_paths
    ]
    if len(set(idx_list)) == len(idx_list):
        return idx_list

    # the still-duplicated indices are numbered by their occurrence order in your input
    idx_freq_dict, unique_idx_list = dict(), []
    for idx in idx_list:
        idx_freq_dict[idx] = idx_freq_dict.get(idx, 0) + 1
        unique_idx_list.append(
            idx if idx_freq_dict[idx] == 1 else f"{idx}_{idx_freq_dict[idx]}"
        )
    return unique_idx_list


@torch.inference_mode()
def asr_inference(
    model, audio_paths: List[str], infer_conf: Dict, device: torch.device, output_path
):
    """Transcribe the given audio files by the ASR model one utterance at a time."""
    transcripts, resamplers = [], {}
    for audio_path in audio_paths:
        # read the waveform: (n_sample, n_channel)
        wav, sample_rate = sf.read(
            resolve_input_path(audio_path), always_2d=True, dtype="float32"
        )
        wav = torch.from_numpy(wav)

        # the frontends of this toolkit only accept single-channel waveforms,
        # so the multi-channel ones are downmixed into one channel here
        if wav.size(-1) > 1:
            print(
                f"Note: {audio_path} has {wav.size(-1)} channels, "
                f"which are averaged into a single channel before inference."
            )
            wav = wav.mean(dim=-1, keepdim=True)
        # the resamplers and the model are placed on `device`, so the waveform must be moved there first
        wav = wav.to(device)

        # on-the-fly resampling if the sampling rate of the input audio is different from the one of the model.
        # the resampler of each encountered sampling rate is cached to avoid re-creating it for every utterance
        if sample_rate != model.sample_rate:
            if sample_rate not in resamplers.keys():
                resamplers[sample_rate] = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=model.sample_rate
                ).to(device)
            wav = resamplers[sample_rate](wav.squeeze(-1)).unsqueeze(-1)

        # do the decoding: feat (1, n_sample, 1), feat_len (1,)
        outputs = model.inference(
            infer_conf,
            feat=wav.unsqueeze(0),
            feat_len=torch.LongTensor([wav.size(0)]).to(device),
            decode_only=True,
        )
        transcripts.append(outputs["text"]["content"][0])
        print(f"{audio_path}\n  -> {transcripts[-1]}")

    # save the transcripts to the disk if specified
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        save_file = os.path.join(output_path, "transcripts.txt")
        with open(save_file, "w", encoding="utf-8") as f:
            for utt_idx, transcript in zip(get_utterance_ids(audio_paths), transcripts):
                f.write(f"{utt_idx} {transcript}\n")
        print(f"\nTranscripts are saved to {save_file}")
    return transcripts


@torch.inference_mode()
def tts_inference(
    model, sentences: List[str], infer_conf: Dict, device: torch.device, output_path
):
    """Synthesize the given raw text sentences by the TTS model one sentence at a
    time."""
    os.makedirs(output_path, exist_ok=True)

    # the reference speaker is randomly picked up by the model itself for a multi-speaker TTS model,
    # so the voice of the synthetic waveforms is neither controllable nor reproducible here
    if hasattr(model, "decoder") and hasattr(model.decoder, "spk_emb"):
        print(
            "Note: your model is a multi-speaker TTS model, but no reference speaker can be given to this "
            "script. A random reference speaker is picked up by the model for each sentence, so the voice "
            "of the synthetic waveforms will be different in each run."
        )

    wav_paths = []
    for i, sentence in enumerate(sentences):
        # tokenize the raw text sentence into a token id sequence attached with <sos/eos> at both ends
        tokens = model.tokenizer.text2tensor(sentence)

        # do the generation: text (1, text_len), text_len (1,)
        outputs = model.inference(
            infer_conf,
            text=tokens.unsqueeze(0).to(device),
            text_len=torch.LongTensor([tokens.size(0)]).to(device),
        )
        # non-autoregressive TTS models return their waveforms by 'wav' while the autoregressive ones
        # return their Griffin-Lim waveforms by 'gl_wav'
        wav_key = "wav" if "wav" in outputs.keys() else "gl_wav"
        if wav_key not in outputs.keys():
            raise RuntimeError(
                f"No waveform is generated for the sentence '{sentence}'! "
                f"Please make sure that 'return_wav' (or 'return_gl_wav') is not set to False in your "
                f"infer_cfg and that the vocoder of your model works properly."
            )

        # save the synthetic waveform to the disk
        wav_path = os.path.join(output_path, f"syn_{i + 1:03d}.wav")
        sf.write(
            wav_path,
            outputs[wav_key]["content"][0],
            samplerate=outputs[wav_key]["sample_rate"],
        )
        wav_paths.append(wav_path)
        print(f"{sentence}\n  -> {wav_path}")

    # save the mapping from the input sentences to the synthetic waveforms for your reference
    with open(os.path.join(output_path, "idx2text.txt"), "w", encoding="utf-8") as f:
        for wav_path, sentence in zip(wav_paths, sentences):
            f.write(f"{os.path.splitext(os.path.basename(wav_path))[0]} {sentence}\n")
    print(f"\n{len(wav_paths)} synthetic waveforms are saved to {output_path}")
    return wav_paths


def get_tts_sentences(text: List[str], text_file: str) -> List[str]:
    """Collect the sentences to be synthesized from either --text or --text_file."""
    if text is not None:
        sentences = [sent.strip() for sent in text]
    elif text_file is not None:
        with open(resolve_input_path(text_file), encoding="utf-8") as f:
            sentences = [line.strip() for line in f.readlines()]
    else:
        sentences = []
    # the empty sentences are meaningless for TTS, so they are skipped here
    return [sent for sent in sentences if len(sent) > 0]


def main():
    args = parse()
    exp_path = resolve_input_path(args.exp_path)
    output_path = (
        os.path.abspath(args.output_path) if args.output_path is not None else None
    )

    # device initialization
    device = get_device(device=args.device, gpu=args.gpu)
    print(f"Inference device: {device}")

    # build the model from the experiment configuration and load the target checkpoint
    exp_cfg = load_exp_cfg(exp_path)
    model = build_model_from_exp(exp_cfg, exp_path, device)
    load_checkpoint(
        model, exp_path, args.test_model, device, trust_checkpoint=args.trust_checkpoint
    )
    model.eval()

    # resolve the inference (decoding) configuration
    infer_conf = resolve_infer_cfg(args.infer_cfg, exp_cfg)

    # decide the task type from the model type recorded in the experiment configuration
    model_type = exp_cfg["train_cfg"]["model"]["model_type"].lower()
    if "asr" in model_type:
        assert args.audio is not None, (
            "Your model is an ASR model. "
            "Please give the audio files you want to transcribe by --audio!"
        )
        if args.text is not None or args.text_file is not None:
            print(
                "Note: --text and --text_file are only used for TTS models, so they are ignored here."
            )
        asr_inference(model, args.audio, infer_conf, device, output_path)

    elif "tts" in model_type:
        sentences = get_tts_sentences(args.text, args.text_file)
        assert len(sentences) > 0, (
            "Your model is a TTS model. "
            "Please give the non-empty sentences you want to synthesize by --text or --text_file!"
        )
        if args.audio is not None:
            print("Note: --audio is only used for ASR models, so it is ignored here.")
        tts_inference(
            model,
            sentences,
            infer_conf,
            device,
            (
                output_path
                if output_path is not None
                else os.path.join(exp_path, "standalone_inference")
            ),
        )

    else:
        raise RuntimeError(
            f"Unknown model_type {exp_cfg['train_cfg']['model']['model_type']}! "
            "Currently, this script only supports ASR and TTS models."
        )


if __name__ == "__main__":
    main()

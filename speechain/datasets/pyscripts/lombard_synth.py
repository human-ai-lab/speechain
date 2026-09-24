"""
Author: Bagus Tris Atmaja (with Claude Code)
Affiliation: NAIST
Date: 2026.09

Synthetic Lombard speech construction for the Lombard machine speech chain
(Novitasari et al., IEEE/ACM TASLP 2022, Sec. IV-A). Following the paper,
normal speech is turned into Lombard speech by modifying its pitch,
intensity and duration with SoX, using the vocal changes observed in natural
Lombard speech. One Lombard version is generated for each noise condition.

For a dumped SpeeChain dataset (idx2wav, idx2text, idx2duration, ...) this
script produces, for every condition `cond` and subset:

    {output_path}/{cond}/{subset}/idx2wav        Lombard (or clean) waveforms
    {output_path}/{cond}/{subset}/idx2wav_len    number of samples
    {output_path}/{cond}/{subset}/idx2wav_ref    the original clean waveform
    {output_path}/{cond}/{subset}/idx2text       phoneme sequence (copied)
    {output_path}/{cond}/{subset}/idx2duration   phoneme durations (rescaled by tempo)
    {output_path}/{cond}/{subset}/idx2text_asr   raw transcript for the ASR feedback
    {output_path}/{cond}/{subset}/idx2snr_cond   the condition name

Utterance indices are suffixed by `-{cond}` so that the conditions can be
concatenated in one data_cfg. It also creates the noise waveforms used for
the simulation (`{output_path}/noise/white.wav` and `babble.wav`, the latter
being a sum of several utterances of a babble corpus, e.g. LibriSpeech
dev-clean).
"""

import argparse
import os
import subprocess
from functools import partial
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
import soundfile as sf
from tqdm import tqdm

from speechain.utilbox.data_loading_util import (
    load_idx2data_file,
    parse_path_args,
    read_data_by_path,
)

# Default vocal modifications of the Lombard effect for each noise condition:
#   gain_db: intensity increase, pitch_cents: F0 increase, tempo: speaking rate factor (<1 = slower)
DEFAULT_RULES = {
    "clean": dict(gain_db=0.0, pitch_cents=0, tempo=1.0),
    "snr0": dict(gain_db=6.0, pitch_cents=150, tempo=0.92),
    "snr-10": dict(gain_db=10.0, pitch_cents=250, tempo=0.85),
}


def parse():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--wav_path",
        type=str,
        required=True,
        help="Folder containing idx2wav of each subset.",
    )
    parser.add_argument(
        "--mfa_path",
        type=str,
        required=True,
        help="Folder containing idx2text & idx2duration of each subset.",
    )
    parser.add_argument(
        "--mfa_subdir",
        type=str,
        default="stress/no-punc",
        help="Sub-folder of mfa_path/subset holding idx2text & idx2duration.",
    )
    parser.add_argument(
        "--asr_text_file",
        type=str,
        default="idx2no-punc_text",
        help="Transcript file in wav_path/subset.",
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--subsets", type=str, nargs="+", default=["train", "valid", "test"]
    )
    parser.add_argument(
        "--conditions", type=str, nargs="+", default=list(DEFAULT_RULES.keys())
    )
    parser.add_argument(
        "--rules",
        type=str,
        default=None,
        help="Optional .yaml overriding DEFAULT_RULES.",
    )
    parser.add_argument(
        "--babble_idx2wav",
        type=str,
        default=None,
        help="idx2wav of the corpus used to build babble noise.",
    )
    parser.add_argument("--babble_speakers", type=int, default=6)
    parser.add_argument(
        "--noise_duration",
        type=float,
        default=120.0,
        help="Length (sec) of the generated noise files.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Sampling rate of the generated noise files.",
    )
    parser.add_argument("--ncpu", type=int, default=8)
    parser.add_argument(
        "--max_utts",
        type=int,
        default=None,
        help="Only process the first N utterances of each subset (for debugging).",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sox_lombard(item, output_dir: str, rule: Dict) -> (str, str, int):
    """Convert one clean waveform into its Lombard version by SoX."""
    idx, src = item
    dst = os.path.join(output_dir, os.path.basename(src))
    if not os.path.exists(dst):
        effects = []
        if rule.get("pitch_cents", 0) != 0:
            effects += ["pitch", str(rule["pitch_cents"])]
        if rule.get("tempo", 1.0) != 1.0:
            effects += ["tempo", "-s", str(rule["tempo"])]
        if rule.get("gain_db", 0.0) != 0.0:
            # the limiter (-l) avoids clipping when the speech is amplified
            effects += ["gain", "-l", str(rule["gain_db"])]
        subprocess.run(
            ["sox", src, dst] + effects, check=True, stderr=subprocess.DEVNULL
        )
    info = sf.info(dst)
    return idx, dst, info.frames


def fix_stale_paths(
    idx2wav: Dict[str, str], base_dir: str, subset: str
) -> Dict[str, str]:
    """The dumped idx2wav files may hold absolute paths of another machine. If a path doesn't
    exist, it is relocated by (1) replacing the part before `/datasets/` or `/data/` by the `data/`
    folder of the toolkit, (2) replacing the part before `/{subset}/` by `base_dir`, or (3) looking
    for the file name directly under `base_dir`."""
    data_root = parse_path_args("data")
    fixed = {}
    for idx, path in idx2wav.items():
        if not os.path.exists(path):
            candidates = []
            for marker in ["/datasets/", "/data/"]:
                if marker in path:
                    candidates.append(os.path.join(data_root, path.split(marker, 1)[1]))
            if f"/{subset}/" in path:
                candidates.append(
                    os.path.join(base_dir, path.split(f"/{subset}/", 1)[1])
                )
            candidates.append(os.path.join(base_dir, os.path.basename(path)))
            for cand in candidates:
                if os.path.exists(cand):
                    path = cand
                    break
            else:
                raise FileNotFoundError(f"Cannot locate the waveform of {idx}: {path}")
        fixed[idx] = path
    return fixed


def write_idx2data(path: str, data: Dict[str, str]):
    with open(path, "w", encoding="utf-8") as f:
        for idx, value in data.items():
            f.write(f"{idx} {value}\n")


def make_noise_files(args):
    """Generate white and babble noise waveforms."""
    noise_dir = os.path.join(args.output_path, "noise")
    os.makedirs(noise_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n_samples = int(args.noise_duration * args.sample_rate)

    white_path = os.path.join(noise_dir, "white.wav")
    if not os.path.exists(white_path):
        sf.write(
            white_path,
            (rng.standard_normal(n_samples) * 0.1).astype(np.float32),
            args.sample_rate,
        )

    babble_path = os.path.join(noise_dir, "babble.wav")
    if args.babble_idx2wav is not None and not os.path.exists(babble_path):
        import torch
        import torchaudio

        babble_file = parse_path_args(args.babble_idx2wav)
        idx2wav = fix_stale_paths(
            load_idx2data_file(babble_file),
            os.path.dirname(babble_file),
            os.path.basename(os.path.dirname(babble_file)),
        )
        paths = list(idx2wav.values())
        babble = np.zeros(n_samples, dtype=np.float32)
        for _ in range(args.babble_speakers):
            # one "speaker" stream = concatenation of random utterances
            stream, cursor = [], 0
            while cursor < n_samples:
                wav, sr = read_data_by_path(
                    paths[rng.integers(len(paths))], return_sample_rate=True
                )
                wav = wav.squeeze(-1).astype(np.float32)
                if sr != args.sample_rate:
                    wav = torchaudio.functional.resample(
                        torch.from_numpy(wav), sr, args.sample_rate
                    ).numpy()
                stream.append(wav)
                cursor += len(wav)
            stream = np.concatenate(stream)[:n_samples]
            babble += stream / (np.sqrt(np.mean(stream**2)) + 1e-8)
        babble = babble / np.max(np.abs(babble)) * 0.9
        sf.write(babble_path, babble, args.sample_rate)
    return noise_dir


def main():
    args = parse()
    rules = dict(DEFAULT_RULES)
    if args.rules is not None:
        from speechain.utilbox.yaml_util import load_yaml

        rules.update(load_yaml(open(parse_path_args(args.rules))))
    args.output_path = parse_path_args(args.output_path)
    os.makedirs(args.output_path, exist_ok=True)
    make_noise_files(args)

    for subset in args.subsets:
        wav_dir, mfa_dir = os.path.join(
            parse_path_args(args.wav_path), subset
        ), os.path.join(parse_path_args(args.mfa_path), subset, args.mfa_subdir)
        idx2wav = load_idx2data_file(os.path.join(wav_dir, "idx2wav"))
        idx2text = load_idx2data_file(os.path.join(mfa_dir, "idx2text"))
        idx2duration = load_idx2data_file(os.path.join(mfa_dir, "idx2duration"))
        idx2text_asr = load_idx2data_file(os.path.join(wav_dir, args.asr_text_file))
        idx2wav = fix_stale_paths(idx2wav, wav_dir, subset)
        indices = [
            i
            for i in idx2wav.keys()
            if i in idx2text and i in idx2duration and i in idx2text_asr
        ]
        if args.max_utts is not None:
            indices = indices[: args.max_utts]

        for cond in args.conditions:
            rule = rules[cond]
            out_dir = os.path.join(args.output_path, cond, subset)
            wav_out_dir = os.path.join(out_dir, "wav")
            os.makedirs(wav_out_dir, exist_ok=True)
            items = [(i, idx2wav[i]) for i in indices]
            if cond == "clean" or all(
                [
                    rule.get("gain_db", 0) == 0,
                    rule.get("pitch_cents", 0) == 0,
                    rule.get("tempo", 1.0) == 1.0,
                ]
            ):
                results = [
                    (i, p, sf.info(p).frames)
                    for i, p in tqdm(items, desc=f"{subset}/{cond}")
                ]
            else:
                with Pool(args.ncpu) as pool:
                    results = list(
                        tqdm(
                            pool.imap(
                                partial(sox_lombard, output_dir=wav_out_dir, rule=rule),
                                items,
                                chunksize=8,
                            ),
                            total=len(items),
                            desc=f"{subset}/{cond}",
                        )
                    )
            tempo = rule.get("tempo", 1.0)
            out = {
                k: {}
                for k in [
                    "wav",
                    "wav_len",
                    "wav_ref",
                    "text",
                    "duration",
                    "text_asr",
                    "snr_cond",
                ]
            }
            for idx, dst, n_frames in results:
                new_idx = f"{idx}-{cond}"
                out["wav"][new_idx] = dst
                out["wav_len"][new_idx] = str(n_frames)
                out["wav_ref"][new_idx] = idx2wav[idx]
                out["text"][new_idx] = idx2text[idx]
                durations = [float(d) for d in idx2duration[idx][1:-1].split(", ")]
                out["duration"][new_idx] = str([round(d / tempo, 4) for d in durations])
                out["text_asr"][new_idx] = idx2text_asr[idx]
                out["snr_cond"][new_idx] = cond
            for key, data in out.items():
                write_idx2data(os.path.join(out_dir, f"idx2{key}"), data)
            print(f"{len(results)} utterances written to {out_dir}")


if __name__ == "__main__":
    main()

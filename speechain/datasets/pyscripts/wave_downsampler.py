"""
Author: Heli Qi
Affiliation: NAIST
Date: 2022.07
"""

import argparse
import os
import shutil
import sys
from functools import partial
from multiprocessing import Pool
from typing import List

import numpy as np
import soundfile as sf
import torchaudio
from tqdm import tqdm

# Add SPEECHAIN_ROOT to the path so that this script can be executed directly
# even when the speechain package is not installed into the current environment
_SPEECHAIN_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _SPEECHAIN_ROOT not in sys.path:
    sys.path.insert(0, _SPEECHAIN_ROOT)

from speechain.utilbox.data_loading_util import (
    load_idx2data_file,
    parse_path_args,
    read_data_by_path,
)


def parse():
    parser = argparse.ArgumentParser(description="params")
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="The desired sample rate for the output waveforms. (default: 16000)",
    )
    parser.add_argument(
        "--src_file",
        type=str,
        required=True,
        help="The path to the source file containing waveform paths and their indices.",
    )
    parser.add_argument(
        "--spk_file",
        type=str,
        default=None,
        help="The path to the file containing speaker information, or None if speaker information is "
        "not used. (default: None)",
    )
    parser.add_argument(
        "--tgt_path",
        type=str,
        required=True,
        help="The target directory path where the downsampled waveform files will be saved.",
    )
    parser.add_argument(
        "--ncpu",
        type=int,
        default=8,
        help="The number of CPU cores to use for parallel processing. (default: 8)",
    )
    return parser.parse_args()


def waveform_downsample(
    idx2src_wav: List[List[str]], tgt_path: str, sample_rate: int
) -> List[List[str]]:
    """
    Downsample the input waveforms to the given sample rate and save the downsampled waveforms in the target directory.
    Note that sf.read(path) + sf.write(path, sample_rate) doesn't work because the values of waveforms remain the same.
    The only thing that is changed is the sampling rate of the waveform files.

    Args:
        idx2src_wav (List[List[str]]):
            A list of source waveform file information, where each element is a list containing the index, source
            waveform path, and speaker ID (if available).
        tgt_path (str):
            The target directory path where the downsampled waveform files will be saved.
        sample_rate (int):
            The desired sample rate for the output waveforms.

    Returns:
        List[List[str]]:
            A list of target waveform file information, where each element is a list containing the index and target
            waveform path.
    """
    idx2tgt_wav, resamplers = [], {}
    # loop each source data wav in the given chunk
    for idx, src_wav_path, spk in tqdm(idx2src_wav):
        file_name = src_wav_path.split("/")[-1]
        if spk is not None:
            os.makedirs(os.path.join(tgt_path, spk), exist_ok=True)
            tgt_wav_path = os.path.join(tgt_path, spk, file_name)
        else:
            os.makedirs(os.path.join(tgt_path, "wav"), exist_ok=True)
            tgt_wav_path = os.path.join(tgt_path, "wav", file_name)

        # create the downsampled waveform file
        if not os.path.exists(tgt_wav_path):
            # read the sampling rate from the file header without loading the whole waveform
            src_sample_rate = sf.info(src_wav_path).samplerate
            # resample and rewrite the waveform only when its sampling rate is different from the target one
            if src_sample_rate != sample_rate:
                src_wav = read_data_by_path(src_wav_path, return_tensor=True)
                if src_sample_rate not in resamplers.keys():
                    resamplers[src_sample_rate] = torchaudio.transforms.Resample(
                        orig_freq=src_sample_rate, new_freq=sample_rate
                    )
                src_wav = resamplers[src_sample_rate](src_wav.squeeze(-1))

                wav_format = file_name.split(".")[-1].upper()
                sf.write(
                    file=tgt_wav_path,
                    data=src_wav,
                    samplerate=sample_rate,
                    format=wav_format,
                    subtype=sf.default_subtype(wav_format),
                )
            # if the source waveform already has the target sampling rate, link the source file to the target path
            # instead of writing a duplicated copy to avoid wasting the disk space
            else:
                try:
                    # hard link: no extra disk space is occupied and the target file remains valid
                    # even if the source file is deleted afterwards
                    os.link(src_wav_path, tgt_wav_path)
                except OSError:
                    try:
                        # symbolic link: used when the hard link is unavailable
                        # (e.g., the source and target paths are on different file systems)
                        os.symlink(os.path.abspath(src_wav_path), tgt_wav_path)
                    except OSError:
                        # normal copy as the last resort
                        shutil.copyfile(src_wav_path, tgt_wav_path)
        # record the target waveform path
        idx2tgt_wav.append([idx, tgt_wav_path])

    return idx2tgt_wav


def main(
    src_file: str, spk_file: str, tgt_path: str, sample_rate: int = 16000, ncpu: int = 8
):
    """
    Downsample the waveforms specified in the source file to the desired sample rate, and save the downsampled
    waveforms in the target directory. The function also copies the statistic information from the source directory
    to the target directory.

    Args:
        src_file (str):
            The path to the source file containing waveform paths and their indices.
        spk_file (str):
            The path to the file containing speaker information, or None if speaker information is not used.
        tgt_path (str):
            The target directory path where the downsampled waveform files will be saved.
        sample_rate (int):
            The desired sample rate for the output waveforms.
        ncpu (int):
            The number of CPU cores to use for parallel processing.

    Returns:
        None
    """
    src_file, tgt_path = parse_path_args(src_file), parse_path_args(tgt_path)
    if spk_file is not None:
        spk_file = parse_path_args(spk_file)
    os.makedirs(tgt_path, exist_ok=True)

    # skip the dowmsampling process if there has already been an idx2wav
    idx2tgt_wav_path = os.path.join(tgt_path, "idx2wav")
    if not os.path.exists(idx2tgt_wav_path):
        # reshape the source waveform paths into individual chunks by the given chunk_size
        idx2src_wav = load_idx2data_file(src_file)
        idx2spk = load_idx2data_file(spk_file) if spk_file is not None else None
        idx2src_wav = [
            [idx, idx2src_wav[idx], idx2spk[idx] if idx2spk is not None else None]
            for idx in idx2src_wav.keys()
        ]
        func_args = [idx2src_wav[i::ncpu] for i in range(ncpu)]

        # saving the downsampled audio files to the disk
        with Pool(ncpu) as executor:
            waveform_downsample_func = partial(
                waveform_downsample, tgt_path=tgt_path, sample_rate=sample_rate
            )
            idx2tgt_wav_list_nproc = executor.map(waveform_downsample_func, func_args)

        idx2tgt_wav = []
        for idx2tgt_wav_list in idx2tgt_wav_list_nproc:
            idx2tgt_wav += idx2tgt_wav_list
        np.savetxt(idx2tgt_wav_path, sorted(idx2tgt_wav, key=lambda x: x[0]), fmt="%s")
    else:
        print(
            f"Downsampled waveforms have already existed in {tgt_path}, so the dowmsampling process is skipped."
        )

    print(
        f"Copying statistic information from {os.path.dirname(src_file)} to {tgt_path}"
    )
    src_dir = os.path.dirname(src_file)
    for file in os.listdir(src_dir):
        # skip idx2wav, folders, and the files with a suffix
        if (
            file in ["idx2wav", "idx2wav_len", "idx2feat", "idx2feat_len"]
            or os.path.isdir(os.path.join(src_dir, file))
            or "." in file
        ):
            continue
        shutil.copy(os.path.join(src_dir, file), tgt_path)

    print("\n")


if __name__ == "__main__":
    args = parse()
    main(**vars(args))

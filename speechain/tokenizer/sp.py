"""
Author: Heli Qi
Affiliation: NAIST
Date: 2022.07
"""

import filecmp
import os
import shutil
import warnings
from typing import List

import sentencepiece as spm
import torch

from speechain.tokenizer.abs import Tokenizer
from speechain.utilbox.import_util import parse_path_args


class SentencePieceTokenizer(Tokenizer):
    """Tokenizer implementation that converts the input sentence string into subword
    tokens, i.e., combinations of graphemes, by the sentencepiece package.

    References: https://github.com/google/sentencepiece
    """

    def tokenizer_init_fn(self, token_path: str, copy_path: str = None, **kwargs):
        """Initialize the sentencepiece tokenizer model.

        Args:
            copy_path: str = None
                The path where you want to paste the given tokenizer model as a backup.
                If not given, no backup will be saved.
            token_path: str
                The path of your specified sentencepiece tokenizer model file.
                If not given, the model will automatically selected in the same folder as the given token_vocab
        """
        # The model in token_path token_model has the highest priority for token_model initialization
        if token_path is not None:
            token_model = os.path.join(parse_path_args(token_path), "model")

        # if token_path is not given or model does not exist, use the backup on in copy_path
        if token_path is None or not os.path.exists(token_model):
            assert (
                copy_path is not None
            ), "Please give copy_path for SentencePiece model backup!"
            token_model = os.path.join(parse_path_args(copy_path), "token_model")

        # initialize the tokenizer model by the sentencepiece package
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(token_model)

        # save the backup if copy_path is given. the backup is only (re)made when it is absent or outdated
        # so that the jobs which merely read the tokenizer model (e.g., standalone inference) don't keep
        # rewriting it and are not interrupted by a read-only or shared copy_path.
        if copy_path is not None:
            backup_model = os.path.join(parse_path_args(copy_path), "token_model")
            if not os.path.exists(backup_model) or not filecmp.cmp(
                token_model, backup_model, shallow=False
            ):
                try:
                    shutil.copy(src=token_model, dst=backup_model)
                # SameFileError is a subclass of OSError, so it is covered here as well
                except OSError as save_error:
                    warnings.warn(
                        f"The backup of the sentencepiece model cannot be saved into {copy_path} "
                        f"({save_error}), but the ongoing job is not affected."
                    )

    def tensor2text(self, tensor: torch.LongTensor or List):
        """

        Args:
            tensor:

        Returns:

        """
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.tolist()
        text = self.sp_model.decode_ids(
            [t for t in tensor if t not in [self.sos_eos_idx, self.ignore_idx]]
        )
        return text

    def text2tensor(
        self,
        text: str,
        no_sos: bool = False,
        no_eos: bool = False,
        return_tensor: bool = True,
    ):
        """

        Args:
            text:
            no_sos:
            no_eos:
            return_tensor:

        Returns:

        """
        # initialize the tensor as an empty list
        tokens = []
        # whether to attach sos at the beginning of the tokens
        if not no_sos:
            tokens.append(self.sos_eos_idx)
        # attach the main body of the text
        tokens.extend(self.sp_model.encode_as_ids(text))
        # whether to attach eos at the end of the tokens
        if not no_eos:
            tokens.append(self.sos_eos_idx)
        # turn the token list into a long-type tensor
        if return_tensor:
            return torch.LongTensor(tokens)
        else:
            return tokens

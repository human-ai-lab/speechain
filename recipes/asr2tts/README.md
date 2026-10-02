# ASR→TTS chain (reconstruction plan — UNTESTED)

The original configs/data for this experiment were lost in a crash. This file records what is
known from the paper and the surviving files, and the stages needed to rebuild it.
Nothing here has been run.

## Evidence for the data setup
| Item | Count | Source |
|---|---|---|
| LibriTTS train-clean-100 (labeled TTS data) | 33,236 | `data/libritts/data/wav16000/train-clean-100/idx2wav` |
| LibriSpeech train-clean-360 (unlabeled speech) | 104,014 | `data/librispeech/data/wav/train-clean-360/idx2wav` |
| LibriTTS train-clean-360 | 116,500 | not used (paper: 137K total = 33K + 104K) |
| LibriTTS test-clean | 4,837 | surviving baseline `test.log` (3864 done + 973 remaining) |

## Surviving artifacts
- Baseline TTS: `recipes/tts/libritts/train-clean-100/exp/16khz_ecapa_mfa_fastspeech2_punc/`
  (checkpoint, `exp_cfg.yaml`; FastSpeech2, 4 layers, d_model 384, 2 heads, ff 1536, 10k warm-up,
  500 epochs, MFA `english_us_arpa`, `punc` text). Its test run was interrupted.
- NOT surviving: the paper's Conformer pseudo-labeling ASR (1k BPE, no CTC), the pseudo-labels,
  MFA alignments for LS-360, and the chain FastSpeech2 config/checkpoint.

## Stages
1. **BPE-1k vocabulary** on LS-100 text: `speechain/datasets/pyscripts/vocab_generator.py`
   (target `data/librispeech/data/sentencepiece/train-clean-100/bpe1k/no-punc`).
2. **Base ASR**: `recipes/asr/librispeech/train-clean-100/exp_cfg/100-bpe1k_conformer-base_noctc.yaml`
   (reconstructed; hyperparameters beyond the paper are inherited from the toolkit recipe).
3. **Pseudo-label LS-360**: decode `train-clean-360` with the base ASR (beam 16, no CTC); write
   the hypotheses as a new `idx2no-punc_text`-style file.
4. **Align**: install Montreal Forced Aligner (not installed here) and run
   `data/mfa_preparation.sh` with the `english_us_arpa` model/lexicon on the pseudo text,
   matching the baseline's token vocabulary (`.../mfa/acoustic=english_us_arpa_lexicon=english_us_arpa/train-clean-100/stress/punc`).
5. **ECAPA speaker embeddings** for LS-360 speakers (same extractor as LT-100).
6. **Merge** LT-100 + pseudo-labeled LS-360 into one train set (137,250 utterances) and copy
   `exp_cfg/16khz_ecapa_mfa_fastspeech2_punc.yaml` with `train_set` changed.
7. **Evaluate** baseline and chain on LibriTTS test-clean (4,837 utts): MCD, MSD, log-F0 RMSE,
   duration ratio.

Open questions the paper cannot answer: whether the chain used 300 or 500 epochs, and whether
pseudo-label filtering was applied (the paper says none).

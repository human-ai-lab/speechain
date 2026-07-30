# Albanian ASR with Speechain

This recipe contains a Speechain-based ASR setup for low-resource Albanian speech recognition.

## Goal

The goal is to build and improve a from-scratch ASR baseline for Albanian using Speechain.

The current setup focuses on:
- character-level ASR
- external language model decoding
- CTC/LM weight tuning

## Dataset

The experiments are based on Mozilla Common Voice Albanian.

Prepared splits:
- Train: 2,633 utterances
- Dev: 1,795 utterances
- Test: 1,908 utterances

The dataset itself is not included in this repository.

Recommended data root directory:
data/albanian_asr

## ASR Model

Final ASR setup:
- Model: Speechain ARASR
- Encoder: Conformer
- Decoder: Transformer
- Tokenizer: character-level tokenizer
- Output vocabulary: 34 tokens
- Parameters: approximately 4.74M

Main ASR config:
exp_cfg/cv_sq_full_char_sp_ctc05_do02_acc2_e100.yaml

## Training Setup

Improved ASR training configuration:
- max_epochs: 100
- early_stopping_patience: 15
- stopped_epoch: 75
- batch_len: 3.0e5
- accum_grad: 2
- training_ctc_weight: 0.5
- label_smoothing: 0.05
- dropout: 0.2
- warmup_steps: 2000
- speed_perturbation: enabled

Important note:
The training CTC weight was 0.5.
The CTC weights 0.3 and 0.4 were only used during offline decoding sweeps.

## External Language Model

The external LM was a character-level Transformer LM trained from scratch in Speechain.

LM experiment name:
wiki_char_transformer

LM recipe:
recipes/lm/albanian/wiki_char_lm_text

The LM was trained using cleaned Albanian Wikipedia text and Common Voice training transcripts.

The LM was used only during beam-search decoding, not during ASR training.

## Final Results

### Key result

The best WER was achieved with the final CTC/LM decoding sweep.

* Best WER setup: `ctc_weight = 0.3`, `lm_weight = 1.4`
* CER: 35.36%
* WER: 63.82%

### Performance progression

| Step | System                              |    CER |    WER |
| ---: | ----------------------------------- | -----: | -----: |
|    1 | First ASR baseline without LM       | 46.51% | 92.79% |
|    2 | First ASR baseline with external LM | 43.60% | 83.92% |
|    3 | Improved ASR with external LM       | 35.14% | 64.40% |
|    4 | Final CTC/LM sweep                  | 35.36% | 63.82% |

### Overall improvement

Compared to the first ASR baseline without LM:

* CER improved from 46.51% to 35.36%
* CER reduction: 11.15 percentage points
* WER improved from 92.79% to 63.82%
* WER reduction: 28.97 percentage points

### Best decoding configurations

| Objective | ctc_weight | lm_weight |    CER |    WER |
| --------- | ---------: | --------: | -----: | -----: |
| Best WER  |        0.3 |       1.4 | 35.36% | 63.82% |
| Best CER  |        0.4 |       1.2 | 34.90% | 64.85% |


## Qualitative Example

Reference:
ky fshat është i izoluar nga bota është i harruar

Hypothesis:
ky fshat dhe shitën guara porta është i hartuar

Utterance-level CER: 35.00%
Utterance-level WER: 60.00%

This example illustrates that the model can recognize parts of the utterance correctly, but still produces plausible yet incorrect Albanian word sequences during LM-based decoding.

## Notes

The model is trained from scratch and remains limited by the small amount of available Albanian speech data.

Future work should compare this Speechain baseline against pretrained multilingual ASR models such as Whisper, MMS, and XLS-R.

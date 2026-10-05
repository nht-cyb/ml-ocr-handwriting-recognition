# Text Recognition - an unofficial implementation of the paper [MaskOCR: Text Recognition with Masked Encoder-Decoder Pretraining](https://arxiv.org/pdf/2206.00311.pdf)

This is an implementation for the MaskOCR paper on recognizing Vietnamese handwritings.
# Architecture
![MaskOCR architecture](https://i.ibb.co/bKnSjKX/Mask-OCR-architecture.png)
# Files

Using MaskOCR_handwriting_recognition.ipynb to experiment.

# Pipelines

We implement 3 pipelines of MaskOCR model as follows:
- Pretraining pipeline for the Encoder.
- Pretraining pipeline for the Decoder.
- Main training pipeline for the MaskOCR model.

All pipelines have the same pre-steps as running these cells:
- Installing dependencies.
- Data collections. (Download IAM & Vietnamese handwritings datasets)
- Data loader helpers.
- Base structures.

## Encoder pretraining pipeline

Run **Pretraining pipeline for encoder** and **Train** cells in **MaskOCR Encoder** cell to start encoder pretraining phase. The best epoch is saved as `EncoderModel_<timestamp>_<epoch>.pth`.

## Decoder pretraining pipeline

Set `pretrain_encoder_path` to the encoder checkpoint, then run **Pretraining pipeline for decoder** and **Train** cells in **MaskOCR Decoder** cell to start decoder pretraining phase. The encoder is frozen in this phase; the best epoch is saved as `DecoderModel_<timestamp>_<epoch>.pth`.

## Main MaskOCR training pipeline

Set `pretrain_model_path` to the decoder checkpoint (or only `pretrain_encoder_path` to skip decoder pretraining), then run **Training pipeline for MaskOCR** and **Train** cells in **Train MaskOCR** cell to start the main training phase. Encoder and decoder are trained together; validation CER is printed every epoch, the best epoch is saved as `MaskOCR_<timestamp>_<epoch>.pth`, and its CER on the held-out test split is printed at the end.

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

Run **Pretraining pipeline for encoder** and **Train** cells in **MaskOCR Encoder** cell to start encoder pretraining phase.

## Decoder pretraining pipeline

Run **Pretraining pipeline for decoder** and **Train** cells in **MaskOCR Decoder** cell to start decoder pretraining phase.

## Main MaskOCR training pipeline

Run **Training pipeline for MaskOCR** and **Train** cells in **Train MaskOCR** cell to start decoder pretraining phase.

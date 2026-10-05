# Vietnamese Handwriting Recognition (OCR)

Machine Learning course project (Group 15, VNU-UET) — recognising Vietnamese handwritten words with Transformer-based encoder–decoder models.

**Authors:** Nguyễn Huyền Trang (20020727), Lê Thị Trang (20020726), Nguyễn Thành Quốc (20020707)

Full write-up: [`Reports/ML_FinalReport_15.pdf`](Reports/ML_FinalReport_15.pdf) · Slides: [`Reports/ML_Presentation_15.pdf`](Reports/ML_Presentation_15.pdf)

## Task

- **Input:** an image containing a single handwritten Vietnamese word.
- **Output:** the recognised text.
- **Approach:** image → **encoder** (feature map) → **decoder** (character sequence).

Vietnamese is hard for OCR: 29 letters (10 carrying tone marks), many syllables and diacritics, individual handwriting styles, and low-quality photos.

## Dataset

[SoICT Hackathon 2023 – Vietnamese Handwritten Text Recognition](https://aihub.vn/competitions/426#participate)

| Split | Images | Notes |
|---|---|---|
| Training | 103,000 | 51k form + 48k wild + 4k GAN, labelled |
| Public test | 33,000 | 17k form + 16k wild, unlabelled |

Labels are `.txt` files, one line per image: `IMAGE_NAME<TAB>GROUND_TRUTH_TEXT`.
Image sizes: height 11–378 px (mean 72), width 0–543 px (mean 131). Most words are 3–4 characters long.

## Repository layout

```
.
├── CRNN/                    # CNN + BiLSTM + CTC baseline (scripts)
├── TrOCR/                   # Fine-tuned microsoft/trocr-base-handwritten (Kaggle notebooks)
├── MaskOCR/                 # MaskOCR re-implementation (Colab notebook)
├── demo.py                  # Streamlit web demo
└── Reports/                 # Final report + presentation (PDF)
```

## Results

Metric: **Character Error Rate** — `CER = (S + D + I) / N` (substitutions, deletions, insertions over reference length; lower is better). Only character-level error is reported because each sample is a single word.

| Model | CER | Accuracy ≈ 1 − CER |
|---|---|---|
| TrOCR (fine-tuned) | 0.1164 | ~89% |
| CRNN | 0.0899 | ~91% |
| MaskOCR | 0.1378 | ~86% |

Both reported models fall in the "average OCR quality" band (CER 2–10%). MaskOCR, with masked encoder–decoder pretraining, beats plain TrOCR fine-tuning by ~3.7 CER points. TrOCR was evaluated with a 95:5 train/validation split (97,850 / 5,150 images).

**Limitations noted in the report:**
- Small training images make the models sensitive to small input changes.
- Blurry, noisy or poorly lit images are handled poorly.
- Limited compute meant the Transformer models had not fully converged.

## How to run each model

> All scripts contain **hard-coded paths** from the authors' machines (`/kaggle/...`, `/content/...`, `/home/tienvh/...`, `D:\...`). Edit them to point to your local dataset and checkpoints before running.

### 1. MaskOCR — `MaskOCR/`

Unofficial implementation of [MaskOCR: Text Recognition with Masked Encoder-Decoder Pretraining](https://arxiv.org/abs/2206.00311). ViT encoder + DETR-style decoder with character queries.

Open `MaskOCR_handwriting_recognition.ipynb` in Google Colab (GPU) and run the shared setup cells first: *Install dependencies → Data collection (IAM + Vietnamese) → Data loader helpers → Base structures*. Then run the three phases in order:

1. **Encoder pretraining** — run *Pretraining pipeline for encoder* + *Train* under **MaskOCR Encoder**. Self-supervised: random vertical patches are masked and the encoder learns to predict their features and pixels (lr 1.5e-4, batch 80, 5 epochs).
2. **Decoder pretraining** — run *Pretraining pipeline for decoder* + *Train* under **MaskOCR Decoder**. Encoder is frozen; characters/patches are masked so the decoder learns a language model (batch 256, 5 epochs). Set `pretrain_encoder_path` to the step 1 output.
3. **Main training** — run *Training pipeline for MaskOCR* + *Train* under **Train MaskOCR** (20 epochs). Set `pretrain_encoder_path` and `pretrain_model_path` to the outputs of steps 1 and 2.

Checkpoints are saved under `saved_models/`; losses are logged to TensorBoard.

### 2. TrOCR — `TrOCR/`

Fine-tunes [`microsoft/trocr-base-handwritten`](https://huggingface.co/microsoft/trocr-base-handwritten) (ViT/BEiT encoder + RoBERTa decoder) with Hugging Face `VisionEncoderDecoderModel`.

- **Checkpoint:** <https://www.kaggle.com/datasets/loinh1106/ckpt4000>
- **Dependencies:** `pip install "transformers[torch]" accelerate datasets jiwer gdown`

**Train:** run `train_trocr.ipynb` on Kaggle. It downloads `train_gt.txt` with `gdown`, builds the dataset from `new_train/` (`max_target_length=128`, generation `max_length=64`, batch 8, save every 1000 steps) and reports CER on the validation split.

**Test:** run `test_ocr.ipynb`. It loads the checkpoint (`/kaggle/input/ckpt4000/ckpt-4000`), predicts on `new_public_test/` and writes `out.csv`.

### 3. CRNN — `CRNN/`

Baseline [CRNN](https://arxiv.org/abs/1507.05717) (CNN + BiLSTM, CTC loss, Adadelta), based on [crnn-pytorch](https://github.com/GitYCC/crnn-pytorch). Character set: `charset.txt`.

```bash
cd CRNN
pip install -r requirements.txt   # contains conflicting torchvision pins; keep one matching your torch/CUDA
```

**Train:** set `img_dir` and `gt_path` in `train.py` (input 64×128, batch 256, 100 epochs; checkpoints in `checkpoints/<exp_name>/`), then:
```bash
python train.py
```

**Predict:** set `img_dir`, `exp_name` and the image size in `predict.py`, then:
```bash
python predict.py   # → result/prediction.txt
```
Note: `train.py` trains at 64×128 (`crnn_64_128`), while `predict.py` loads `crnn_32_256` at 32×256. Make the size and experiment name match the checkpoint you use.

## Web demo

`demo.py` is a Streamlit app: upload a JPG/PNG and it shows the predicted text. It uses the Handwritten_OCR (VGG-Transformer) model.

```bash
pip install streamlit
# set config['weights'] in demo.py to your downloaded checkpoint
PYTHONPATH=Handwritten_OCR-main streamlit run demo.py
```
(`PYTHONPATH` is needed so that `tool.predictor` / `tool.config` can be imported.)

## References

- Vaswani et al., *Attention Is All You Need*, 2017
- Li et al., *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models*, AAAI 2023
- Lyu et al., *MaskOCR: Text Recognition with Masked Encoder-Decoder Pretraining*, arXiv:2206.00311
- Bao et al., *BEiT: BERT Pre-Training of Image Transformers*, 2021
- Shi et al., *An End-to-End Trainable Neural Network for Image-based Sequence Recognition*, 2015

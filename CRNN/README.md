# CRNN
## Overview
This repo implemented CRNN for handwritten recognition. For detail, please refer to this paper [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717).

## Install requirements
```command
pip install -r requirements.txt
```

## Dataset
Download at [SoICT Hackathon 2023 - Vietnamese Handwritten Text Recognition](https://aihub.vn/competitions/426#participate)

## Train
Change dataset path in file train.py (img_dir and gt_path).
``` command
python train.py
```

## Inference
Change dataset path in file predict.py (img_dir).
```command
python predict.py
```

## Acknowledgement
This code is based on this repo: [crnn-pytorch](https://github.com/GitYCC/crnn-pytorch)


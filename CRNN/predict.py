import numpy as np
import os


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim

from dataset_loader import ImageDataset, read_gt
from models.crnn import CRNN
from dataset_loader import OCRDataLoader

from utils import get_charset, ctc_decode, decode_char, cer


def predict(dataloader, model, device):
    model.eval()
    all_preds = []
    data_size = len(dataloader)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images = data.to(device)
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=2)
            preds = ctc_decode(log_probs, dataloader.charset, decode_char)
            all_preds += preds

            print(f'Predicted batch {i + 1}/{data_size}.')


    return all_preds

def export_prediction(output_dir, img_names, preds):
    filename = os.path.join(output_dir, 'prediction.txt')
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        for img_name, pred in zip(img_names, preds):
            pred = "".join(pred)
            if len(pred) == 0:
                pred = '-'
            f.write(f'{img_name}\t{pred}\n')

def main():
    img_dir = "Datasets/SoICT2023/OCR/new_public_test/"
    # optional: a label file for img_dir (same format as train_gt.txt) to compute CER
    gt_path = None
    exp_name = "crnn_32_128"

    checkpoint_dir = os.path.join("checkpoints", exp_name)
    checkpoint_path = os.path.join(checkpoint_dir, 'model_final.pt')

    output_dir = 'result'

    # must match the size used in train.py
    img_height = 32
    img_width = 128
    img_channel = 3

    img_names = os.listdir(img_dir)
    dataloader = OCRDataLoader(
        img_dir,
        img_names,
        img_w=img_width,
        img_h=img_height,
        batch_size=256,
        num_workers=min(8, os.cpu_count() or 1),
        shuffle=False,
    )

    charset = dataloader.charset
    num_class = len(charset) + 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    crnn = CRNN(img_height, img_width, img_channel, num_class)
    print('Loading checkpoint:', checkpoint_path)
    crnn.load_state_dict(torch.load(checkpoint_path, map_location=device))
    crnn.to(device)

    preds = predict(dataloader, crnn, device)

    export_prediction(output_dir, img_names, preds)

    if gt_path:
        gt_dict = read_gt(gt_path)
        pairs = [("".join(p), gt_dict[n]) for n, p in zip(img_names, preds) if n in gt_dict]
        print(f'CER on {len(pairs)} labelled images: {cer(*zip(*pairs)):.4f}')

if __name__ == "__main__":
    main()

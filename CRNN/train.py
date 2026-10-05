import numpy as np
import os

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset_loader import ImageDataset
from models.crnn import CRNN
from dataset_loader import OCRDataLoader
from utils import get_charset


def forward_batch(model, data, criterion, optimizer, device, mode='train'):
    images, targets, target_lengths = [d.to(device) for d in data]
    
    logits = model(images)
    log_probs = F.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    loss = criterion(log_probs, targets, input_lengths, target_lengths)
    
    if mode == 'train':
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss


def main():
    img_dir = "Datasets/SoICT2023/OCR/new_train/"
    gt_path = "Datasets/SoICT2023/OCR/train_gt.txt"
    exp_name = "crnn_64_128"
    
    checkpoint_dir = os.path.join("checkpoints", exp_name)
    log_writer = SummaryWriter(checkpoint_dir)
    # os.makedirs(checkpoint_dir, exist_ok=True)
    
    IMG_H = 64
    IMG_W = 128
    IMG_C = 3

    img_names = os.listdir(img_dir)
    train_image_names, val_image_names = train_test_split(img_names, test_size=0.2)
    print('Number of train samples:', len(train_image_names))
    print('Number of validation samples:', len(val_image_names))
    
    train_loader = OCRDataLoader(
        img_dir,
        train_image_names,
        gt_path,
        batch_size=256,
        num_workers=32,
        shuffle=True,
    )
    val_loader = OCRDataLoader(
        img_dir,
        val_image_names,
        gt_path,
        batch_size=256,
        num_workers=32,
        shuffle=False,
    )

    charset = get_charset()
    num_class = len(charset) + 1

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    model = CRNN(IMG_H, IMG_W, IMG_C, num_class)
    model.to(device)
    
    criterion = nn.CTCLoss(reduction="sum")
    criterion.to(device)
    
    optimizer = optim.Adadelta(model.parameters())
    
    show_interval = 20
    save_interval = 5000

    num_epoch = 100
    i = 1
    for epoch in range(1, num_epoch + 1):
        total_loss = 0
        total_size = 0

        model.train()
        for data in train_loader:
            loss = forward_batch(model, data, criterion, optimizer, device)
            batch_size = data[0].size(0)

            total_loss += loss
            total_size += batch_size

            if i % show_interval == 0:
                print(f"Epoch {epoch} - iter {i} - training loss: {total_loss / total_size}")
                log_writer.add_scalar('ctc loss', total_loss / total_size, i)

            if i % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter{i}.pt")
                torch.save(model.state_dict(), checkpoint_path)
            
            i += 1
            
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_size = 0
            for data in val_loader:
                loss = forward_batch(model, data, criterion, optimizer, device, mode='val')
                batch_size = data[0].size(0)

                total_loss += loss
                total_size += batch_size

            print(f"Epoch {epoch} - iter {i} - validation loss: {total_loss / total_size}")
            log_writer.add_scalar('ctc validation loss', total_loss / total_size, i)

    checkpoint_path = os.path.join(checkpoint_dir, f"model_final.pt")
    torch.save(model.state_dict(), checkpoint_path)
    
if __name__ == "__main__":
    main()

import torch
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import ToTensor, Normalize
import numpy as np
from utils import CharEncoder, get_charset

import os


class ImageDataset(data.Dataset):
    def __init__(
        self, img_dir, img_names, gt_path, transform=None, target_transform=None
    ):
        self.img_dir = img_dir
        self.img_names = img_names
        self.transform = transform

        self.is_test = False if gt_path else True
        if not self.is_test:
            self.target_transform = target_transform
            self.gt_dict = {}
            with open(gt_path) as f:
                for line in f:
                    file_name, label = line.split()
                    self.gt_dict[file_name] = label

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img
        else:
            label = self.gt_dict[self.img_names[idx]]
            if self.target_transform:
                label, label_len = self.target_transform(label)

            return img, label, label_len


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class OCRDataLoader(data.DataLoader):
    def __init__(
        self,
        img_dir,
        img_names,
        gt_path=None,
        img_w=128,
        img_h=32,
        batch_size=4,
        num_workers=4,
        shuffle=True,
    ):
        self.img_w = img_w
        self.img_h = img_h
        self.is_test = False if gt_path else True
        self.charset = get_charset()

        dataset = ImageDataset(
            img_dir,
            img_names,
            gt_path,
            transform=self._transform,
            target_transform=self._encode_char,
        )
    
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch, keep_ratio=False):
        if self.is_test:
            images = batch
        else:
            images, targets, target_lengths = zip(*batch)

        # img_w = self.img_w
        # img_h = self.img_h

        # if keep_ratio:
        #     ratios = []
        #     for img in images:
        #         w, h = img.size
        #         ratios.append(1.0 * w / h)
        #     ratios.sort()
        #     max_ratio = ratios[-1]
        #     img_w = int(max_ratio * img_h)
        #     img_w = max(img_h, img_w)

        # images = [self._transform(img, img_w, img_h) for img in images]
        images = torch.cat([img.unsqueeze(0) for img in images], 0)
        if not self.is_test:
            targets = torch.cat(targets, 0)
            target_lengths = torch.cat(target_lengths, 0)
            return images, targets, target_lengths
        else:
            return images

        # return images, targets, target_lengths

    def _transform(self, img):
        img = img.resize((self.img_w, self.img_h), Image.BILINEAR)
        img = ToTensor()(img)
        img = Normalize(mean, std)(img)

        return img

    def _encode_char(self, text):
        char_code = {}
        char_code[" "] = 0
        for i, char in enumerate(self.charset):
            char_code[char] = i + 1
        encoded_text = [char_code[c] for c in text]

        return torch.LongTensor(encoded_text), torch.LongTensor([len(encoded_text)])

# ============================================================
# dataset.py — Polyp Segmentation Project 
# ============================================================

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PolypDataset(Dataset):

    def __init__(self, img_ids, img_dir, mask_dir, img_ext, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]

        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        mask_path = os.path.join(self.mask_dir, img_id + self.img_ext)

        # -----------------------------
        # Load image
        # -----------------------------
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # -----------------------------
        # Load mask
        # -----------------------------
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Convert mask to binary (IMPORTANT)
        mask = (mask > 127).astype(np.float32)

        # -----------------------------
        # Augmentation
        # -----------------------------
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # -----------------------------
        # Normalize image
        # -----------------------------
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        # -----------------------------
        # Convert mask
        # -----------------------------
        mask = mask.astype(np.float32)

        # Ensure shape (1, H, W)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)
        else:
            mask = mask.transpose(2, 0, 1)

        mask = torch.from_numpy(mask)

        return image, mask, img_id
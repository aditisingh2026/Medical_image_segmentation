# ============================================================
# evaluate_kvasir_seg.py
# ============================================================
# Evaluate your ALREADY-TRAINED UNet++ and DeepLabV3+ models on
# Kvasir-SEG -- an INDEPENDENT dataset they have never seen.
#
# This is INFERENCE ONLY. No training happens here. Both models
# are loaded from their saved .pth checkpoints and run on Kvasir-SEG
# images to see how well they generalize beyond your own dataset.
#
# HOW TO USE IN COLAB:
#   1. Make sure your trained models are accessible:
#        models/best_model_unet++.pth
#        models/best_model_deeplabV3+.pth
#   2. Run the download cell below ONCE to get Kvasir-SEG (needs
#      a free Kaggle account + API token -- instructions in the
#      download cell's comments).
#   3. Run the rest of this script as-is.
# ============================================================

# ---------------- CELL 1: Install + Download Kvasir-SEG ----------------
# Run this cell first. You need a free Kaggle account.
#
# Steps to get your kaggle.json:
#   a) Go to kaggle.com -> click your profile picture -> Settings
#   b) Scroll to "API" section -> click "Create New Token"
#   c) This downloads kaggle.json to your computer
#   d) In Colab, run the upload prompt below and select that file
#
# !pip install kaggle --quiet
#
# from google.colab import files
# print("Upload your kaggle.json file:")
# files.upload()   # select kaggle.json when prompted
#
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
#
# !kaggle datasets download -d debeshjha1/kvasirseg
# !unzip -q kvasirseg.zip -d kvasir_seg_raw
#
# After unzipping, check the folder structure with:
# !find kvasir_seg_raw -maxdepth 3 -type d
#
# It will typically look like:
#   kvasir_seg_raw/Kvasir-SEG/Kvasir-SEG/images/*.jpg
#   kvasir_seg_raw/Kvasir-SEG/Kvasir-SEG/masks/*.jpg
#
# If your folder names differ, just update KVASIR_IMG_DIR and
# KVASIR_MASK_DIR below to match.


# ---------------- CELL 2: Evaluation script ----------------
import os
import glob
import cv2
import numpy as np
import torch
import albumentations as A
from tqdm import tqdm

from source.model import UNetPP
from source.dataset import PolypDataset
from source.utils import iou_score, dice_score, AverageMeter

import segmentation_models_pytorch as smp

# ---------------- CONFIG: update these paths if your folder names differ ----------------
KVASIR_IMG_DIR = "kvasir_seg_raw/Kvasir-SEG/Kvasir-SEG/images"
KVASIR_MASK_DIR = "kvasir_seg_raw/Kvasir-SEG/Kvasir-SEG/masks"
KVASIR_IMG_EXT = ".jpg"

IMAGE_SIZE = 384      # SAME resolution your models were trained on
BATCH_SIZE = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- Build the list of Kvasir-SEG image IDs ----------------
kvasir_img_paths = glob.glob(os.path.join(KVASIR_IMG_DIR, "*" + KVASIR_IMG_EXT))
kvasir_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in kvasir_img_paths]
print(f"Found {len(kvasir_img_ids)} Kvasir-SEG images")

if len(kvasir_img_ids) == 0:
    raise RuntimeError(
        "No images found! Check KVASIR_IMG_DIR / KVASIR_IMG_EXT match your "
        "actual unzipped folder structure (run the !find command in Cell 1)."
    )

# Optional: evaluate on a random subset for speed (e.g. 200 images) instead
# of all 1000. Comment this out to use the full dataset.
SUBSET_SIZE = 200   # set to None to use all images
if SUBSET_SIZE is not None and len(kvasir_img_ids) > SUBSET_SIZE:
    import random
    random.seed(42)
    kvasir_img_ids = random.sample(kvasir_img_ids, SUBSET_SIZE)
    print(f"Using a random subset of {SUBSET_SIZE} images for faster evaluation")

# ---------------- Dataset / DataLoader ----------------
# NOTE: only resize, no other augmentation -- this is evaluation, not training
kvasir_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE)
])

kvasir_dataset = PolypDataset(
    kvasir_img_ids, KVASIR_IMG_DIR, KVASIR_MASK_DIR, KVASIR_IMG_EXT, kvasir_transform
)
kvasir_loader = torch.utils.data.DataLoader(kvasir_dataset, batch_size=BATCH_SIZE, shuffle=False)


def evaluate(model, loader, deep_supervision=False):
    model.eval()
    iou_meter = AverageMeter()
    dice_meter = AverageMeter()

    with torch.no_grad():
        for inputs, targets, _ in tqdm(loader, desc="Evaluating on Kvasir-SEG"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            if deep_supervision:
                outputs = outputs[-1]

            iou = iou_score(outputs, targets)
            dice = dice_score(outputs, targets)

            iou_meter.update(iou, inputs.size(0))
            dice_meter.update(dice, inputs.size(0))

    return iou_meter.avg, dice_meter.avg


# ---------------- UNet++ ----------------
print("\n" + "=" * 60)
print("Evaluating UNet++ on Kvasir-SEG (independent dataset)")
print("=" * 60)

unet = UNetPP(num_classes=1, input_channels=3, deep_supervision=True)
unet.load_state_dict(torch.load("models/best_model_unet++.pth", map_location=device))
unet = unet.to(device)

unet_kvasir_iou, unet_kvasir_dice = evaluate(unet, kvasir_loader, deep_supervision=True)
print(f"\nUNet++  on Kvasir-SEG  ->  IoU: {unet_kvasir_iou:.4f}   Dice: {unet_kvasir_dice:.4f}")

# ---------------- DeepLabV3+ ----------------
print("\n" + "=" * 60)
print("Evaluating DeepLabV3+ on Kvasir-SEG (independent dataset)")
print("=" * 60)

dlab = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
dlab.load_state_dict(torch.load("models/best_model_deeplabV3+.pth", map_location=device))
dlab = dlab.to(device)

dlab_kvasir_iou, dlab_kvasir_dice = evaluate(dlab, kvasir_loader, deep_supervision=False)
print(f"\nDeepLabV3+  on Kvasir-SEG  ->  IoU: {dlab_kvasir_iou:.4f}   Dice: {dlab_kvasir_dice:.4f}")

# ---------------- SUMMARY ----------------
print("\n" + "=" * 60)
print("CROSS-DATASET GENERALIZATION RESULTS")
print("Trained on: your colonoscopy dataset (610 images)")
print(f"Tested on:  Kvasir-SEG, {len(kvasir_img_ids)} images (NEVER seen during training)")
print("=" * 60)
print(f"{'Model':<15} {'Kvasir IoU':<14} {'Kvasir Dice':<14}")
print(f"{'UNet++':<15} {unet_kvasir_iou:<14.4f} {unet_kvasir_dice:<14.4f}")
print(f"{'DeepLabV3+':<15} {dlab_kvasir_iou:<14.4f} {dlab_kvasir_dice:<14.4f}")

with open("kvasir_seg_results.txt", "w") as f:
    f.write("CROSS-DATASET GENERALIZATION RESULTS\n")
    f.write("Trained on: own colonoscopy dataset (610 images)\n")
    f.write(f"Tested on:  Kvasir-SEG, {len(kvasir_img_ids)} images (independent, never seen during training)\n\n")
    f.write(f"{'Model':<15} {'Kvasir IoU':<14} {'Kvasir Dice':<14}\n")
    f.write(f"{'UNet++':<15} {unet_kvasir_iou:<14.4f} {unet_kvasir_dice:<14.4f}\n")
    f.write(f"{'DeepLabV3+':<15} {dlab_kvasir_iou:<14.4f} {dlab_kvasir_dice:<14.4f}\n")

print("\nSaved to kvasir_seg_results.txt")
print("\nDone! Copy the numbers above (or the .txt file contents) back to Claude")
print("to update the report with these cross-dataset generalization results.")

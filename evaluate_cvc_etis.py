# ============================================================
# evaluate_cvc_etis.py
# ============================================================
# Evaluate your ALREADY-TRAINED UNet++ and DeepLabV3+ models on
# CVC-ColonDB and ETIS-LaribPolypDB -- two more INDEPENDENT
# datasets they have never seen (same idea as evaluate_kvasir_seg.py,
# just two datasets in one run instead of one).
#
# This is INFERENCE ONLY. No training happens here. Both models
# are loaded from their saved .pth checkpoints and run on each
# dataset to see how well they generalize beyond your own dataset.
#
# HOW TO USE:
#   1. Make sure your trained models are accessible:
#        models/best_model_unet++.pth
#        models/best_model_deeplabV3+.pth
#   2. Make sure these folders exist (already confirmed on your machine):
#        TestDataset/CVC-ColonDB/images/*.png
#        TestDataset/CVC-ColonDB/masks/*.png
#        TestDataset/ETIS-LaribPolypDB/images/*.png
#        TestDataset/ETIS-LaribPolypDB/masks/*.png
#   3. Run this script as-is from the project root:
#        python evaluate_cvc_etis.py
# ============================================================

import os
import glob
import torch
import albumentations as A
from tqdm import tqdm

from source.model import UNetPP
from source.dataset import PolypDataset
from source.utils import iou_score, dice_score, AverageMeter

import segmentation_models_pytorch as smp

# ---------------- CONFIG: update these paths if your folder names differ ----------------
DATASETS = {
    "CVC-ColonDB": {
        "img_dir": "TestDataset/CVC-ColonDB/images",
        "mask_dir": "TestDataset/CVC-ColonDB/masks",
        "ext": ".png",
    },
    "ETIS-LaribPolypDB": {
        "img_dir": "TestDataset/ETIS-LaribPolypDB/images",
        "mask_dir": "TestDataset/ETIS-LaribPolypDB/masks",
        "ext": ".png",
    },
}

IMAGE_SIZE = 384      # SAME resolution your models were trained on
BATCH_SIZE = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# NOTE: only resize, no other augmentation -- this is evaluation, not training
eval_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE)
])


def build_loader(img_dir, mask_dir, ext):
    img_paths = glob.glob(os.path.join(img_dir, "*" + ext))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_paths]

    if len(img_ids) == 0:
        raise RuntimeError(
            f"No images found in {img_dir} with extension {ext}! "
            "Check the path and extension match your actual folder structure."
        )

    dataset = PolypDataset(img_ids, img_dir, mask_dir, ext, eval_transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return loader, img_ids


def evaluate(model, loader, dataset_name, deep_supervision=False):
    model.eval()
    iou_meter = AverageMeter()
    dice_meter = AverageMeter()

    with torch.no_grad():
        for inputs, targets, _ in tqdm(loader, desc=f"Evaluating on {dataset_name}"):
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


# ---------------- Load models ONCE ----------------
print("\nLoading models...")
unet = UNetPP(num_classes=1, input_channels=3, deep_supervision=True)
unet.load_state_dict(torch.load("models/best_model_unet++.pth", map_location=device))
unet = unet.to(device)

dlab = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
dlab.load_state_dict(torch.load("models/best_model_deeplabV3+.pth", map_location=device))
dlab = dlab.to(device)

# ---------------- Evaluate both models on both datasets ----------------
results = {}  # results[dataset_name] = {"unet_iou":.., "unet_dice":.., "dlab_iou":.., "dlab_dice":.., "n":..}

for dataset_name, cfg in DATASETS.items():
    print("\n" + "=" * 60)
    print(f"Dataset: {dataset_name}")
    print("=" * 60)

    loader, img_ids = build_loader(cfg["img_dir"], cfg["mask_dir"], cfg["ext"])
    print(f"Found {len(img_ids)} images in {dataset_name}")

    unet_iou, unet_dice = evaluate(unet, loader, dataset_name, deep_supervision=True)
    print(f"UNet++       on {dataset_name}  ->  IoU: {unet_iou:.4f}   Dice: {unet_dice:.4f}")

    dlab_iou, dlab_dice = evaluate(dlab, loader, dataset_name, deep_supervision=False)
    print(f"DeepLabV3+   on {dataset_name}  ->  IoU: {dlab_iou:.4f}   Dice: {dlab_dice:.4f}")

    results[dataset_name] = {
        "n": len(img_ids),
        "unet_iou": unet_iou, "unet_dice": unet_dice,
        "dlab_iou": dlab_iou, "dlab_dice": dlab_dice,
    }

# ---------------- SUMMARY ----------------
print("\n" + "=" * 60)
print("CROSS-DATASET GENERALIZATION RESULTS")
print("Trained on: your colonoscopy dataset (610 images)")
print("Tested on:  CVC-ColonDB and ETIS-LaribPolypDB (NEVER seen during training)")
print("=" * 60)

header = f"{'Dataset':<20} {'Model':<12} {'IoU':<10} {'Dice':<10}"
print(header)
lines = [header]

for dataset_name, r in results.items():
    line1 = f"{dataset_name:<20} {'UNet++':<12} {r['unet_iou']:<10.4f} {r['unet_dice']:<10.4f}"
    line2 = f"{'':<20} {'DeepLabV3+':<12} {r['dlab_iou']:<10.4f} {r['dlab_dice']:<10.4f}"
    print(line1)
    print(line2)
    lines.append(line1)
    lines.append(line2)

with open("cvc_etis_results.txt", "w") as f:
    f.write("CROSS-DATASET GENERALIZATION RESULTS\n")
    f.write("Trained on: own colonoscopy dataset (610 images)\n")
    f.write("Tested on:  CVC-ColonDB and ETIS-LaribPolypDB (independent, never seen during training)\n\n")
    for dataset_name, r in results.items():
        f.write(f"\n{dataset_name} ({r['n']} images)\n")
        f.write(f"  UNet++      ->  IoU: {r['unet_iou']:.4f}   Dice: {r['unet_dice']:.4f}\n")
        f.write(f"  DeepLabV3+  ->  IoU: {r['dlab_iou']:.4f}   Dice: {r['dlab_dice']:.4f}\n")

print("\nSaved to cvc_etis_results.txt")
print("\nDone! Copy the numbers above (or the .txt file contents) back to Claude")
print("to update the report with these cross-dataset generalization results.")

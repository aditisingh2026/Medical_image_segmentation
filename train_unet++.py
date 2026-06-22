# ============================================================
# train.py — Improved Polyp Segmentation Training (UNet++)
#
# UPDATED:
#   - Dice score now logged alongside IoU (train + val)
#   - Checkpoint/resume support added (safe against Colab disconnects)
#   - epochs bumped 80 -> 120 (val IoU hadn't plateaued)
#   - scheduler patience 5 -> 8 (val IoU is noisy; avoid premature LR drops)
# ============================================================

import os
import yaml
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from visualize import visualize_predictions

from glob import glob
from tqdm import tqdm
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import albumentations as A

from source.utils import iou_score, dice_score, AverageMeter
from source.model import UNetPP
from source.dataset import PolypDataset


# ============================================================
# TRAIN FUNCTION
# ============================================================
def train(deep_sup, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter()}
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pbar = tqdm(total=len(train_loader))

    for image, mask, _ in train_loader:
        image = image.to(device)
        mask = mask.to(device)

        if deep_sup:
            outputs = model(image)
            loss = 0
            for output in outputs:
                loss += criterion(output, mask)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], mask)
            dice = dice_score(outputs[-1], mask)
        else:
            output = model(image)
            loss = criterion(output, mask)
            iou = iou_score(output, mask)
            dice = dice_score(output, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), image.size(0))
        avg_meters['iou'].update(iou, image.size(0))
        avg_meters['dice'].update(dice, image.size(0))

        pbar.set_postfix({
            'loss': avg_meters['loss'].avg,
            'iou': avg_meters['iou'].avg,
            'dice': avg_meters['dice'].avg
        })
        pbar.update(1)

    pbar.close()

    return {
        'loss': avg_meters['loss'].avg,
        'iou': avg_meters['iou'].avg,
        'dice': avg_meters['dice'].avg
    }


# ============================================================
# VALIDATION FUNCTION
# ============================================================
def validate(deep_sup, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter()}
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))

        for image, mask, _ in val_loader:
            image = image.to(device)
            mask = mask.to(device)

            if deep_sup:
                outputs = model(image)
                loss = 0
                for output in outputs:
                    loss += criterion(output, mask)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], mask)
                dice = dice_score(outputs[-1], mask)
            else:
                output = model(image)
                loss = criterion(output, mask)
                iou = iou_score(output, mask)
                dice = dice_score(output, mask)

            avg_meters['loss'].update(loss.item(), image.size(0))
            avg_meters['iou'].update(iou, image.size(0))
            avg_meters['dice'].update(dice, image.size(0))

            pbar.set_postfix({
                'loss': avg_meters['loss'].avg,
                'iou': avg_meters['iou'].avg,
                'dice': avg_meters['dice'].avg
            })
            pbar.update(1)

        pbar.close()

    return {
        'loss': avg_meters['loss'].avg,
        'iou': avg_meters['iou'].avg,
        'dice': avg_meters['dice'].avg
    }


# ============================================================
# DICE LOSS
# ============================================================
class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1e-5

        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()

        dice = (2. * intersection + smooth) / (
            pred.sum() + target.sum() + smooth
        )

        return 1 - dice


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # ---------------- CONFIG ----------------
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    image_ext  = config["image_ext"]
    epochs     = 120                         # bumped from 80 — val IoU hadn't plateaued
    log_path   = config["log_path"]
    mask_path  = config["mask_path"]
    image_path = config["image_path"]
    model_path = config["model_path"]
    ckpt_path  = model_path.replace(".pth", "_checkpoint.pth")   # full resume state

    image_size = 384   #  improved from 256
    batch_size = config["batch_size"]
    lr = 1e-4          #  improved learning rate

    # ---------------- DATA ----------------
    img_ids = glob(os.path.join(image_path, "*" + image_ext))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(
        img_ids,
        test_size=0.2,
        random_state=42
    )

    print(f"Training images:   {len(train_img_ids)}")
    print(f"Validation images: {len(val_img_ids)}")

    # ---------------- AUGMENTATION ----------------
    train_transform = A.Compose([
        A.Resize(image_size, image_size),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=30, p=0.5),

        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.ElasticTransform(p=0.2),
        A.CLAHE(p=0.2),
        A.GaussianBlur(p=0.2),

        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.GaussNoise(p=0.2),
    ])

    val_transform = A.Compose([
        A.Resize(image_size, image_size)
    ])

    # ---------------- DATASET ----------------
    train_dataset = PolypDataset(train_img_ids, image_path, mask_path, image_ext, train_transform)
    val_dataset   = PolypDataset(val_img_ids, image_path, mask_path, image_ext, val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ---------------- MODEL ----------------
    model = UNetPP(num_classes=1, input_channels=3, deep_supervision=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Using device: {device}")

    # ---------------- LOSS ----------------
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    def criterion(pred, target):
        return 0.5 * bce_loss(pred, target) + 0.5 * dice_loss(pred, target)

    # ---------------- OPTIMIZER ----------------
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)

    #  LR SCHEDULER (patience raised: val IoU is noisy)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=8,
        factor=0.5
    )

    # ---------------- LOG ----------------
    log = OrderedDict([
        ('epoch', []),
        ('loss', []),
        ('iou', []),
        ('dice', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

    best_iou = 0
    start_epoch = 0

    # ---------------- RESUME FROM CHECKPOINT (if Colab disconnected) ----------------
    if os.path.exists(ckpt_path):
        print(f"\n>> Found checkpoint at {ckpt_path} — resuming training.")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_iou = checkpoint['best_iou']

        # restore log history so the CSV stays continuous, not overwritten
        if os.path.exists(log_path):
            prev_log = pd.read_csv(log_path)
            for col in log:
                log[col] = prev_log[col].tolist()

        print(f">> Resuming from epoch {start_epoch + 1}, best IoU so far: {best_iou:.4f}\n")
    else:
        print("\n>> No checkpoint found — starting fresh.\n")

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")

        train_log = train(True, train_loader, model, criterion, optimizer)
        val_log = validate(True, val_loader, model, criterion)

        scheduler.step(val_log['iou'])  #  important

        print(f"loss: {train_log['loss']:.4f} | iou: {train_log['iou']:.4f} | dice: {train_log['dice']:.4f} | "
              f"val_loss: {val_log['loss']:.4f} | val_iou: {val_log['iou']:.4f} | val_dice: {val_log['dice']:.4f}")

        log['epoch'].append(epoch + 1)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv(log_path, index=False)

        if val_log['iou'] > best_iou:
            best_iou = val_log['iou']
            torch.save(model.state_dict(), model_path)
            print(f"=> Best model saved! IoU: {best_iou:.4f}")

        # ---- Save FULL checkpoint every epoch (cheap insurance against disconnects) ----
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_iou': best_iou,
        }, ckpt_path)

    print(f"\nTraining complete! Best IoU: {best_iou:.4f}")
    print(f"Model saved at: {model_path}")


    # for visualization 
    visualize_predictions(model, val_loader, device)
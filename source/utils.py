# ============================================================
# utils.py — Polyp Segmentation Project 
# ============================================================

import torch


# ============================================================
# AverageMeter
# ============================================================
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ============================================================
# Helper: Convert logits → binary mask
# ============================================================
def _convert(pred):
    if isinstance(pred, list):
        pred = pred[-1]   # deep supervision fix

    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    return pred


# ============================================================
# IoU Score (FIXED)
# ============================================================
def iou_score(output, target, smooth=1e-5):

    output = _convert(output)
    target = target.float()

    # flatten
    output = output.view(-1)
    target = target.view(-1)

    intersection = (output * target).sum()
    union = output.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


# ============================================================
# Dice Score (FIXED)
# ============================================================
def dice_score(output, target, smooth=1e-5):

    output = _convert(output)
    target = target.float()

    # flatten
    output = output.view(-1)
    target = target.view(-1)

    intersection = (output * target).sum()

    dice = (2. * intersection + smooth) / (
        output.sum() + target.sum() + smooth
    )

    return dice.item()
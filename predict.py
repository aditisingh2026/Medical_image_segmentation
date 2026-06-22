# ============================================================
# predict.py — Polyp Segmentation Project (IMPROVED)
# ============================================================

import os
import yaml
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import albumentations as A

from source.model import UNetPP


# ============================================================
# Prediction Function
# ============================================================
def predict_single(model, image_path, mask_path, transform, device, save_path):

    # -----------------------------
    # Load image
    # -----------------------------
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # Load mask (optional for visualization)
    # -----------------------------
    mask = cv2.imread(mask_path, 0) if mask_path else None

    # -----------------------------
    # Apply transform
    # -----------------------------
    if transform:
        augmented = transform(image=image, mask=mask) if mask is not None else transform(image=image)
        image = augmented["image"]
        mask = augmented.get("mask", None)

    # -----------------------------
    # Normalize image
    # -----------------------------
    image = image.astype(np.float32) / 255.0

    # HWC → CHW
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)

    model.eval()

    # -----------------------------
    # Inference
    # -----------------------------
    with torch.no_grad():
        output = model(image)

        if isinstance(output, list):
            output = output[-1]

        output = torch.sigmoid(output)
        pred_mask = output.squeeze().cpu().numpy()

        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    # -----------------------------
    # Plot results
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    if mask is not None:
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Ground Truth")
    else:
        axes[1].imshow(np.zeros_like(pred_mask), cmap="gray")
        axes[1].set_title("Ground Truth (Not Available)")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return pred_mask


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    image_ext   = config["image_ext"]
    mask_path   = config["mask_path"]
    image_path  = config["image_path"]
    model_path  = config["model_path"]
    image_size  = config["image_size"]
    result_path = config["result_path"]

    os.makedirs(result_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Load model
    # -----------------------------
    model = UNetPP(num_classes=1, input_channels=3, deep_supervision=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    print(f"Model loaded from: {model_path}")

    # -----------------------------
    # Transform (no augmentation)
    # -----------------------------
    transform = A.Compose([
        A.Resize(image_size, image_size)
    ])

    # -----------------------------
    # Predict all images
    # -----------------------------
    img_ids = glob(os.path.join(image_path, "*" + image_ext))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    print(f"Total images: {len(img_ids)}")

    for img_id in img_ids:

        img_path = os.path.join(image_path, img_id + image_ext)
        mask_path_single = os.path.join(mask_path, img_id + image_ext)

        save_path = os.path.join(result_path, img_id + "_result.png")

        predict_single(
            model=model,
            image_path=img_path,
            mask_path=mask_path_single,
            transform=transform,
            device=device,
            save_path=save_path
        )

    print(f"\nDone! Results saved in: {result_path}")
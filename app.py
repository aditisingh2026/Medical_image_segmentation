# ============================================================
# app.py — Polyp Segmentation: UNet++ vs DeepLabV3+ Comparison
# ============================================================
#
# Run with:  streamlit run app.py
#
# This app is built to be genuinely useful, not just a demo:
#   1. Side-by-side predictions (Original | GT | UNet++ | DeepLabV3+)
#   2. Per-image metrics (IoU, Dice, inference time) for both models
#   3. Overlay visualization (TP/FP/FN) showing WHERE each model
#      gets it right or wrong, not just a binary mask
#   4. Aggregate stats across your whole dataset, so you can show
#      "Model A wins on X% of images" instead of one cherry-picked case
#
# ============================================================

import os
import time
import yaml
import torch
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import albumentations as A
import segmentation_models_pytorch as smp

from source.model import UNetPP

st.set_page_config(
    page_title="Polyp Segmentation — UNet++ vs DeepLabV3+",
    layout="wide",
)

# ============================================================
# CONFIG — update these paths to match your project
# ============================================================
CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

IMAGE_PATH = config["image_path"]
MASK_PATH = config["mask_path"]
IMAGE_EXT = config["image_ext"]
IMAGE_SIZE = 384

UNET_MODEL_PATH = "models/best_model_unet++.pth"
DEEPLAB_MODEL_PATH = "models/best_model_deeplabV3+.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL LOADING (cached so it only happens once per session)
# ============================================================
@st.cache_resource
def load_models():
    unet_model = UNetPP(num_classes=1, input_channels=3, deep_supervision=True)
    unet_model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=DEVICE))
    unet_model.to(DEVICE).eval()

    deeplab_model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    deeplab_model.load_state_dict(torch.load(DEEPLAB_MODEL_PATH, map_location=DEVICE))
    deeplab_model.to(DEVICE).eval()

    return unet_model, deeplab_model


@st.cache_data
def list_dataset_ids():
    ids = [
        os.path.splitext(f)[0]
        for f in os.listdir(IMAGE_PATH)
        if f.endswith(IMAGE_EXT)
    ]
    return sorted(ids)


# ============================================================
# INFERENCE HELPERS
# ============================================================
resize = A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE)])


def load_image_and_mask(img_id):
    img = np.array(Image.open(os.path.join(IMAGE_PATH, img_id + IMAGE_EXT)).convert("RGB"))
    mask_file = os.path.join(MASK_PATH, img_id + IMAGE_EXT)
    mask = np.array(Image.open(mask_file).convert("L")) if os.path.exists(mask_file) else None

    resized = resize(image=img, mask=mask) if mask is not None else resize(image=img)
    img_resized = resized["image"]
    mask_resized = resized.get("mask")

    return img, img_resized, mask_resized


def preprocess(img_resized):
    tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0).to(DEVICE)


def predict(model, img_tensor, is_unet):
    start = time.time()
    with torch.no_grad():
        output = model(img_tensor)
        if is_unet:  # deep supervision -> list of outputs, use the last (finest)
            output = output[-1]
        pred = torch.sigmoid(output)
        pred_mask = (pred > 0.5).float().cpu().numpy()[0, 0]
    elapsed = time.time() - start
    return pred_mask, elapsed


def compute_metrics(pred_mask, gt_mask, smooth=1e-5):
    if gt_mask is None:
        return None, None
    gt = (gt_mask > 127).astype(np.float32)
    pred = pred_mask.astype(np.float32)

    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * intersection + smooth) / (pred.sum() + gt.sum() + smooth)
    return float(iou), float(dice)


def make_overlay(pred_mask, gt_mask):
    """
    Green  = correctly predicted polyp (True Positive)
    Red    = model missed it (False Negative — GT says polyp, model said no)
    Blue   = model over-predicted (False Positive — model said polyp, GT says no)
    """
    h, w = pred_mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    if gt_mask is None:
        overlay[..., 1] = (pred_mask * 255).astype(np.uint8)
        return overlay

    gt = (gt_mask > 127).astype(np.float32)
    pred = pred_mask.astype(np.float32)

    tp = (pred == 1) & (gt == 1)
    fn = (pred == 0) & (gt == 1)
    fp = (pred == 1) & (gt == 0)

    overlay[tp] = [46, 204, 113]   # green
    overlay[fn] = [231, 76, 60]    # red
    overlay[fp] = [52, 120, 246]   # blue

    return overlay


# ============================================================
# UI
# ============================================================
st.title("Polyp Segmentation — UNet++ vs DeepLabV3+")
st.caption(
    "Both models trained from scratch on identical data, augmentations, and "
    "hyperparameters — differences below reflect architecture alone."
)

try:
    unet_model, deeplab_model = load_models()
except FileNotFoundError as e:
    st.error(
        f"Couldn't find a saved model file: {e}\n\n"
        "Make sure training has completed and model_path / "
        "model_path_deeplab point to valid .pth files in config.yaml."
    )
    st.stop()

tab_explore, tab_aggregate = st.tabs(["🔍 Explore a single image", "📊 Aggregate comparison"])

# ------------------------------------------------------------
# TAB 1 — single image explorer
# ------------------------------------------------------------
with tab_explore:
    col_select, col_upload = st.columns(2)

    with col_select:
        st.subheader("Pick an image from the dataset")
        ids = list_dataset_ids()
        chosen_id = st.selectbox("Image ID", ids, index=0)

    with col_upload:
        st.subheader("...or upload your own")
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        img_full = np.array(Image.open(uploaded).convert("RGB"))
        resized = resize(image=img_full)
        img_resized = resized["image"]
        gt_resized = None
        st.info("No ground truth available for uploaded images — showing predictions only.")
    else:
        img_full, img_resized, gt_resized = load_image_and_mask(chosen_id)

    img_tensor = preprocess(img_resized)

    unet_pred, unet_time = predict(unet_model, img_tensor, is_unet=True)
    deeplab_pred, deeplab_time = predict(deeplab_model, img_tensor, is_unet=False)

    unet_iou, unet_dice = compute_metrics(unet_pred, gt_resized)
    deeplab_iou, deeplab_dice = compute_metrics(deeplab_pred, gt_resized)

    st.divider()
    st.subheader("Side-by-side prediction")

    cols = st.columns(4)
    cols[0].image(img_resized, caption="Original", use_container_width=True)

    if gt_resized is not None:
        cols[1].image(gt_resized, caption="Ground Truth", use_container_width=True, clamp=True)
    else:
        cols[1].info("No ground truth for this image.")

    cols[2].image(
        (unet_pred * 255).astype(np.uint8),
        caption=f"UNet++ ({unet_time*1000:.0f} ms)",
        use_container_width=True,
    )
    cols[3].image(
        (deeplab_pred * 255).astype(np.uint8),
        caption=f"DeepLabV3+ ({deeplab_time*1000:.0f} ms)",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Error overlay — where each model gets it right or wrong")
    st.caption("🟩 Correct (TP)   🟥 Missed polyp (FN)   🟦 Over-predicted (FP)")

    overlay_cols = st.columns(2)
    overlay_cols[0].image(make_overlay(unet_pred, gt_resized), caption="UNet++ overlay", use_container_width=True)
    overlay_cols[1].image(make_overlay(deeplab_pred, gt_resized), caption="DeepLabV3+ overlay", use_container_width=True)

    if gt_resized is not None:
        st.divider()
        st.subheader("Metrics for this image")
        metrics_df = pd.DataFrame(
            {
                "Model": ["UNet++", "DeepLabV3+"],
                "IoU": [unet_iou, deeplab_iou],
                "Dice": [unet_dice, deeplab_dice],
                "Inference time (ms)": [unet_time * 1000, deeplab_time * 1000],
            }
        )
        st.dataframe(metrics_df.style.format({"IoU": "{:.4f}", "Dice": "{:.4f}", "Inference time (ms)": "{:.1f}"}),
                     use_container_width=True, hide_index=True)

        better = "UNet++" if unet_iou > deeplab_iou else ("DeepLabV3+" if deeplab_iou > unet_iou else "Tie")
        st.success(f"On this image, **{better}** has the higher IoU.")


# ------------------------------------------------------------
# TAB 2 — aggregate comparison across the whole dataset
# ------------------------------------------------------------
with tab_aggregate:
    st.subheader("Run both models across the full dataset")
    st.caption(
        "This computes IoU/Dice for every image with a ground truth mask. "
        "May take a minute depending on dataset size."
    )

    if st.button("Run aggregate comparison"):
        ids = list_dataset_ids()
        progress = st.progress(0, text="Running models...")

        rows = []
        for i, img_id in enumerate(ids):
            _, img_resized, gt_resized = load_image_and_mask(img_id)
            if gt_resized is None:
                continue

            img_tensor = preprocess(img_resized)
            unet_pred, _ = predict(unet_model, img_tensor, is_unet=True)
            deeplab_pred, _ = predict(deeplab_model, img_tensor, is_unet=False)

            u_iou, u_dice = compute_metrics(unet_pred, gt_resized)
            d_iou, d_dice = compute_metrics(deeplab_pred, gt_resized)

            rows.append({
                "image_id": img_id,
                "unet_iou": u_iou, "unet_dice": u_dice,
                "deeplab_iou": d_iou, "deeplab_dice": d_dice,
            })
            progress.progress((i + 1) / len(ids), text=f"Processing {img_id}...")

        progress.empty()
        results_df = pd.DataFrame(rows)
        st.session_state["results_df"] = results_df

    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"]

        st.divider()
        st.subheader("Summary")

        summary_cols = st.columns(4)
        summary_cols[0].metric("UNet++ mean IoU", f"{results_df['unet_iou'].mean():.4f}")
        summary_cols[1].metric("DeepLabV3+ mean IoU", f"{results_df['deeplab_iou'].mean():.4f}")
        summary_cols[2].metric("UNet++ mean Dice", f"{results_df['unet_dice'].mean():.4f}")
        summary_cols[3].metric("DeepLabV3+ mean Dice", f"{results_df['deeplab_dice'].mean():.4f}")

        unet_wins = (results_df["unet_iou"] > results_df["deeplab_iou"]).sum()
        deeplab_wins = (results_df["deeplab_iou"] > results_df["unet_iou"]).sum()
        ties = len(results_df) - unet_wins - deeplab_wins

        st.write(
            f"**UNet++ has higher IoU on {unet_wins} images ({unet_wins/len(results_df)*100:.1f}%)** — "
            f"**DeepLabV3+ on {deeplab_wins} images ({deeplab_wins/len(results_df)*100:.1f}%)** — "
            f"**{ties} ties**"
        )

        st.divider()
        st.subheader("Per-image results")
        st.dataframe(
            results_df.style.format({
                "unet_iou": "{:.4f}", "unet_dice": "{:.4f}",
                "deeplab_iou": "{:.4f}", "deeplab_dice": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download full results as CSV", csv, "comparison_results.csv", "text/csv")
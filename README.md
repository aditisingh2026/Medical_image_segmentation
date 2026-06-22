# Polyp Segmentation: UNet++ vs DeepLabV3+

A controlled architectural comparison of two segmentation models — **UNet++** (implemented from scratch) and **DeepLabV3+** (via `segmentation_models_pytorch`) — trained under identical conditions on a polyp segmentation dataset, to evaluate which architecture is better suited to this task.

---

## Objective

Both UNet++ and DeepLabV3+ solve the same task — pixel-level semantic segmentation — but through fundamentally different architectural philosophies:

- **UNet++** uses nested, dense skip connections between encoder and decoder to progressively close the semantic gap between low-level and high-level features.
- **DeepLabV3+** uses Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context via parallel dilated convolutions, paired with a lightweight decoder for boundary refinement.

This project trains both **from random initialization**, under **identical data splits, augmentations, loss functions, and hyperparameters**, to isolate the effect of architecture alone — rather than relying on benchmark numbers from unrelated datasets (e.g. natural images).

---

## Architectures

| | UNet++ | DeepLabV3+ |
|---|---|---|
| Implementation | From scratch (custom `VGGBlock`-based encoder/decoder) | `segmentation_models_pytorch` (ResNet34 encoder) |
| Pretrained weights | None — random init | None — random init (`encoder_weights=None`), for a fair comparison |
| Key mechanism | Nested dense skip pathways + deep supervision | Atrous Spatial Pyramid Pooling (ASPP) + decoder |
| Parameters | ~9.2M | ~22.4M |

---

## Training Setup (identical for both models)

| Setting | Value |
|---|---|
| Image size | 384 x 384 |
| Train / Validation split | 80% / 20% (`random_state=42`) |
| Augmentations | Horizontal/Vertical flip, Random rotate 90, Rotate +-30, Shift-Scale-Rotate, Elastic Transform, CLAHE, Gaussian Blur, Brightness/Contrast, Gaussian Noise |
| Loss function | 0.5 x BCEWithLogitsLoss + 0.5 x Dice Loss |
| Optimizer | Adam (lr = 1e-4, weight decay = 1e-4) |
| LR Scheduler | ReduceLROnPlateau (mode='max', patience=8, factor=0.5) |
| Epochs | 120 |
| Batch size | 4 |

---

## Results

| Model | Best Val IoU | Best Val Dice | Best Epoch |
|---|---|---|---|
| UNet++ | 0.8040 | 0.8886 | 116 |
| **DeepLabV3+** | **0.8612** | **0.9244** | 116 |

**DeepLabV3+ outperformed UNet++ on this dataset**, by ~5.7 points IoU and ~3.6 points Dice — a meaningful margin given both models were trained under identical conditions.

### Why this might be happening

Polyps vary substantially in size — from small, flat lesions to large, irregular ones. DeepLabV3+'s ASPP module processes the same feature map through multiple parallel dilation rates (6, 12, 18) plus global pooling, giving it built-in multi-scale awareness. UNet++ instead relies on dense skip connections to recover spatial detail, which is highly effective for sharpening boundaries but does not explicitly model multiple receptive field scales the way ASPP does.

This suggests that for datasets with high object-scale variance — as polyp datasets often are — multi-scale context modeling (DeepLabV3+'s core mechanism) provides a measurable advantage over boundary-focused dense skip connections (UNet++'s core mechanism) alone.

### Caveats

- Single training run per model (no multi-seed averaging) — the gap is large enough to be meaningful, but results could vary slightly with different seeds.
- Comparison made on one dataset; results may not generalize to other polyp datasets or other segmentation tasks.
- Hyperparameters (learning rate, loss weighting, augmentation set) were not separately tuned per architecture — both used the same configuration by design, to isolate architecture as the variable. Light tuning could shift results for either model.

---

## Interactive Comparison App

A Streamlit app is included (`app.py`) for exploring both models' predictions side by side:
- Original image, ground truth, and both models' predictions
- Error overlay (correct / missed polyp / false alarm) for visual error analysis
- Per-image and dataset-wide metrics (IoU, Dice, inference time)

Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
polyp_segmentation/
├── train.py              # UNet++ training (from scratch, deep supervision)
├── train_deeplab.py       # DeepLabV3+ training (ResNet34 encoder, no pretrained weights)
├── app.py                 # Streamlit comparison app
├── config.yaml            # Shared configuration (paths, image size, batch size)
├── requirements.txt
├── source/
│   ├── model.py            # UNet++ architecture (from scratch)
│   ├── dataset.py           # Dataset loading
│   └── utils.py              # Metrics (IoU, Dice) and helpers
└── data/
    ├── original/           # Input images
    └── ground_truth/        # Segmentation masks
```

Note: Trained model weights (`.pth` files) are not included in this repository due to file size. They are available on request / via the links below.

---

## Trained Models

- UNet++: https://drive.google.com/file/d/1d9x6RoTaZ9VraXjX8NnPl5EDE_i2EXLY/view?usp=drive_link
- DeepLabV3+: https://drive.google.com/file/d/1MMKiQxfT_C8S-afJClddsbufF8BNaK2O/view?usp=drive_link

---

## Conclusion

This comparison shows that under matched training conditions, **DeepLabV3+ achieved higher segmentation accuracy than UNet++ on this polyp dataset**, most likely due to its explicit multi-scale context modeling (ASPP) being well suited to the size variability of polyps. UNet++ remains a strong, parameter-efficient baseline, particularly valuable when implemented from scratch to understand the mechanics of dense skip connections and deep supervision in detail. Future work could explore hybrid approaches combining ASPP-style multi-scale context with denser skip connections, or evaluate both architectures on additional polyp datasets to confirm generalization of this finding.

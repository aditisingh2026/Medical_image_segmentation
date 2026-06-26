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

| Model | Train IoU | Train Dice | Val IoU | Val Dice | Best Epoch |
|---|---|---|---|---|---|
| UNet++ | 0.7481 | 0.8491 | 0.8040 | 0.8886 | 116 |
| **DeepLabV3+** | **0.8118** | **0.8919** | **0.8612** | **0.9244** | 116 |

**DeepLabV3+ outperformed UNet++ on this dataset**, by ~5.7 points IoU and ~3.6 points Dice on validation — a meaningful margin given both models were trained under identical conditions. Notably, validation performance was on par with (or slightly above) training performance for both models, suggesting the data augmentation strategy was effective in preventing overfitting on this small (612-image) dataset.

### Why this might be happening

Polyps vary substantially in size — from small, flat lesions to large, irregular ones. DeepLabV3+'s ASPP module processes the same feature map through multiple parallel dilation rates (6, 12, 18) plus global pooling, giving it built-in multi-scale awareness. UNet++ instead relies on dense skip connections to recover spatial detail, which is highly effective for sharpening boundaries but does not explicitly model multiple receptive field scales the way ASPP does.

This suggests that for datasets with high object-scale variance — as polyp datasets often are — multi-scale context modeling (DeepLabV3+'s core mechanism) provides a measurable advantage over boundary-focused dense skip connections (UNet++'s core mechanism) alone.

---

## Cross-Dataset Generalization

Validation accuracy on your own dataset only tells you how well a model fits *that* data distribution. To test how well each model generalizes to **completely unseen** data — different hospitals, cameras, and patient populations — both trained models were evaluated, with no further training, on three independent public polyp datasets:

| Dataset | Images | UNet++ IoU | UNet++ Dice | DeepLabV3+ IoU | DeepLabV3+ Dice |
|---|---|---|---|---|---|
| Kvasir-SEG | 200 (subset) | 0.4250 | 0.5833 | 0.5755 | 0.7179 |
| CVC-ColonDB | 380 | 0.3723 | 0.5065 | 0.5614 | 0.6706 |
| ETIS-LaribPolypDB | 196 | 0.1122 | 0.1816 | 0.2718 | 0.3760 |

**DeepLabV3+ generalized better than UNet++ on every single unseen dataset** — not just on the in-domain validation set. This consistency across four independent evaluations (1 in-domain + 3 cross-dataset) strengthens the case that DeepLabV3+'s advantage isn't a one-off result of this particular train/val split, but a real architectural effect.

The performance drop from in-domain validation to unseen datasets is itself expected and consistent with published literature: ETIS-LaribPolypDB in particular is widely regarded as the most challenging polyp dataset, due to its smaller, more concealed polyps. Both models show their largest drop on ETIS, which is the expected, literature-consistent pattern rather than a sign of a broken model.

Both encoders were trained from random initialization in this study (`encoder_weights=None` for DeepLabV3+, consistent with the from-scratch comparison philosophy), so the generalization gap cannot be attributed to ImageNet pretraining. The more likely explanation lies in the architectures themselves: ResNet's residual connections are known to ease optimization and produce smoother loss landscapes even from random initialization, which may help the encoder learn more generic, transferable features from a small 612-image dataset. DeepLabV3+'s ASPP module, by design, also captures multi-scale context explicitly rather than relying on the network to learn scale-invariance implicitly — this is a more robust prior when test images differ in resolution and polyp size from the training distribution, as is the case across Kvasir-SEG, CVC-ColonDB, and ETIS.

A natural follow-up experiment — using actual ImageNet-pretrained weights for DeepLabV3+'s encoder — would help isolate how much of this generalization gap comes from architecture alone versus the additional boost transfer learning could provide.

### Caveats

- Single training run per model (no multi-seed averaging) — the gap is large enough to be meaningful, but results could vary slightly with different seeds.
- Both models were trained from random initialization (no ImageNet pretraining) to isolate architecture as the variable; results may differ if pretrained weights were used.
- Hyperparameters (learning rate, loss weighting, augmentation set) were not separately tuned per architecture — both used the same configuration by design, to isolate architecture as the variable. Light tuning could shift results for either model.
- Cross-dataset evaluation used the trained `.pth` checkpoints with no fine-tuning on the new datasets — this measures zero-shot generalization, not transfer learning performance.

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
Medical_image_segmentation/
├── train_unet++.py            # UNet++ training (from scratch, deep supervision)
├── train_deeplabV3+.py        # DeepLabV3+ training (ResNet34 encoder, no pretrained weights)
├── evaluate_kvasir_seg.py      # Cross-dataset evaluation on Kvasir-SEG (unseen)
├── evaluate_cvc_etis.py         # Cross-dataset evaluation on CVC-ColonDB + ETIS-LaribPolypDB (unseen)
├── predict.py                  # Single-image inference
├── visualize.py                 # Visualization helpers
├── app.py                       # Streamlit comparison app
├── config.yaml                  # Shared configuration (paths, image size, batch size)
├── requirements.txt
├── source/
│   ├── model.py                  # UNet++ architecture (from scratch)
│   ├── dataset.py                  # Dataset loading
│   └── utils.py                     # Metrics (IoU, Dice) and helpers
├── data/
│   ├── original/                  # Input images (612 images)
│   └── ground_truth/                # Segmentation masks
├── logs/
│   ├── training_log_unet++.csv       # Per-epoch train/val metrics, UNet++
│   └── training_log_deeplabV3+.csv     # Per-epoch train/val metrics, DeepLabV3+
├── TestDataset/                    # Cross-dataset evaluation data (not used for training)
│   ├── CVC-ColonDB/
│   │   ├── images/
│   │   └── masks/
│   └── ETIS-LaribPolypDB/
│       ├── images/
│       └── masks/
├── kvasir_seg_results.txt          # Saved cross-dataset results (Kvasir-SEG)
└── cvc_etis_results.txt             # Saved cross-dataset results (CVC-ColonDB + ETIS)
```

Note: Trained model weights (`.pth` files), the raw `TestDataset` images, and Kvasir-SEG images are not included in this repository due to file size. Model weights are available via the links below; the public datasets can be re-downloaded using the sources noted in the evaluation scripts.

---

## Trained Models

- UNet++: https://drive.google.com/file/d/1d9x6RoTaZ9VraXjX8NnPl5EDE_i2EXLY/view?usp=drive_link
- DeepLabV3+: https://drive.google.com/file/d/1MMKiQxfT_C8S-afJClddsbufF8BNaK2O/view?usp=drive_link

---

## Conclusion

This comparison shows that under matched training conditions, **DeepLabV3+ achieved higher segmentation accuracy than UNet++ on this polyp dataset**, most likely due to its explicit multi-scale context modeling (ASPP) being well suited to the size variability of polyps. This advantage was **not limited to the in-domain validation set** — it held consistently across three independent, unseen public datasets (Kvasir-SEG, CVC-ColonDB, ETIS-LaribPolypDB), suggesting DeepLabV3+'s architectural choices genuinely improve generalization rather than simply overfitting better to this specific dataset's distribution.

UNet++ remains a strong, parameter-efficient baseline, particularly valuable when implemented from scratch to understand the mechanics of dense skip connections and deep supervision in detail. Future work could explore hybrid approaches combining ASPP-style multi-scale context with denser skip connections, use ImageNet-pretrained encoder weights to isolate the effect of transfer learning from architecture, or fine-tune on a combined multi-dataset training set to test whether the generalization gap narrows with more diverse training data.

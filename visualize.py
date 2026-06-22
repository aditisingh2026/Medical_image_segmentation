# this file is for the visualization of predicted masks -->

import torch
import matplotlib.pyplot as plt

def visualize_predictions(model, dataloader, device, num_samples=5):
    model.eval()
    images_shown = 0

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            for i in range(images.size(0)):
                if images_shown >= num_samples:
                    return

                img = images[i].cpu().permute(1,2,0).numpy()
                mask = masks[i].cpu().squeeze().numpy()
                pred = preds[i].cpu().squeeze().numpy()

                plt.figure(figsize=(12,4))

                plt.subplot(1,3,1)
                plt.title("Input")
                plt.imshow(img)
                plt.axis("off")

                plt.subplot(1,3,2)
                plt.title("Ground Truth")
                plt.imshow(mask, cmap="gray")
                plt.axis("off")

                plt.subplot(1,3,3)
                plt.title("Prediction")
                plt.imshow(pred, cmap="gray")
                plt.axis("off")

                plt.tight_layout()
                plt.savefig(f"prediction_{images_shown}.png")
                plt.show()

                images_shown += 1
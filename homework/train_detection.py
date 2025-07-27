
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb
import numpy as np

from .models import Detector, load_model, ClassificationLoss, save_model
from .datasets.road_dataset import load_data
from torch.nn import CrossEntropyLoss, L1Loss


def train_detection(
    exp_dir="logs",
    model_name="detector",
    num_epoch=30,
    lr=1e-4,
    batch_size=8,
    seed=2024,
    **kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built()
                          else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = Detector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    segmentation_loss_fn = CrossEntropyLoss()
    depth_loss_fn = L1Loss()

    train_data = load_data("drive_data/train", batch_size=batch_size, shuffle=True)
    val_data = load_data("drive_data/val", batch_size=batch_size, shuffle=False)

    for epoch in range(num_epoch):
        model.train()
        total_seg_loss, total_depth_loss = 0.0, 0.0

        for batch in train_data:
            images = batch["image"].to(device)
            depths = batch["depth"].to(device)
            labels = batch["track"].to(device).long()

            optimizer.zero_grad()

            logits, pred_depths = model(images)

            seg_loss = segmentation_loss_fn(logits, labels)
            depth_loss = depth_loss_fn(pred_depths, depths)
            loss = seg_loss + depth_loss

            loss.backward()
            optimizer.step()

            total_seg_loss += seg_loss.item()
            total_depth_loss += depth_loss.item()

        avg_seg_loss = total_seg_loss / len(train_data)
        avg_depth_loss = total_depth_loss / len(train_data)

        logger.add_scalar("train/seg_loss", avg_seg_loss, epoch)
        logger.add_scalar("train/depth_loss", avg_depth_loss, epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:2d}/{num_epoch}: seg_loss={avg_seg_loss:.4f}, depth_loss={avg_depth_loss:.4f}")

    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    train_detection(**vars(args))

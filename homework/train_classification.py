import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb
import numpy as np

from .models import load_model, ClassificationLoss, save_model
from .datasets.classification_dataset import load_data



def train_classification(
    exp_dir="logs",
    model_name="linear",
    num_epoch=50,
    lr=1e-3,
    batch_size=128,
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

    model = load_model(model_name, **kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = ClassificationLoss()

    train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("classification_data/val", shuffle=False)

    global_step = 0
    for epoch in range(num_epoch):
        model.train()
        train_acc = []

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model(img)
            loss = loss_func(logits, label)
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(dim=1) == label).float().mean()
            train_acc.append(acc.item())
            logger.add_scalar("train/acc", acc.item(), global_step)
            logger.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        model.eval()
        val_acc = []
        with torch.inference_mode():
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                logits = model(img)
                acc = (logits.argmax(dim=1) == label).float().mean()
                val_acc.append(acc.item())

        epoch_train_acc = sum(train_acc) / len(train_acc)
        epoch_val_acc = sum(val_acc) / len(val_acc)

        logger.add_scalar("epoch/train_acc", epoch_train_acc, epoch)
        logger.add_scalar("epoch/val_acc", epoch_val_acc, epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:2d}/{num_epoch}: train_acc={epoch_train_acc:.4f} val_acc={epoch_val_acc:.4f}")

    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    train_classification(**vars(args))

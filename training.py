import csv
import time
from pathlib import Path

import torch
from torch import nn

from data import PROJECT_DIR, count_per_class, make_loaders
from model import AnimalNet, count_parameters, save_model

EPOCHS = 80
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
MODEL_PATH = PROJECT_DIR / "animal_model.pt"
LOG_PATH = PROJECT_DIR / "training_log.csv"


def pick_device():
    """Use the GPU if there is one, otherwise the CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def class_weights(dataset):
    """Give rare classes more weight, so the model can't just guess the biggest class."""
    counts = count_per_class(dataset).float().clamp(min=1)
    return counts.sum() / (len(counts) * counts)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """One pass over the training data. Returns (average loss, accuracy)."""
    model.train()                                   # dropout on, BatchNorm learning
    total_loss, correct, seen = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        scores = model(images)                      # 1. forward: make a guess
        loss = loss_fn(scores, labels)              # 2. measure how wrong it was
        optimizer.zero_grad()                       # 3. clear old gradients
        loss.backward()                             # 4. work out the gradients
        optimizer.step()                            # 5. adjust the weights

        total_loss += loss.item() * len(labels)
        correct += (scores.argmax(dim=1) == labels).sum().item()
        seen += len(labels)

    return total_loss / seen, correct / seen


@torch.no_grad()                                    # no gradients needed when only checking
def evaluate(model, loader, loss_fn, device):
    """Check the model on data it doesn't train on. Returns (average loss, accuracy)."""
    model.eval()                                    # dropout off, BatchNorm fixed
    total_loss, correct, seen = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        scores = model(images)
        total_loss += loss_fn(scores, labels).item() * len(labels)
        correct += (scores.argmax(dim=1) == labels).sum().item()
        seen += len(labels)

    return total_loss / seen, correct / seen


def main(epochs=EPOCHS, model_path=MODEL_PATH, log_path=LOG_PATH):
    device = pick_device()
    train_loader, val_loader, classes = make_loaders(batch_size=BATCH_SIZE)

    model = AnimalNet(num_classes=len(classes)).to(device)
    weights = class_weights(train_loader.dataset).to(device)
    # label_smoothing: aim for 90% sure instead of 100%, so it can't be 99% sure of wrong answers
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    # AdamW + weight_decay: gently pulls the weights toward zero, which discourages memorising.
    # 1e-4 (was 5e-4): the model now underfits, so it needs less holding back
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) 
    print(f"Device: {device}")
    print(f"Classes: {classes}")
    print(f"Train images: {len(train_loader.dataset)}, val images: {len(val_loader.dataset)}")
    print(f"Parameters: {count_parameters(model):,}\n")

    best_val_acc, best_val_loss = -1.0, float("inf")
    with open(log_path, "w", newline="") as log_file:
        log = csv.writer(log_file)
        log.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
            log.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}"])
            scheduler.step()

            # Keep the version with the lowest validation loss. Loss is steadier than accuracy
            # on a small val set, where one photo moves accuracy by about 1%.
            saved = ""
            if val_loss < best_val_loss:
                best_val_acc, best_val_loss = val_acc, val_loss
                save_model(model, classes, model_path)
                saved = "  <- saved"


            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"lr {optimizer.param_groups[0]['lr']:.6f}  "
                f"train loss {train_loss:.3f} acc {train_acc:.0%}  |  "
                f"val loss {val_loss:.3f} acc {val_acc:.0%}  "
                f"({time.time() - start:.1f}s){saved}"
            )

    print(f"\nBest model saved to {Path(model_path).name} (val acc {best_val_acc:.0%}, val loss {best_val_loss:.3f})")
    print(f"Training log saved to {Path(log_path).name}")


if __name__ == "__main__":
    main()

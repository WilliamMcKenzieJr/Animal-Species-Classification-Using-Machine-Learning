import torch
from torch import nn

from data import IMAGE_SIZE


def conv_block(in_channels, out_channels):
    """Two (Conv -> BatchNorm -> ReLU) layers, then MaxPool. Halves the image's height and width.

    The second conv combines the first one's patterns before the image shrinks,
    which gives the network more capacity to learn harder photos.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class AnimalNet(nn.Module):
    """A small CNN for 96x96 colour images, trained from scratch."""

    def __init__(self, num_classes, dropout=0.1):
        super().__init__()
        # Each block finds more complex patterns on a smaller image
        self.features = nn.Sequential(
            conv_block(3, 32),      # (3, 96, 96)   -> (32, 48, 48)   edges, colours
            conv_block(32, 64),     # (32, 48, 48)  -> (64, 24, 24)   textures, fur
            conv_block(64, 128),    # (64, 24, 24)  -> (128, 12, 12)  ears, eyes, snouts
            conv_block(128, 256),   # (128, 12, 12) -> (256, 6, 6)    animal parts together
        )
        # Turn the patterns into one score per class
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # (256, 6, 6) -> (256, 1, 1): average each pattern
            nn.Flatten(),               # -> (256,)
            nn.Dropout(dropout),        # randomly drop some values while training
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """x: (batch, 3, H, W) -> scores of shape (batch, num_classes)."""
        return self.classifier(self.features(x))


def save_model(model, classes, path):
    """Save the weights together with the class names."""
    torch.save({"model": model.state_dict(), "classes": classes}, path)


def load_model(path, device="cpu"):
    """Load a model saved with save_model. Returns (model, classes)."""
    checkpoint = torch.load(path, map_location=device)
    classes = checkpoint["classes"]
    model = AnimalNet(num_classes=len(classes))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, classes


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = AnimalNet(num_classes=7)
    print(model)
    print(f"\nParameters: {count_parameters(model):,}")

    # A fake batch of 4 images, to check the shapes line up
    fake_images = torch.rand(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    scores = model(fake_images)
    print("Input: ", tuple(fake_images.shape))
    print("Output:", tuple(scores.shape))

    # Scores become probabilities with softmax; untrained, they're just random guesses
    probabilities = torch.softmax(scores, dim=1)
    print("Probabilities for the first image:", [round(p, 3) for p in probabilities[0].tolist()])

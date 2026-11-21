from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_DIR = Path(__file__).parent
TRAIN_DIR = PROJECT_DIR / "train"
VAL_DIR = PROJECT_DIR / "validate"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".jfif", ".png")
IMAGE_SIZE = 96


def is_image(path):
    """True for the file types we use. ImageFolder skips .jfif unless told otherwise."""
    return path.lower().endswith(IMAGE_EXTENSIONS)


def load_image(path):
    """Open an image as RGB, fixing sideways phone photos."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")


def build_transform(size=IMAGE_SIZE, augment=False):
    """Turn a PIL image into a (3, size, size) tensor with values 0-1."""
    if augment:
        # Training: a different version of each image every epoch, so the model
        # has to learn the animal's shape instead of shortcuts like colour or background.
        # scale 0.6: keep at least 60% of the photo. 0.4 cut too much off now that
        # there are enough photos, and the model was underfitting.
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
        ])
    # Validation and prediction: always the same centre crop
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])


def load_datasets(size=IMAGE_SIZE, augment=True):
    """Load train and validate. Labels are the top-level folders; subfolders are included."""
    train_set = datasets.ImageFolder(
        TRAIN_DIR,
        transform=build_transform(size, augment),
        loader=load_image,
        is_valid_file=is_image,
        allow_empty=True,
    )
    val_set = datasets.ImageFolder(
        VAL_DIR,
        transform=build_transform(size),
        loader=load_image,
        is_valid_file=is_image,
        allow_empty=True,
    )

    # Label numbers only mean the same thing if both folders have the same classes
    if val_set.classes != train_set.classes:
        raise ValueError(
            f"train and validate have different classes:\n"
            f"  train:    {train_set.classes}\n"
            f"  validate: {val_set.classes}"
        )
    return train_set, val_set


def make_loaders(batch_size=16, size=IMAGE_SIZE, augment=True, num_workers=0):
    """Return (train_loader, val_loader, classes)."""
    train_set, val_set = load_datasets(size, augment)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_set.classes


def count_per_class(dataset):
    """Number of images in each class, in class order."""
    return torch.bincount(torch.tensor(dataset.targets), minlength=len(dataset.classes))


def show_examples(images, labels, classes, count=8):
    """Show a few images with their labels, to check the data looks right."""
    import matplotlib.pyplot as plt

    count = min(count, len(images))
    fig, axes = plt.subplots(1, count, figsize=(2 * count, 2.5))
    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img.permute(1, 2, 0))       # (3, H, W) -> (H, W, 3) for display
        ax.set_title(classes[label])
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    train_loader, val_loader, classes = make_loaders()
    train_set, val_set = train_loader.dataset, val_loader.dataset

    print("Classes:", classes)
    print("Train images:", len(train_set))
    print("Val images:  ", len(val_set))

    print("\nImages per class (train / val):")
    for name, t, v in zip(classes, count_per_class(train_set), count_per_class(val_set)):
        print(f"  {name:10s} {t:4d} / {v}")

    images, labels = next(iter(train_loader))
    print("\nOne batch:", tuple(images.shape), tuple(labels.shape))
    print("Value range:", images.min().item(), "to", images.max().item())

    show_examples(images, labels, classes)

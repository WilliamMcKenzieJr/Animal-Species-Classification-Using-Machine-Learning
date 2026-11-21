import sys
from pathlib import Path

import torch

from data import IMAGE_EXTENSIONS, build_transform, load_image
from model import load_model
from training import MODEL_PATH, pick_device


@torch.no_grad()
def predict(model, classes, image_path, device="cpu", top=3):
    """Return the top guesses for one image as a list of (class name, probability)."""
    image = load_image(image_path)
    tensor = build_transform()(image)                # same prep as validation: (3, 64, 64)
    batch = tensor.unsqueeze(0).to(device)           # the model expects a batch: (1, 3, 64, 64)

    probabilities = torch.softmax(model(batch), dim=1)[0]
    best = probabilities.topk(min(top, len(classes)))
    return [(classes[i], p.item()) for p, i in zip(best.values, best.indices)]


def find_images(path):
    """A single image, or every image inside a folder (including subfolders)."""
    path = Path(path)
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    return [path]


def main(paths, model_path=MODEL_PATH):
    if not Path(model_path).exists():
        print(f"No trained model found at {model_path}. Run training.py first.")
        return

    device = pick_device()
    model, classes = load_model(model_path, device)

    for path in paths:
        for image_path in find_images(path):
            try:
                guesses = predict(model, classes, image_path, device)
            except Exception as error:
                print(f"{image_path.name}: could not read ({error})")
                continue

            name, confidence = guesses[0]
            others = ", ".join(f"{n} {p:.0%}" for n, p in guesses[1:])
            print(f"{image_path.name}\n  -> {name} ({confidence:.0%})   also: {others}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image or folder> [more images or folders...]")
        print('Example: python predict.py "C:\\Users\\you\\Pictures\\fox.jpg"')
    else:
        main(sys.argv[1:])

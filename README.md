# Animal Species Classification Using Machine Learning

A convolutional neural network built **from scratch in PyTorch** that identifies 11 kinds of animals in photos, including four look-alike canines (coyote, wolf, fox and dog). No pretrained weights are used in the model; every weight was learned from a dataset I collected, filtered and curated myself.

**71% validation accuracy across 11 classes** (random guessing: 9%) with a 1.2M-parameter network trained on a CPU.



---

## Results

| Metric | Value |
|---|---|
| Validation accuracy | **71%** (157 / 220 photos) |
| Training accuracy | 79% |
| Classes | 11 |
| Training images | 7,370 |
| Validation images | 220 (20 per class) |
| Model size | 1,177,003 parameters (~5 MB) |

### Accuracy per class

| Class | Correct | Most common mistake |
|---|---|---|
| Deer | 18 / 20 | — |
| Raccoon | 17 / 20 | — |
| Bird | 16 / 20 | — |
| Squirrel | 16 / 20 | Bird |
| Bear | 15 / 20 | Fox, deer |
| Dog | 14 / 20 | — |
| Fox | 13 / 20 | Coyote, wolf |
| Wolf | 13 / 20 | Coyote |
| Coyote | 12 / 20 | Wolf |
| House cat | 12 / 20 | Wild cat |
| Wild cat (bobcat, lynx, cougar) | 11 / 20 | Deer |

The model does best on animals with distinctive markings (raccoon masks, deer, birds) and worst on species that differ only in subtle shape details, such as coyote vs. wolf and house cat vs. bobcat.



---

## How It Works

### Model architecture

```
Input (3 x 96 x 96)
 -> Block 1: [Conv3x3 -> BatchNorm -> ReLU] x2 -> MaxPool    32 channels
 -> Block 2: [Conv3x3 -> BatchNorm -> ReLU] x2 -> MaxPool    64 channels
 -> Block 3: [Conv3x3 -> BatchNorm -> ReLU] x2 -> MaxPool   128 channels
 -> Block 4: [Conv3x3 -> BatchNorm -> ReLU] x2 -> MaxPool   256 channels
 -> Global average pooling -> Dropout (0.1) -> Linear (11 classes)
```

### Training setup

| Setting | Value | Why |
|---|---|---|
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 | Weight decay discourages memorising |
| Schedule | Cosine annealing over 80 epochs | Large steps early, fine-tuning at the end |
| Loss | Cross-entropy, class-weighted, label smoothing 0.1 | Handles uneven class sizes; avoids overconfident wrong answers |
| Augmentation | Random crop (60-100%), flip, rotation, colour jitter, random greyscale | Forces the model to learn shape instead of colour or background |
| Checkpointing | Saves the epoch with the lowest validation loss | Loss is more stable than accuracy on a small validation set |
| Hardware | CPU only, ~7.5 minutes per epoch (~10 hours total) | |

---

## Dataset

I built the dataset myself rather than using an existing one.

- **Sources:** openly licensed photos (CC0, CC BY, CC BY-SA) from [iNaturalist](https://www.inaturalist.org) and [Wikimedia Commons](https://commons.wikimedia.org), plus an initial hand-collected set.
- **Automated filtering:** before human review, each candidate photo passed a filter that rejected images where the animal was too small, a person was in the shot, the photo was a near-duplicate, or the photo looked more like a different class. Pretrained models (an object detector and an image-feature model) were used **only for this filtering step**, never in the classifier itself.
- **Diversity limits:** a maximum of 3 photos per photographer per class, and caps per species, so no single source dominates a class.
- **Manual review:** every photo was then reviewed by hand before being added.
- **Credits:** every downloaded photo is credited in [`credits/`](credits/). The images themselves are not included in this repository because of licence terms.

---

## What I Learned

The most valuable part of this project was diagnosing *why* the model performed the way it did and fixing the cause rather than guessing.

| Run | Change | Train acc | Val acc | Diagnosis |
|---|---|---|---|---|
| 1 | Baseline: small CNN, 7 classes | 48% | 58% | **Underfitting.** The model was too small to learn the task |
| 2 | Two convolutions per block | 58% | 60% | Better, but merged "canine" and "feline" classes were too broad |
| 3 | Split into 11 specific classes, 90 epochs | 81% | 56% | **Overfitting.** Memorising colour and background shortcuts |
| 4 | Stronger regularisation + 3x more data | 56% | 59% | **Underfitting again.** Too much regularisation for the larger dataset |
| 5 | Balanced regularisation, 80 epochs | 79% | **71%** | Healthy gap. Learning and generalising together |

*Runs 1-2 used a smaller 7-class setup and an 88-photo validation set, so they are not directly comparable with runs 4-5.*

Other lessons:

- **Watch for dataset shortcuts.** I found 75% of my new cougar photos were black-and-white trail-camera images, compared with 1-17% in every other class. The model could have learned "black-and-white means wild cat". I capped greyscale cougar photos at 17% to match the other classes.
- **A bigger validation set matters.** Growing validation from 8 to 20 photos per class cut the effect of a single photo from ~1.1% to ~0.5%, making results trustworthy.
- **Training vs. validation accuracy is the key diagnostic.** Comparing the two told me whether to add capacity (underfitting) or regularisation (overfitting) at each step.

---

## Getting Started

### Requirements

- Python 3.10+

```bash
git clone https://github.com/WilliamMcKenzieJr/Animal-Species-Classification-Using-Machine-Learning.git
cd Animal-Species-Classification-Using-Machine-Learning
python -m pip install -r requirements.txt
```

### Predicting with the trained model

The trained model (`animal_model.pt`) is included, so you can classify your own photos straight away:

```bash
python predict.py path/to/photo.jpg
python predict.py path/to/folder_of_photos
```

Example output:

```
coyote_example.jpg
  -> coyote (97%)   also: fox 2%, wolf 1%
```

### Training your own model

Put your images in one folder per class:

```
train/
  bear/
  bird/
  ...
validate/
  bear/
  bird/
  ...
```

Then run:

```bash
python training.py
```

The best model is saved to `animal_model.pt`, and per-epoch results are saved to `training_log.csv`.

---

## Limitations

- **Look-alike species** are the main source of errors, especially coyote vs. wolf vs. fox, and house cat vs. bobcat.
- **Cougars** are sometimes mistaken for deer, as both are tan animals often photographed in brown brush.
- At 96x96 pixels, fine details such as ear shape and snout length are partly lost.
- Validation uses 220 photos, so results can vary by a few percent between runs.

## Future Improvements

- [ ] Add more colour cougar photos to fix the cougar/deer confusion
- [ ] Train at 128x128 resolution to keep finer details
- [ ] Widen the later convolution blocks for more capacity
- [ ] Add a confusion matrix plot to the results
- [ ] Compare against a pretrained model (transfer learning) as a benchmark

---

## Project Structure

```
├── data.py              # Dataset loading and image augmentation
├── model.py             # CNN architecture, save/load helpers
├── training.py          # Training loop, validation, checkpointing
├── predict.py           # Classify new images with the trained model
├── animal_model.pt      # Trained model weights
├── training_log.csv     # Loss and accuracy for every epoch
├── credits/             # Photographer and licence credits for every photo
└── requirements.txt
```

## Built With

Python · PyTorch · torchvision · Pillow · NumPy · Matplotlib

## License

The code is available under the MIT License. Photos used for training belong to their photographers and are credited in [`credits/`](credits/) under their original licences.

## Author

**William McKenzie** · [LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/WilliamMcKenzieJr)

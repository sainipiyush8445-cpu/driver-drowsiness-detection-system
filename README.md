# Driver Drowsiness Detection System

A real-time driver drowsiness detection system that combines a fine-tuned **MobileNetV2** binary classifier with **MediaPipe FaceMesh** for face localisation. The classifier is trained on the [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) (~41k labelled images of drowsy vs non-drowsy drivers) and runs inference frame-by-frame on a webcam stream, raising an audible alert when drowsiness is sustained over a configurable number of frames.

The original training pipeline is preserved in [`drowsiness_detection.ipynb`](./drowsiness_detection.ipynb) (designed for Google Colab). The same logic has been refactored into reusable Python modules with a CLI entry point so it can train, predict, and run on a webcam locally.

---

## Results

| Epoch | Train acc | Val acc | Val loss |
|-------|-----------|---------|----------|
| 1     | 93.3%     | 98.3%   | 0.054    |
| 5     | 98.8%     | 99.6%   | 0.013    |
| 10    | 99.1%     | 99.7%   | 0.010    |
| 15    | 99.2%     | **99.7%** | **0.008** |

Final validation accuracy of **~99.7%** on the DDD dataset (8,358-image validation split) after 15 epochs of training the head only (MobileNetV2 backbone frozen).

---

## How it works

1. **Face localisation** — Each webcam frame is passed through MediaPipe FaceMesh (468 landmarks). A padded bounding box around all landmarks is cropped out as the face region.
2. **Classification** — The face crop is resized to 96×96, normalised to `[0, 1]`, and run through MobileNetV2 → GlobalAveragePooling → Dense(128) → Dropout(0.5) → Dense(1, sigmoid). Output > 0.5 means **awake**, ≤ 0.5 means **drowsy**.
3. **Temporal smoothing** — A drowsy prediction is only treated as a real alert after it persists for `--consec-frames` consecutive frames (default 20). This filters out blinks and brief glances.
4. **Alarm** — Once the threshold is crossed, the system overlays a red warning banner and emits a terminal beep on every frame until the driver becomes alert again.

---

## Project structure

```
driver-drowsiness-detection-system/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                       # CLI: train / predict / webcam
├── model.py                      # MobileNetV2 architecture + training loop
├── utils.py                      # Preprocessing, FaceMesh helpers, prediction
├── drowsiness_detection.ipynb    # Original Colab training notebook
├── research_paper.pdf            # Project write-up (add manually)
├── images/                       # Screenshots / demo media for README
└── data/                         # DDD dataset goes here (gitignored)
    ├── open/                     # Non-Drowsy class
    └── closed/                   # Drowsy class
```

---

## Setup

```bash
git clone https://github.com/<your-username>/driver-drowsiness-detection-system.git
cd driver-drowsiness-detection-system

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **Python version note:** MediaPipe is fussy about Python versions. As of this writing, Python 3.9–3.11 is the safe range. TensorFlow 2.10+ also works on those versions.

---

## Usage

### 1. Prepare the dataset

Download [DDD from Kaggle](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) and arrange it like this:

```
data/
├── open/      <- Non Drowsy images go here
└── closed/    <- Drowsy images go here
```

The notebook contains a copy block (Cell 5) that does this restructuring automatically if you have the original Kaggle layout.

### 2. Train

```bash
python main.py train --data ./data --epochs 15 --save ./models/drowsiness_model.keras
```

On a Colab T4 GPU each epoch takes ~9–20 minutes for the full 41k-image dataset; total training is roughly 2–4 hours. Reduce `--epochs` for a quick smoke test.

### 3. Predict on a single image

```bash
python main.py predict --image path/to/face.jpg \
                       --model ./models/drowsiness_model.keras \
                       --show
```

Prints the sigmoid score and class label, and (with `--show`) opens a matplotlib window with the image and predicted label.

### 4. Real-time webcam detection

```bash
python main.py webcam --model ./models/drowsiness_model.keras --consec-frames 20
```

Press `q` to quit. The overlay shows the live label, FPS, and a red banner with a terminal beep when drowsiness is sustained beyond the threshold. Lower `--consec-frames` for a more sensitive alarm; raise it to suppress false positives from ordinary blinking.

---

## Hyper-parameters & design choices

| Parameter | Value | Why |
|-----------|-------|------|
| Input size | 96×96 | MobileNetV2's smallest standard input — keeps webcam inference fast |
| Backbone | MobileNetV2 | Mobile-friendly, ImageNet-pretrained, ~3.5M params |
| Frozen base | Yes | Head-only training is fast and avoids overfitting on a small task |
| Optimiser | Adam (default LR) | Works well out of the box for transfer learning |
| Loss | Binary crossentropy | Two classes (drowsy / not drowsy) |
| Threshold | 0.5 | Standard for sigmoid output; tune if precision/recall trade-off matters |
| Consec frames | 20 | At ~25–30 FPS this is roughly 0.7 s of sustained eye closure |

---

## Future work

- Fine-tune the MobileNetV2 backbone (unfreeze last few blocks) for an extra 0.1–0.3% accuracy.
- Replace the full-face crop with explicit per-eye crops using FaceMesh's eye landmarks for more interpretable alerts.
- Compute Eye Aspect Ratio (EAR) from FaceMesh landmarks as a complementary signal — a classic, lightweight drowsiness cue.
- Export to TFLite for on-device inference (Raspberry Pi, Android) with int8 quantisation.
- Replace the terminal beep with a louder sound file and integrate with a vehicle's CAN bus for production deployment.

---

## Dataset

[Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) by Ismail Nasri — 41,793 images split into `Drowsy` and `Non Drowsy` classes.

---

## License

MIT — feel free to use, modify, and distribute.

"""
Driver Drowsiness Detection System — command-line entry point.

Three subcommands:

    # 1. Train the classifier (DDD dataset arranged as data/{open,closed})
    python main.py train --data ./data --epochs 15 \\
                         --save ./models/drowsiness_model.keras

    # 2. Run inference on a single image
    python main.py predict --image path/to/face.jpg \\
                           --model ./models/drowsiness_model.keras

    # 3. Real-time webcam drowsiness detection (uses MediaPipe FaceMesh)
    python main.py webcam --model ./models/drowsiness_model.keras \\
                          --consec-frames 20
"""

import argparse
import os
import time
from collections import deque

import cv2
import numpy as np

from utils import (
    predict_from_path, predict_image, preprocess_image,
    get_face_mesh, extract_face_region, play_alarm,
    IMG_SIZE, DROWSY_THRESHOLD,
)


# ── Train ──────────────────────────────────────────────────────────

def cmd_train(args):
    from model import train
    train(
        data_dir=args.data,
        save_path=args.save,
        epochs=args.epochs,
        img_size=IMG_SIZE,
        batch_size=args.batch_size,
    )


# ── Predict on a single image ──────────────────────────────────────

def cmd_predict(args):
    from tensorflow.keras.models import load_model
    model = load_model(args.model)
    score, label = predict_from_path(model, args.image)
    print(f"Image           : {args.image}")
    print(f"Prediction score: {score:.4f}")
    print(f"Result          : {label}")

    if args.show:
        img = cv2.imread(args.image)
        try:
            import matplotlib.pyplot as plt
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(label)
            plt.axis("off")
            plt.show()
        except ImportError:
            cv2.imshow(label, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# ── Real-time webcam ───────────────────────────────────────────────

def cmd_webcam(args):
    from tensorflow.keras.models import load_model

    print("Loading model...")
    model = load_model(args.model)
    face_mesh = get_face_mesh()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    print("Press 'q' to quit.")
    drowsy_streak = 0
    fps_window = deque(maxlen=30)
    last_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        face_crop = extract_face_region(frame, face_mesh)
        label = "NO FACE"
        color = (128, 128, 128)
        score = None

        if face_crop is not None and face_crop.size > 0:
            score = float(model.predict(
                preprocess_image(face_crop), verbose=0
            )[0][0])

            if score > DROWSY_THRESHOLD:
                label = f"AWAKE  ({score:.2f})"
                color = (0, 200, 0)
                drowsy_streak = 0
            else:
                label = f"DROWSY ({score:.2f})"
                color = (0, 0, 255)
                drowsy_streak += 1

        # Trigger alarm if drowsy for N consecutive frames
        if drowsy_streak >= args.consec_frames:
            cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            play_alarm()

        # FPS overlay
        now = time.time()
        fps_window.append(1.0 / max(now - last_t, 1e-6))
        last_t = now
        fps = sum(fps_window) / len(fps_window)

        cv2.putText(frame, label, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (30, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── argparse plumbing ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Driver Drowsiness Detection System",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train the MobileNetV2 classifier.")
    p_train.add_argument("--data", default="./data",
                         help="Dataset root with subfolders 'open' and 'closed'.")
    p_train.add_argument("--save", default="./models/drowsiness_model.keras",
                         help="Path for the saved .keras model file.")
    p_train.add_argument("--epochs", type=int, default=15)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="Predict on a single image.")
    p_pred.add_argument("--image", required=True, help="Path to a face image.")
    p_pred.add_argument("--model", default="./models/drowsiness_model.keras")
    p_pred.add_argument("--show", action="store_true",
                        help="Display the image with the predicted label.")
    p_pred.set_defaults(func=cmd_predict)

    p_cam = sub.add_parser("webcam", help="Real-time webcam drowsiness detection.")
    p_cam.add_argument("--model", default="./models/drowsiness_model.keras")
    p_cam.add_argument("--camera", type=int, default=0,
                       help="OpenCV camera index (default 0).")
    p_cam.add_argument("--consec-frames", type=int, default=20,
                       help="Consecutive drowsy frames before triggering the alarm.")
    p_cam.set_defaults(func=cmd_webcam)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

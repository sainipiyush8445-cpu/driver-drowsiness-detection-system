"""Utility functions: image preprocessing, single-image prediction,
and MediaPipe-based face / eye region extraction for real-time detection.
"""

import os
import cv2
import numpy as np

IMG_SIZE = 96
DROWSY_THRESHOLD = 0.5  # pred > 0.5 -> OPEN (not drowsy)


# ── Image preprocessing ────────────────────────────────────────────

def preprocess_image(img: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """Resize, normalise to [0, 1], add batch dimension."""
    img_resized = cv2.resize(img, (img_size, img_size))
    img_resized = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(img_resized, axis=0)


def predict_image(model, img: np.ndarray, img_size: int = IMG_SIZE) -> tuple:
    """
    Run the model on a single BGR image (numpy array as returned by cv2.imread).

    Returns
    -------
    score : float  - sigmoid output in [0, 1]; > 0.5 means OPEN/not drowsy.
    label : str    - human-readable label.
    """
    if img is None:
        raise ValueError("Image is None — check the input path or frame.")
    batch = preprocess_image(img, img_size)
    score = float(model.predict(batch, verbose=0)[0][0])
    label = "OPEN EYES (NOT DROWSY)" if score > DROWSY_THRESHOLD else "CLOSED EYES (DROWSY)"
    return score, label


def predict_from_path(model, image_path: str, img_size: int = IMG_SIZE) -> tuple:
    """Convenience wrapper: read from disk then call predict_image()."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    return predict_image(model, img, img_size)


# ── MediaPipe face / eye region extraction ─────────────────────────
#
# MediaPipe FaceMesh returns 468 landmarks. We use the standard left and
# right eye landmark indices to compute a tight bounding box around each
# eye, which is then fed to the classifier. This makes the system robust
# to head movement and varying camera distances.

# Left & right eye landmark indices from MediaPipe FaceMesh
LEFT_EYE_IDX  = [33, 133, 160, 158, 153, 144, 145, 153, 154, 155, 173, 246]
RIGHT_EYE_IDX = [362, 263, 387, 385, 380, 373, 374, 380, 381, 382, 398, 466]


def get_face_mesh():
    """Lazy-construct a MediaPipe FaceMesh instance (single-face)."""
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _eye_bbox(landmarks, indices, frame_w: int, frame_h: int, pad: int = 10):
    xs = [int(landmarks[i].x * frame_w) for i in indices]
    ys = [int(landmarks[i].y * frame_h) for i in indices]
    x_min, x_max = max(min(xs) - pad, 0), min(max(xs) + pad, frame_w)
    y_min, y_max = max(min(ys) - pad, 0), min(max(ys) + pad, frame_h)
    return x_min, y_min, x_max, y_max


def extract_face_region(frame: np.ndarray, face_mesh) -> np.ndarray:
    """
    Run FaceMesh on a BGR frame and return a cropped face region suitable
    for the classifier. Returns None if no face is detected.

    Notes
    -----
    The training data (DDD) is full-face crops labelled drowsy/non-drowsy,
    so we feed a generous face crop rather than per-eye crops here.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    pad = 20
    x_min = max(min(xs) - pad, 0)
    x_max = min(max(xs) + pad, w)
    y_min = max(min(ys) - pad, 0)
    y_max = min(max(ys) + pad, h)
    return frame[y_min:y_max, x_min:x_max]


# ── Audible alarm ──────────────────────────────────────────────────

def play_alarm():
    """Cross-platform terminal beep used to alert the driver."""
    print("\a", end="", flush=True)

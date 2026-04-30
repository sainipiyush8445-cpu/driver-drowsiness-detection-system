"""MobileNetV2-based binary classifier for drowsiness detection.

Architecture (from drowsiness_detection.ipynb, Cell 7):
    MobileNetV2 (ImageNet, frozen)
      -> GlobalAveragePooling2D
      -> Dense(128, relu)
      -> Dropout(0.5)
      -> Dense(1, sigmoid)

Output: P(open eyes / not drowsy).
    pred > 0.5  -> OPEN  (not drowsy)
    pred <= 0.5 -> CLOSED (drowsy)
"""

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, Model

IMG_SIZE = 96


def build_model(img_size: int = IMG_SIZE, freeze_base: bool = True) -> Model:
    """Build and compile the drowsiness classifier."""
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = not freeze_base

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_datasets(data_dir: str, img_size: int = IMG_SIZE, batch_size: int = 32):
    """Load train/val splits from a directory laid out as data_dir/{open,closed}/*.jpg."""
    import tensorflow as tf

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )
    return train_ds, val_ds


def train(data_dir: str, save_path: str, epochs: int = 15,
          img_size: int = IMG_SIZE, batch_size: int = 32):
    """End-to-end training: load data, build model, fit, save."""
    import os

    train_ds, val_ds = load_datasets(data_dir, img_size, batch_size)
    model = build_model(img_size=img_size, freeze_base=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    return model, history

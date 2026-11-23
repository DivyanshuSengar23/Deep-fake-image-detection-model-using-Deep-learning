"""
Train the DeepFake CNN model for the Streamlit app.

Expected folders:
split_dataset/
    train/
        fake/
        real/
    val/
        fake/
        real/
    test/
        fake/
        real/

The model is saved to:
tmp_checkpoint/best_model.keras

IMPORTANT:
- Your current dataset has only 8 train, 1 validation, and 2 test images
  per class. This is only enough to test the training pipeline, not to
  build a reliable deepfake detector.
- DirectoryIterator assigns labels alphabetically: fake=0, real=1.
  Therefore the sigmoid output is P(REAL), matching the app.py expectation.
"""

import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

try:
    from efficientnet.tfkeras import EfficientNetB0
except ImportError as exc:
    raise SystemExit(
        "EfficientNet package missing. Install it with: "
        "python -m pip install efficientnet"
    ) from exc


# -------------------- Settings --------------------
DATASET_PATH = Path("split_dataset")
CHECKPOINT_DIR = Path("tmp_checkpoint")
MODEL_PATH = CHECKPOINT_DIR / "best_model.keras"

IMAGE_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 1e-4
SEED = 42

tf.keras.utils.set_random_seed(SEED)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------- Verify dataset --------------------
required_dirs = [
    DATASET_PATH / "train" / "real",
    DATASET_PATH / "train" / "fake",
    DATASET_PATH / "val" / "real",
    DATASET_PATH / "val" / "fake",
]

for folder in required_dirs:
    if not folder.is_dir():
        raise SystemExit(f"Required dataset folder not found: {folder}")

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def count_images(folder):
    return sum(
        1 for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    )

print("\nDataset image counts:")
for split in ("train", "val", "test"):
    for label in ("real", "fake"):
        folder = DATASET_PATH / split / label
        if folder.is_dir():
            print(f"{split:5s} / {label:4s}: {count_images(folder)}")


# -------------------- Data generators --------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    str(DATASET_PATH / "train"),
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    color_mode="rgb",
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_generator = valid_datagen.flow_from_directory(
    str(DATASET_PATH / "val"),
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    color_mode="rgb",
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# This mapping is essential: sigmoid output must mean P(REAL).
print("\nClass mapping:", train_generator.class_indices)
if train_generator.class_indices != {"fake": 0, "real": 1}:
    raise SystemExit(
        "Unexpected class mapping. Expected {'fake': 0, 'real': 1}. "
        "Check the train folder names."
    )
if val_generator.class_indices != {"fake": 0, "real": 1}:
    raise SystemExit(
        "Train/validation class mappings differ. Check folder names."
    )

if train_generator.samples < 4 or val_generator.samples < 2:
    raise SystemExit("Not enough training/validation images to start training.")


# -------------------- Build model --------------------
# Keep the same EfficientNet package family and preprocessing convention
# used in your original 03-train_cnn.py (ImageDataGenerator rescale=1/255).
backbone = EfficientNetB0(
    weights="imagenet",
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    pooling="max",
)

# With the tiny current dataset, freeze the pretrained backbone to reduce
# overfitting. A larger dataset may justify a separate fine-tuning stage.
backbone.trainable = False

model = models.Sequential(
    [
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        backbone,
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid", name="real_probability"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# -------------------- Train and save --------------------
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    ),
]

print(f"\nTraining starts. Best model will be saved to: {MODEL_PATH}")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
)

# Save the final in-memory model too, in case no checkpoint was written.
if not MODEL_PATH.exists():
    model.save(MODEL_PATH)

print("\nTraining finished.")
print(f"Model saved at: {MODEL_PATH.resolve()}")
print("Output meaning: sigmoid score = probability of REAL (subject to model quality).")
print("Reminder: evaluate on a large, independent test set before trusting predictions.")

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

Model output:
tmp_checkpoint/best_model.keras

Class mapping:
fake = 0
real = 1

Therefore sigmoid output = P(REAL).
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
except ImportError:
    raise SystemExit(
        "EfficientNet package missing. Install using:\n"
        "python -m pip install efficientnet"
    )


# -------------------- SETTINGS --------------------

DATASET_PATH = Path("split_dataset")
CHECKPOINT_DIR = Path("tmp_checkpoint")
MODEL_PATH = CHECKPOINT_DIR / "best_model.keras"

IMAGE_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 0.0001
SEED = 42

tf.keras.utils.set_random_seed(SEED)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------- VERIFY DATASET --------------------

required_dirs = [
    DATASET_PATH / "train" / "real",
    DATASET_PATH / "train" / "fake",
    DATASET_PATH / "val" / "real",
    DATASET_PATH / "val" / "fake",
]

for folder in required_dirs:
    if not folder.is_dir():
        raise SystemExit(
            f"Dataset folder not found: {folder}"
        )


image_extensions = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def count_images(folder):
    return sum(
        1
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in image_extensions
    )


print("\nDataset image counts:")

for split in ("train", "val", "test"):
    for label in ("real", "fake"):

        folder = DATASET_PATH / split / label

        if folder.is_dir():
            print(
                f"{split:5s} / {label:4s}: "
                f"{count_images(folder)}"
            )


# -------------------- DATA GENERATORS --------------------

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

valid_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)


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


# -------------------- CHECK CLASS MAPPING --------------------

print("\nClass mapping:", train_generator.class_indices)

expected_mapping = {
    "fake": 0,
    "real": 1
}

if train_generator.class_indices != expected_mapping:
    raise SystemExit(
        "Unexpected training class mapping."
    )

if val_generator.class_indices != expected_mapping:
    raise SystemExit(
        "Unexpected validation class mapping."
    )

if train_generator.samples < 4:
    raise SystemExit("Not enough training images.")

if val_generator.samples < 2:
    raise SystemExit("Not enough validation images.")


# -------------------- BUILD MODEL --------------------

print("\nBuilding EfficientNetB0 model...")

backbone = EfficientNetB0(
    weights=None,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    pooling="max",
)

# Freeze backbone for initial pipeline testing.
backbone.trainable = False


model = models.Sequential([
    layers.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    ),

    backbone,

    layers.Dense(
        512,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dense(
        1,
        activation="sigmoid",
        name="real_probability"
    )
])


# -------------------- COMPILE MODEL --------------------

model.compile(
    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="binary_crossentropy",

    metrics=["accuracy"]
)


model.summary()


# -------------------- CALLBACKS --------------------

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )

]


# -------------------- TRAIN MODEL --------------------

print("\nTraining started...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)


# -------------------- SAVE MODEL --------------------

if not MODEL_PATH.exists():

    print("\nSaving final model...")

    model.save(MODEL_PATH)


print("\n================================")
print("TRAINING COMPLETED")
print("================================")

print(
    f"Model saved at: {MODEL_PATH.resolve()}"
)

print(
    "Output: sigmoid score = probability of REAL."
)

print(
    "WARNING: Current dataset is too small "
    "for reliable DeepFake detection."
)
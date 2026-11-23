"""
Transfer-learning training script for the DeepFake Streamlit app.

Expected:
split_dataset/
  train/fake, train/real
  val/fake,   val/real

The app currently normalizes images to [0, 1]. This model accepts [0, 1]
and rescales internally to [0, 255] before EfficientNetB0's built-in
preprocessing, so the app's existing /255 preprocessing can remain unchanged.

IMPORTANT: 16 training images and 2 validation images are far too few for
a reliable detector. This script is for pipeline testing only.
"""

from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

DATASET = Path("split_dataset")
OUT_DIR = Path("tmp_checkpoint")
MODEL_PATH = OUT_DIR / "best_model.keras"

IMG_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 12
SEED = 42

tf.keras.utils.set_random_seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

for p in [
    DATASET / "train" / "fake",
    DATASET / "train" / "real",
    DATASET / "val" / "fake",
    DATASET / "val" / "real",
]:
    if not p.is_dir():
        raise SystemExit(f"Missing dataset folder: {p}")

train_aug = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=8,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.08,
    horizontal_flip=True,
    fill_mode="nearest",
)
valid_aug = ImageDataGenerator(rescale=1.0 / 255.0)

train_gen = train_aug.flow_from_directory(
    str(DATASET / "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)
val_gen = valid_aug.flow_from_directory(
    str(DATASET / "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=False,
)

expected = {"fake": 0, "real": 1}
print("Training class mapping:", train_gen.class_indices)
print("Validation class mapping:", val_gen.class_indices)
if train_gen.class_indices != expected or val_gen.class_indices != expected:
    raise SystemExit(f"Expected class mapping {expected}; check folder names.")
if train_gen.samples < 4 or val_gen.samples < 2:
    raise SystemExit("Not enough training or validation images.")

print("\nLoading ImageNet-pretrained EfficientNetB0...")
backbone = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="max",
)
backbone.trainable = False

# Input to this model is [0,1], matching webapp.py's current preprocessing.
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="normalized_rgb_image")
x = layers.Rescaling(255.0, name="restore_0_255")(inputs)
x = backbone(x, training=False)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation="sigmoid", name="real_probability")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    ModelCheckpoint(
        str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
]

print("\nStarting transfer-learning head training...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Ensure the app has a model even if no checkpoint callback was triggered.
if not MODEL_PATH.exists():
    model.save(MODEL_PATH)

print(f"\nSaved model: {MODEL_PATH.resolve()}")
print("Output meaning: sigmoid score is P(REAL), because fake=0 and real=1.")
print("WARNING: Do not trust predictions until trained/evaluated on a large,")
print("independent dataset with source videos separated across splits.")

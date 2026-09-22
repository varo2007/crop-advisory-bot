import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================
# PATHS
# ============================================================

DATASET_DIR = r"C:\Users\ragup\Downloads\crop advisory bot\datasets\plant_disease\archive (1)\PlantVillage"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "disease_model.keras")
CLASS_NAMES_PATH = "class_names.json"

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# First stage + fine tuning
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 3

# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(TRAIN_DIR):
    print("ERROR: Training folder not found:")
    print(TRAIN_DIR)
    print("\nCheck your PlantVillage folder structure.")
    exit()

if not os.path.exists(VAL_DIR):
    print("ERROR: Validation folder not found:")
    print(VAL_DIR)
    exit()

print("\n========================================")
print("PLANT DISEASE MODEL TRAINING")
print("========================================")

print("\nTraining folder:")
print(TRAIN_DIR)

print("\nValidation folder:")
print(VAL_DIR)

# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names

print("\n========================================")
print("CLASSES FOUND:", len(class_names))
print("========================================")

for i, name in enumerate(class_names):
    print(i, "->", name)

if len(class_names) != 38:
    print("\nWARNING!")
    print("Expected 38 classes but found:", len(class_names))
    print("Check your dataset structure before continuing.")

# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=4)

print("\nClass names saved to:")
print(CLASS_NAMES_PATH)

# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
], name="data_augmentation")

# ============================================================
# MOBILE NET V2
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze base model initially
base_model.trainable = False

# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.Dropout(0.20)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n========================================")
print("STAGE 1 TRAINING")
print("========================================")

model.summary()

# ============================================================
# STAGE 1
# ============================================================

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS
)

# ============================================================
# FINE TUNING
# ============================================================

print("\n========================================")
print("STAGE 2 - FINE TUNING")
print("========================================")

base_model.trainable = True

# Freeze the first 100 layers
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=INITIAL_EPOCHS
)

# ============================================================
# SAVE MODEL
# ============================================================

print("\n========================================")
print("SAVING MODEL")
print("========================================")

model.save(MODEL_PATH)

print("\nMODEL SAVED SUCCESSFULLY!")
print(MODEL_PATH)

# ============================================================
# FINAL VALIDATION
# ============================================================

loss, accuracy = model.evaluate(
    val_ds,
    verbose=1
)

print("\n========================================")
print("FINAL RESULTS")
print("========================================")

print("Validation accuracy:",
      round(accuracy * 100, 2), "%")

print("Validation loss:",
      round(loss, 4))

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print("\nCreated files:")

print("1.", MODEL_PATH)
print("2.", CLASS_NAMES_PATH)

print("\nYou can now run:")
print("python -m streamlit run app.py")
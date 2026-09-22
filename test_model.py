import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/disease_model.keras"
CLASS_NAMES_PATH = "class_names.json"

DATASET = r"datasets\plant_disease\archive (1)\PlantVillage"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("\nMODEL TEST")
print("=" * 60)
print("Number of classes:", len(class_names))

# Find folders that actually contain images
image_folders = []

for root, dirs, files in os.walk(DATASET):

    images = [
        f for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if images:
        image_folders.append((root, images[0]))

print("Image folders found:", len(image_folders))

tested = 0

for folder, image_file in image_folders:

    image_path = os.path.join(folder, image_file)

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    img = np.array(image, dtype=np.float32)

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0]

    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    actual_class = os.path.basename(folder)

    print("\nActual class :", actual_class)
    print("Predicted    :", predicted_class)
    print("Confidence   :", f"{confidence:.2f}%")

    tested += 1

    if tested >= 10:
        break

print("\n" + "=" * 60)
print("TEST COMPLETE")
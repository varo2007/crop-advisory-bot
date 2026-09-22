import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

MODEL_PATH = "models/disease_model.keras"
CLASS_NAMES_PATH = "class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)


def predict_disease(image):
    if isinstance(image, str):
        image = Image.open(image)

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])

    print("\n==============================")
    print("CLASS NAMES:", CLASS_NAMES)
    print("PREDICTIONS:", predictions)
    print("PREDICTED INDEX:", predicted_index)
    print("PREDICTED CLASS:", CLASS_NAMES[predicted_index])
    print("CONFIDENCE:", confidence)
    print("==============================")

    return {
        "disease": CLASS_NAMES[predicted_index],
        "confidence": confidence,
        "all_predictions": predictions.tolist()
    }
import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR /
    "models" /
    "crop_recommendation_model.pkl"
)


def load_crop_model():

    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


def predict_crop(
    nitrogen,
    phosphorus,
    potassium,
    temperature,
    humidity,
    ph,
    rainfall
):

    model = load_crop_model()

    if model is None:
        return None

    input_data = pd.DataFrame([
        {
            "n": nitrogen,
            "p": phosphorus,
            "k": potassium,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }
    ])

    prediction = model.predict(input_data)

    return prediction[0]
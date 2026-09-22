CROP ADVISORY BOT - STAGE 3

Files:
- app.py
- Crop_recommendation.csv  (copy your existing dataset here)
- farm_ai_background.png
- requirements.txt
- .streamlit/secrets.toml.example

1) Put app.py, Crop_recommendation.csv and farm_ai_background.png
   in your existing:
   C:\Users\ragup\Downloads\crop advisory bot

2) Install:
   python -m pip install -r requirements.txt

3) Run:
   python -m streamlit run app.py

LOGIN:
   Demo username: farmer
   Demo password: 1234

LOCATION:
   Try:
   - Puduvoyal, Tamil Nadu
   - Thiruvallur, Tamil Nadu
   - Sholinghur, Tamil Nadu
   - Ponneri, Tamil Nadu

GOOGLE MAP:
   Create .streamlit\secrets.toml using the example.
   Add GOOGLE_MAPS_API_KEY.
   The Google Maps Embed API requires a Google Cloud API key.

AI IMAGE CHATBOT:
   Add OPENAI_API_KEY to secrets.toml.
   Without the key, image upload still works in the UI, but
   AI vision analysis is disabled.

IMPORTANT:
   The Random Forest model remains based on the original dataset.
   Current precipitation is NOT automatically substituted for the
   dataset's rainfall feature because those measurements have
   different meanings.

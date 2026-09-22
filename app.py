import streamlit as st
import pandas as pd
import requests
import sqlite3
import hashlib
import base64
import html

from datetime import datetime
from pathlib import Path
from textwrap import dedent
from PIL import Image


# ============================================================
# OPTIONAL MODEL IMPORTS
# ============================================================

try:
    from services.disease_model import predict_disease
    DISEASE_MODEL_IMPORT_ERROR = None
except Exception as error:
    predict_disease = None
    DISEASE_MODEL_IMPORT_ERROR = str(error)


try:
    from services.crop_model import predict_crop
    CROP_MODEL_IMPORT_ERROR = None
except Exception as error:
    predict_crop = None
    CROP_MODEL_IMPORT_ERROR = str(error)


try:
    from services.agrigpt import agrigpt_response
    AGRIGPT_IMPORT_ERROR = None
except Exception as error:
    agrigpt_response = None
    AGRIGPT_IMPORT_ERROR = str(error)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgriGPT - AI Crop Advisory Bot",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "agrigpt.db"
BACKGROUND_PATH = BASE_DIR / "assets" / "agri_background.png"


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def get_connection():
    return sqlite3.connect(
        str(DB_PATH),
        check_same_thread=False
    )


def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def hash_password(password):
    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


def register_user(username, password):
    username = username.strip()

    if not username:
        return False, "Username cannot be empty."

    if len(username) < 3:
        return False, "Username must contain at least 3 characters."

    if len(password) < 4:
        return False, "Password must contain at least 4 characters."

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO users(username, password, created_at)
            VALUES (?, ?, ?)
        """, (
            username,
            hash_password(password),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()

        return True, "Registration successful."

    except sqlite3.IntegrityError:
        return False, "Username already exists."

    finally:
        conn.close()


def login_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT username
        FROM users
        WHERE username = ? AND password = ?
    """, (
        username.strip(),
        hash_password(password)
    ))

    result = cursor.fetchone()

    conn.close()

    return result is not None


def save_chat(username, question, answer):
    if not username or not question or not answer:
        return

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO chat_history(username, question, answer, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        username,
        question,
        answer,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_chat_history(username):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT question, answer, created_at
        FROM chat_history
        WHERE username = ?
        ORDER BY id DESC
    """, (username,))

    rows = cursor.fetchall()

    conn.close()

    return rows


initialize_database()


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_SESSION_VALUES = {
    "logged_in": False,
    "username": "",
    "page": "Dashboard",
    "chat_messages": [],
    "weather_data": None,
    "location_name": "",
    "latitude": None,
    "longitude": None
}

for key, value in DEFAULT_SESSION_VALUES.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# BACKGROUND IMAGE
# ============================================================

def get_background_base64():

    if not BACKGROUND_PATH.exists():
        return None

    try:

        with open(BACKGROUND_PATH, "rb") as file:

            return base64.b64encode(
                file.read()
            ).decode("utf-8")

    except Exception:

        return None


# ============================================================
# CSS
# ============================================================

def load_css():

    background_image = get_background_base64()

    if not st.session_state.logged_in and background_image:

        app_background = f"""
        background:
            linear-gradient(
                rgba(4, 28, 16, 0.60),
                rgba(4, 28, 16, 0.60)
            ),
            url("data:image/png;base64,{background_image}")
            center center / cover fixed no-repeat;
        """

    else:

        app_background = """
        background: #f5f8f3;
        """

    css = f"""
    <style>

    .stApp {{
        {app_background}
        min-height: 100vh;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stToolbar"] {{
        visibility: hidden;
    }}

    [data-testid="stSidebar"] {{
        background-color: #1f4d35;
    }}

    [data-testid="stSidebar"] * {{
        color: white !important;
    }}

    h1, h2, h3 {{
        color: #1f4d35 !important;
    }}

    p, span, label {{
        color: #222222;
    }}

    .stMarkdown p {{
        color: #222222;
    }}

    .metric-card {{
        padding: 22px;
        border-radius: 18px;
        background-color: white;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        min-height: 150px;
        border: 1px solid #e4eee5;
    }}

    .metric-card h2 {{
        font-size: 38px;
        margin-bottom: 8px;
    }}

    .metric-card b {{
        font-size: 17px;
        color: #1f4d35 !important;
    }}

    .metric-card p {{
        color: #66756b !important;
        font-size: 13px;
    }}

    .chat-user {{
        background-color: #d9f0df;
        padding: 14px;
        border-radius: 12px;
        margin: 10px 0;
        color: #173d28 !important;
    }}

    .chat-user * {{
        color: #173d28 !important;
    }}

    .chat-bot {{
        background-color: white;
        padding: 14px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 4px solid #2e8b57;
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
        color: #222222 !important;
    }}

    .chat-bot * {{
        color: #222222 !important;
    }}

    .login-panel {{
        max-width: 650px;
        margin: 7vh auto 20px auto;
        padding: 34px;
        border-radius: 26px;
        background: rgba(255,255,255,0.96);
        box-shadow: 0 20px 60px rgba(0,0,0,0.30);
        text-align: center;
    }}

    .login-logo {{
        width: 88px;
        height: 88px;
        margin: auto;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #e8f5e9;
        font-size: 46px;
    }}

    .login-title {{
        color: #1f4d35 !important;
        font-size: 38px;
        font-weight: 800;
        margin-top: 14px;
    }}

    .login-subtitle {{
        color: #607568 !important;
        font-size: 16px;
        margin-top: 5px;
        margin-bottom: 20px;
    }}

    .feature-row {{
        display: flex;
        justify-content: center;
        gap: 8px;
        flex-wrap: wrap;
    }}

    .feature-pill {{
        background: #edf7ef;
        color: #28623e !important;
        padding: 8px 13px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 600;
    }}

    .login-caption {{
        text-align: center;
        color: white !important;
        font-size: 13px;
        margin-top: 15px;
        text-shadow: 0 1px 5px black;
    }}

    .disease-result {{
        background: #ffffff !important;
        color: #222222 !important;
        padding: 25px !important;
        border-radius: 16px !important;
        border: 2px solid #2e8b57 !important;
        border-left: 6px solid #2e8b57 !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.10) !important;
        margin-top: 15px !important;
        min-height: 130px !important;
        visibility: visible !important;
        opacity: 1 !important;
    }}

    .disease-result h2 {{
        color: #c0392b !important;
        font-size: 28px !important;
        font-weight: 800 !important;
        line-height: 1.4 !important;
        margin: 10px 0 0 0 !important;
        visibility: visible !important;
        opacity: 1 !important;
        word-wrap: break-word !important;
    }}

    .disease-result h3 {{
        color: #1f4d35 !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        margin: 0 !important;
        visibility: visible !important;
        opacity: 1 !important;
    }}

    .confidence-box {{
        background: #e8f5e9 !important;
        color: #155d43 !important;
        padding: 15px !important;
        border-radius: 10px !important;
        margin-top: 15px !important;
        font-size: 20px !important;
        font-weight: bold !important;
        border: 1px solid #a5d6a7 !important;
        visibility: visible !important;
        opacity: 1 !important;
    }}

    .confidence-box * {{
        color: #155d43 !important;
        visibility: visible !important;
        opacity: 1 !important;
    }}

    .stButton > button {{
        border-radius: 10px;
        font-weight: 600;
    }}

    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span {{
        color: #222222 !important;
    }}

    [data-testid="stAlert"] {{
        color: #222222 !important;
    }}

    [data-testid="stAlert"] p {{
        color: #222222 !important;
    }}

    [data-testid="stDataFrame"] {{
        border-radius: 12px;
        overflow: hidden;
    }}

    </style>
    """

    st.markdown(
        dedent(css),
        unsafe_allow_html=True
    )


load_css()


# ============================================================
# WEATHER FUNCTIONS
# ============================================================

def geocode_location(location):

    if not location or not location.strip():
        return None

    try:

        url = "https://geocoding-api.open-meteo.com/v1/search"

        params = {
            "name": location.strip(),
            "count": 1,
            "language": "en",
            "format": "json"
        }

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        if not data.get("results"):
            return None

        result = data["results"][0]

        return {
            "name": result.get("name", location),
            "latitude": result.get("latitude"),
            "longitude": result.get("longitude"),
            "country": result.get("country", "")
        }

    except Exception:

        return None


def get_weather(latitude, longitude):

    if latitude is None or longitude is None:
        return None

    try:

        url = "https://api.open-meteo.com/v1/forecast"

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "apparent_temperature,"
                "rain,"
                "precipitation,"
                "wind_speed_10m,"
                "weather_code"
            ),
            "daily": (
                "temperature_2m_max,"
                "temperature_2m_min,"
                "rain_sum,"
                "precipitation_probability_max"
            ),
            "timezone": "auto",
            "forecast_days": 7
        }

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        return response.json()

    except Exception:

        return None


def weather_description(code):

    codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Light rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Light snow",
        73: "Moderate snow",
        75: "Heavy snow",
        80: "Rain showers",
        81: "Moderate rain showers",
        82: "Heavy rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm with hail",
        99: "Heavy thunderstorm with hail"
    }

    try:
        code = int(code)
    except (TypeError, ValueError):
        return "Unknown"

    return codes.get(code, "Unknown")


def load_weather_for_location(location):

    geo = geocode_location(location)

    if geo is None:
        return None, None

    weather = get_weather(
        geo["latitude"],
        geo["longitude"]
    )

    return geo, weather


# ============================================================
# CROP DATABASE
# ============================================================

CROP_DATABASE = {

    "Rice": {
        "soil": "Clayey or loamy soil",
        "water": "High",
        "season": "Kharif / Rabi",
        "fertilizer": "Nitrogen, phosphorus and potassium",
        "pests": "Stem borer, brown planthopper, leaf folder",
        "diseases": "Blast, bacterial leaf blight, sheath blight"
    },

    "Wheat": {
        "soil": "Well-drained loamy soil",
        "water": "Medium",
        "season": "Rabi",
        "fertilizer": "Nitrogen, phosphorus and potassium",
        "pests": "Aphids, termites",
        "diseases": "Rust, powdery mildew"
    },

    "Maize": {
        "soil": "Fertile well-drained soil",
        "water": "Medium",
        "season": "Kharif / Rabi",
        "fertilizer": "Balanced NPK",
        "pests": "Fall armyworm, stem borer",
        "diseases": "Leaf blight, downy mildew"
    },

    "Cotton": {
        "soil": "Black soil or loamy soil",
        "water": "Medium",
        "season": "Kharif",
        "fertilizer": "Nitrogen, phosphorus and potassium",
        "pests": "Bollworm, whitefly",
        "diseases": "Wilt, bacterial blight"
    },

    "Tomato": {
        "soil": "Sandy loam to loamy soil",
        "water": "Medium",
        "season": "Year-round depending on climate",
        "fertilizer": "Organic manure with balanced NPK",
        "pests": "Fruit borer, whitefly, aphids",
        "diseases": "Early blight, late blight, bacterial wilt"
    },

    "Groundnut": {
        "soil": "Sandy loam soil",
        "water": "Medium",
        "season": "Kharif / Rabi",
        "fertilizer": "Phosphorus, calcium and balanced nutrients",
        "pests": "Leaf miner, red hairy caterpillar",
        "diseases": "Tikka disease, rust"
    },

    "Sugarcane": {
        "soil": "Deep fertile loamy soil",
        "water": "High",
        "season": "Year-round",
        "fertilizer": "Nitrogen, phosphorus and potassium",
        "pests": "Early shoot borer, top borer",
        "diseases": "Red rot, smut"
    },

    "Chilli": {
        "soil": "Well-drained loamy soil",
        "water": "Medium",
        "season": "Kharif / Rabi",
        "fertilizer": "Organic manure and balanced NPK",
        "pests": "Thrips, mites, aphids",
        "diseases": "Leaf curl, anthracnose"
    }
}


# ============================================================
# CROP ADVICE
# ============================================================

def get_crop_advice(
    crop,
    soil,
    weather=None,
    growth_stage="Not specified"
):

    info = CROP_DATABASE.get(crop)

    if info is None:
        return "Crop information is not available."

    advice = f"""
### 🌱 Crop: {crop}

**Suitable Soil:** {info["soil"]}

**Water Requirement:** {info["water"]}

**Growing Season:** {info["season"]}

**Fertilizer Guidance:** {info["fertilizer"]}

**Common Pests:** {info["pests"]}

**Common Diseases:** {info["diseases"]}

**Selected Soil Type:** {soil}

**Growth Stage:** {growth_stage}
"""

    if weather:

        current = weather.get("current", {})

        temperature = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        rainfall = current.get("rain", 0)

        advice += "\n### 🌦️ Weather-Based Advice\n"

        if temperature is not None:

            advice += f"\n- Current temperature: {temperature} °C."

            if temperature > 35:

                advice += (
                    "\n- High temperature detected. Monitor for heat stress "
                    "and maintain proper irrigation."
                )

            elif temperature < 15:

                advice += (
                    "\n- Low temperature detected. Monitor crops for cold stress."
                )

            else:

                advice += (
                    "\n- Temperature is generally suitable for normal crop growth."
                )

        if humidity is not None and humidity > 80:

            advice += "\n- High humidity may increase fungal disease risk."

        if rainfall is not None and rainfall > 5:

            advice += (
                "\n- Rainfall detected. Avoid unnecessary irrigation "
                "and check field drainage."
            )

    return advice


# ============================================================
# AI PROMPT
# ============================================================

def create_prompt(
    question,
    crop="",
    soil="",
    location="",
    weather=None,
    growth_stage=""
):

    weather_info = "Weather information unavailable."

    if weather:

        current = weather.get("current", {})

        weather_info = f"""
Temperature: {current.get("temperature_2m")} °C
Humidity: {current.get("relative_humidity_2m")} %
Rainfall: {current.get("rain")} mm
Wind Speed: {current.get("wind_speed_10m")} km/h
Weather Condition: {weather_description(current.get("weather_code", -1))}
"""

    prompt = f"""
You are AgriGPT, an agricultural AI assistant for Indian farmers.

Farmer Location: {location or "Not specified"}
Crop: {crop or "Not specified"}
Soil Type: {soil or "Not specified"}
Crop Growth Stage: {growth_stage or "Not specified"}

Current Weather:
{weather_info}

Farmer Question:
{question}

Give practical and simple agricultural advice.

Consider:
- Irrigation
- Fertilizer
- Crop growth stage
- Soil condition
- Pests
- Diseases
- Weather

Do not invent pesticide doses.
Do not recommend dangerous or unverified chemical treatments.
Recommend consulting a local agricultural expert when field inspection is required.
"""

    return prompt


# ============================================================
# LOCAL FALLBACK AI RESPONSE
# ============================================================

def fallback_response(
    question,
    crop="",
    soil="",
    weather=None
):

    q = question.lower()
    answers = []

    if any(word in q for word in [
        "water",
        "irrigation",
        "dry",
        "watering"
    ]):

        answers.append(
            "Check soil moisture before irrigation. Avoid overwatering "
            "and consider recent rainfall and crop growth stage."
        )

    if any(word in q for word in [
        "fertilizer",
        "urea",
        "npk",
        "manure"
    ]):

        answers.append(
            "Apply fertilizers based on soil testing and crop growth stage. "
            "Excessive fertilizer can damage crops and increase pest problems."
        )

    if any(word in q for word in [
        "pest",
        "insect",
        "worm",
        "bug"
    ]):

        answers.append(
            "Inspect leaves, stems and the underside of leaves regularly. "
            "Use integrated pest management methods such as field sanitation, "
            "traps and biological control where appropriate."
        )

    if any(word in q for word in [
        "disease",
        "yellow",
        "spot",
        "leaf",
        "fungus"
    ]):

        answers.append(
            "Check the crop for spreading symptoms, leaf spots, wilting "
            "and waterlogging. A clear image of the affected plant can help "
            "identify the problem more accurately."
        )

    if any(word in q for word in [
        "weather",
        "rain",
        "temperature",
        "climate"
    ]):

        answers.append(
            "Weather affects irrigation, disease development and fertilizer "
            "application. Check the real-time weather section before making "
            "field decisions."
        )

    if any(word in q for word in [
        "growth",
        "grow",
        "growing",
        "development",
        "increase yield",
        "yield"
    ]):

        answers.append(
            "For healthy crop growth, maintain proper soil moisture, "
            "provide balanced nutrients based on soil testing, remove weeds, "
            "monitor pests regularly and ensure sufficient sunlight. "
            "Crop age and growth stage are important for giving more specific advice."
        )

    if any(word in q for word in [
        "rice",
        "paddy"
    ]):

        answers.append(
            "For rice, maintain suitable soil moisture, avoid prolonged "
            "unnecessary standing water, monitor stem borers and planthoppers, "
            "and apply nutrients according to soil-test recommendations."
        )

    if any(word in q for word in [
        "tomato"
    ]):

        answers.append(
            "For tomato, maintain consistent soil moisture, provide good drainage, "
            "remove weeds and monitor leaves regularly for early blight, late blight "
            "and insect damage."
        )

    if not answers:

        answers.append(
            "Please provide the crop name, crop age, soil type, location "
            "and a detailed description of the problem for better advice."
        )

    if crop:

        answers.insert(
            0,
            f"Regarding your {crop} crop:"
        )

    return "\n\n".join(answers)


# ============================================================
# AI RESPONSE FUNCTION
# ============================================================

def get_ai_response(
    question,
    crop="",
    soil="",
    location="",
    weather=None,
    growth_stage=""
):

    if not question or not question.strip():
        return "Please enter an agricultural question."

    prompt = create_prompt(
        question=question,
        crop=crop,
        soil=soil,
        location=location,
        weather=weather,
        growth_stage=growth_stage
    )

    if agrigpt_response is not None:

        try:

            result = agrigpt_response(prompt)

            if result and str(result).strip():
                return str(result).strip()

        except Exception as error:

            print("AI service error:", error)

    return fallback_response(
        question=question,
        crop=crop,
        soil=soil,
        weather=weather
    )


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    login_html = """
    <div class="login-panel">
        <div class="login-logo">🌾</div>
        <div class="login-title">AgriGPT</div>
        <div class="login-subtitle">
            AI-Powered Crop Advisory Assistant
        </div>

           
    """

    st.markdown(
        dedent(login_html),
        unsafe_allow_html=True
    )

    left, middle, right = st.columns([1, 2, 1])

    with middle:

        login_tab, register_tab = st.tabs([
            "🔐 Login",
            "📝 Register"
        ])

        with login_tab:

            st.subheader("Login to AgriGPT")

            username = st.text_input(
                "Username",
                key="login_username"
            )

            password = st.text_input(
                "Password",
                type="password",
                key="login_password"
            )

            if st.button(
                "Login",
                type="primary",
                use_container_width=True
            ):

                if not username.strip() or not password:

                    st.warning(
                        "Please enter username and password."
                    )

                elif login_user(username, password):

                    st.session_state.logged_in = True
                    st.session_state.username = username.strip()
                    st.session_state.page = "Dashboard"

                    st.success("Login successful.")
                    st.rerun()

                else:

                    st.error(
                        "Invalid username or password."
                    )

        with register_tab:

            st.subheader("Create New Account")

            new_username = st.text_input(
                "New Username",
                key="new_username"
            )

            new_password = st.text_input(
                "New Password",
                type="password",
                key="new_password"
            )

            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                key="confirm_password"
            )

            if st.button(
                "Register",
                type="primary",
                use_container_width=True
            ):

                if not new_username.strip() or not new_password:

                    st.warning(
                        "Please fill all fields."
                    )

                elif new_password != confirm_password:

                    st.error(
                        "Passwords do not match."
                    )

                elif len(new_password) < 4:

                    st.warning(
                        "Password must contain at least 4 characters."
                    )

                else:

                    success, message = register_user(
                        new_username,
                        new_password
                    )

                    if success:

                        st.success(message)

                    else:

                        st.error(message)

    st.markdown(
        dedent("""
        <div class="login-caption">
            Smart farming decisions powered by artificial intelligence
        </div>
        """),
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

def sidebar():

    with st.sidebar:

        st.title("🌾 AgriGPT")

        st.write(
            f"Welcome, {st.session_state.username}"
        )

        st.divider()

        pages = [
            "Dashboard",
            "Crop Advisory",
            "AI Crop Recommendation",
            "Plant Disease Detection",
            "AI Agrigpt",
            "Weather",
            "Chat History"
        ]

        if st.session_state.page not in pages:
            st.session_state.page = "Dashboard"

        selected_page = st.radio(
            "Navigation",
            pages,
            index=pages.index(st.session_state.page)
        )

        if selected_page != st.session_state.page:

            st.session_state.page = selected_page
            st.rerun()

        st.divider()

        if st.button(
            "Logout",
            use_container_width=True
        ):

            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.chat_messages = []
            st.session_state.page = "Dashboard"
            st.session_state.weather_data = None
            st.session_state.location_name = ""
            st.session_state.latitude = None
            st.session_state.longitude = None

            st.rerun()


# ============================================================
# DASHBOARD PAGE
# ============================================================

def dashboard_page():

    st.title("🌾 AgriGPT Dashboard")

    st.write(
        "Your intelligent agricultural assistant."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            dedent("""
            <div class="metric-card">
                <h2>🌱</h2>
                <b>Crop Selection</b>
                <p>AI-based crop recommendation.</p>
            </div>
            """),
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            dedent("""
            <div class="metric-card">
                <h2>💧</h2>
                <b>Irrigation</b>
                <p>Water management guidance.</p>
            </div>
            """),
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            dedent("""
            <div class="metric-card">
                <h2>🐛</h2>
                <b>Pest Management</b>
                <p>Crop pest and disease advice.</p>
            </div>
            """),
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            dedent("""
            <div class="metric-card">
                <h2>🌦️</h2>
                <b>Live Weather</b>
                <p>Real-time weather integration.</p>
            </div>
            """),
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("Quick Agricultural Question")

    question = st.text_input(
        "Ask your question",
        placeholder="Example: My tomato leaves are turning yellow."
    )

    if st.button(
        "Get Advice",
        type="primary"
    ):

        if not question.strip():

            st.warning("Please enter a question.")

            return

        answer = get_ai_response(
            question=question,
            location=st.session_state.location_name,
            weather=st.session_state.weather_data
        )

        st.markdown("### AgriGPT Response")

        st.markdown(answer)

        save_chat(
            st.session_state.username,
            question,
            answer
        )


# ============================================================
# CROP ADVISORY PAGE
# ============================================================

def crop_advisory_page():

    st.title("🌱 Crop Advisory")

    st.write(
        "Get crop-specific advice using agricultural knowledge and weather data."
    )

    col1, col2 = st.columns(2)

    with col1:

        crop = st.selectbox(
            "Select Crop",
            list(CROP_DATABASE.keys()),
            key="advisory_crop"
        )

        soil = st.selectbox(
            "Soil Type",
            [
                "Clay soil",
                "Sandy soil",
                "Loamy soil",
                "Black soil",
                "Red soil",
                "Alluvial soil",
                "Not sure"
            ],
            key="advisory_soil"
        )

    with col2:

        location = st.text_input(
            "Location / District",
            placeholder="Example: Chennai",
            key="advisory_location"
        )

        stage = st.selectbox(
            "Crop Growth Stage",
            [
                "Seedling",
                "Vegetative",
                "Flowering",
                "Fruiting",
                "Harvesting"
            ],
            key="advisory_stage"
        )

    if st.button(
        "Generate Crop Advice",
        type="primary"
    ):

        weather = None

        if location.strip():

            with st.spinner("Fetching location and weather data..."):

                geo, weather = load_weather_for_location(location)

            if geo and weather:

                st.session_state.location_name = geo["name"]
                st.session_state.latitude = geo["latitude"]
                st.session_state.longitude = geo["longitude"]
                st.session_state.weather_data = weather

                st.success(
                    f"Weather loaded for {geo['name']}."
                )

            elif geo and not weather:

                st.warning(
                    "Location found, but weather data could not be fetched."
                )

            else:

                st.warning("Location not found.")

        advice = get_crop_advice(
            crop=crop,
            soil=soil,
            weather=weather,
            growth_stage=stage
        )

        st.markdown(advice)

        st.info(
            f"Growth stage: {stage}"
        )

        question = st.text_area(
            "Ask a specific question about this crop",
            placeholder="Example: How can I improve the growth of this crop?",
            key="advisory_question"
        )

        if question.strip():

            answer = get_ai_response(
                question=question,
                crop=crop,
                soil=soil,
                location=location,
                weather=weather,
                growth_stage=stage
            )

            st.subheader("AI Response")

            st.markdown(answer)

            save_chat(
                st.session_state.username,
                question,
                answer
            )


# ============================================================
# AI CROP RECOMMENDATION PAGE
# ============================================================

def crop_recommendation_page():

    st.title("🧠 AI Crop Recommendation")

    st.write(
        "This model predicts the most suitable crop using soil nutrients "
        "and environmental conditions."
    )

    col1, col2 = st.columns(2)

    with col1:

        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            max_value=200.0,
            value=50.0
        )

        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            max_value=200.0,
            value=50.0
        )

        potassium = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            max_value=200.0,
            value=50.0
        )

        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5
        )

    with col2:

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=25.0
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=1000.0,
            value=100.0
        )

    st.divider()

    if st.button(
        "Recommend Crop Using AI",
        type="primary",
        use_container_width=True
    ):

        if predict_crop is None:

            st.error("Crop model could not be loaded.")

            if CROP_MODEL_IMPORT_ERROR:
                st.code(CROP_MODEL_IMPORT_ERROR)

            st.info(
                "Check that services/crop_model.py and the trained crop model exist."
            )

            return

        try:

            prediction = predict_crop(
                nitrogen=nitrogen,
                phosphorus=phosphorus,
                potassium=potassium,
                temperature=temperature,
                humidity=humidity,
                ph=ph,
                rainfall=rainfall
            )

            if prediction:

                st.success(
                    f"Recommended Crop: {str(prediction).upper()}"
                )

                st.info(
                    "Prediction generated by the trained crop model."
                )

            else:

                st.warning(
                    "Model not found. Please train the crop model first."
                )

        except Exception as error:

            st.error(
                f"Prediction error: {error}"
            )


# ============================================================
# WEATHER PAGE
# ============================================================

def weather_page():

    st.title("🌦️ Real-Time Weather")

    st.write(
        "Weather is fetched from the Open-Meteo API."
    )

    location = st.text_input(
        "Enter Location",
        value=st.session_state.location_name,
        placeholder="Example: Gummidipundi",
        key="weather_location"
    )

    if st.button(
        "Get Real-Time Weather",
        type="primary"
    ):

        if not location.strip():

            st.warning("Please enter a location.")

        else:

            with st.spinner("Fetching real-time weather..."):

                geo, weather = load_weather_for_location(location)

            if geo is None:

                st.error("Location not found.")

            elif weather is None:

                st.error("Unable to fetch weather.")

            else:

                st.session_state.location_name = geo["name"]
                st.session_state.latitude = geo["latitude"]
                st.session_state.longitude = geo["longitude"]
                st.session_state.weather_data = weather

                st.success(
                    f"Weather loaded for {geo['name']}, {geo['country']}."
                )

    weather = st.session_state.weather_data

    if weather:

        current = weather.get("current", {})

        st.subheader(
            f"Current Weather - {st.session_state.location_name}"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Temperature",
                f"{current.get('temperature_2m', 'N/A')} °C"
            )

        with col2:

            st.metric(
                "Humidity",
                f"{current.get('relative_humidity_2m', 'N/A')} %"
            )

        with col3:

            st.metric(
                "Rainfall",
                f"{current.get('rain', 'N/A')} mm"
            )

        with col4:

            st.metric(
                "Wind Speed",
                f"{current.get('wind_speed_10m', 'N/A')} km/h"
            )

        st.info(
            weather_description(
                current.get("weather_code", -1)
            )
        )

        st.divider()

        st.subheader("7-Day Forecast")

        daily = weather.get("daily", {})

        dates = daily.get("time", [])
        max_temp = daily.get("temperature_2m_max", [])
        min_temp = daily.get("temperature_2m_min", [])
        rainfall = daily.get("rain_sum", [])
        probability = daily.get("precipitation_probability_max", [])

        forecast = []

        for i, date in enumerate(dates):

            forecast.append({
                "Date": date,
                "Max Temperature (°C)": (
                    max_temp[i]
                    if i < len(max_temp)
                    else None
                ),
                "Min Temperature (°C)": (
                    min_temp[i]
                    if i < len(min_temp)
                    else None
                ),
                "Rainfall (mm)": (
                    rainfall[i]
                    if i < len(rainfall)
                    else None
                ),
                "Rain Probability (%)": (
                    probability[i]
                    if i < len(probability)
                    else None
                )
            })

        df = pd.DataFrame(forecast)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        if not df.empty:

            st.subheader("Temperature Forecast")

            st.line_chart(
                df.set_index("Date")[
                    [
                        "Max Temperature (°C)",
                        "Min Temperature (°C)"
                    ]
                ]
            )

            st.subheader("Rainfall Forecast")

            st.bar_chart(
                df.set_index("Date")["Rainfall (mm)"]
            )

    else:

        st.info(
            "Enter a location and click Get Real-Time Weather."
        )


# ============================================================
# AI AGRIGPT CHAT PAGE
# ============================================================

def agrigpt_page():

    st.title("🤖 AI AgriGPT")

    st.write(
        "Ask agriculture-related questions using natural language."
    )

    col1, col2 = st.columns(2)

    with col1:

        crop = st.selectbox(
            "Crop Context",
            ["Not specified"] + list(CROP_DATABASE.keys()),
            key="chat_crop"
        )

    with col2:

        soil = st.selectbox(
            "Soil Context",
            [
                "Not specified",
                "Clay soil",
                "Sandy soil",
                "Loamy soil",
                "Black soil",
                "Red soil",
                "Alluvial soil"
            ],
            key="chat_soil"
        )

    st.divider()

    for message in st.session_state.chat_messages:

        safe_content = html.escape(
            str(message.get("content", ""))
        ).replace("\n", "<br>")

        if message["role"] == "user":

            st.markdown(
                dedent(f"""
                <div class="chat-user">
                    <b>You:</b><br>
                    {safe_content}
                </div>
                """),
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                dedent(f"""
                <div class="chat-bot">
                    <b>AgriGPT:</b><br>
                    {safe_content}
                </div>
                """),
                unsafe_allow_html=True
            )

    question = st.chat_input(
        "Ask your agricultural question..."
    )

    if question and question.strip():

        selected_crop = (
            ""
            if crop == "Not specified"
            else crop
        )

        selected_soil = (
            ""
            if soil == "Not specified"
            else soil
        )

        st.session_state.chat_messages.append({
            "role": "user",
            "content": question
        })

        answer = get_ai_response(
            question=question,
            crop=selected_crop,
            soil=selected_soil,
            location=st.session_state.location_name,
            weather=st.session_state.weather_data
        )

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": answer
        })

        save_chat(
            st.session_state.username,
            question,
            answer
        )

        st.rerun()

    if st.button("Clear Chat"):

        st.session_state.chat_messages = []

        st.rerun()


# ============================================================
# CHAT HISTORY PAGE
# ============================================================

def chat_history_page():

    st.title("📚 Chat History")

    history = get_chat_history(
        st.session_state.username
    )

    if not history:

        st.info("No chat history available.")

        return

    for question, answer, created_at in history:

        with st.expander(
            f"{created_at} - {question[:70]}"
        ):

            st.markdown("**Question:**")
            st.write(question)

            st.markdown("**Answer:**")
            st.markdown(answer)


# ============================================================
# PLANT DISEASE DETECTION PAGE
# ============================================================

def disease_detection_page():

    st.title("🔬 Plant Disease Detection")

    st.write(
        "Upload a clear image of a crop leaf. "
        "The trained AI model will analyze the image and predict "
        "the possible disease."
    )

    st.warning(
        "For better accuracy, upload a clear image showing the affected leaf."
    )

    if predict_disease is None:

        st.error("Disease detection model could not be loaded.")

        if DISEASE_MODEL_IMPORT_ERROR:
            st.code(DISEASE_MODEL_IMPORT_ERROR)

        st.info(
            "Check that services/disease_model.py exists and that the trained "
            "model files are inside the models folder."
        )

        return

    uploaded_file = st.file_uploader(
        "Upload Leaf Image",
        type=["jpg", "jpeg", "png"],
        key="disease_image_uploader"
    )

    if uploaded_file is None:

        st.info(
            "Please upload a crop leaf image to begin detection."
        )

        return

    try:

        image = Image.open(uploaded_file).convert("RGB")

    except Exception as error:

        st.error(
            f"Unable to open image: {error}"
        )

        return

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Leaf Image",
            use_container_width=True
        )

    with col2:

        st.subheader("AI Analysis")

        if st.button(
            "Detect Disease",
            type="primary",
            use_container_width=True,
            key="detect_disease_button"
        ):

            with st.spinner(
                "Analyzing leaf image..."
            ):

                try:

                    result = predict_disease(image)

                except Exception as error:

                    result = {
                        "disease": None,
                        "confidence": 0.0,
                        "error": str(error)
                    }

            if not isinstance(result, dict):

                result = {
                    "disease": str(result),
                    "confidence": 0.0,
                    "error": None
                }

            error_message = result.get("error")

            if error_message:

                st.error(
                    f"Disease prediction failed: {error_message}"
                )

                st.info(
                    "Make sure these files exist:\n\n"
                    "models/plant_disease_model.keras\n"
                    "models/disease_class_names.txt"
                )

                return

            disease = result.get(
                "disease",
                "Unknown Disease"
            )

            confidence = result.get(
                "confidence",
                0.0
            )

            try:

                confidence = float(confidence)

            except (TypeError, ValueError):

                confidence = 0.0

            if confidence > 1:

                confidence = confidence / 100.0

            confidence = max(
                0.0,
                min(confidence, 1.0)
            )

            display_disease = (
                str(disease)
                .replace("___", " - ")
                .replace("_", " ")
            )

            # =================================================
            # VISIBLE DISEASE RESULT
            # =================================================

            disease_html = dedent(f"""
            <div class="disease-result">
                <h3>Predicted Disease</h3>
                <h2>{html.escape(display_disease)}</h2>
            </div>
            """)

            st.markdown(
                disease_html,
                unsafe_allow_html=True
            )

            # =================================================
            # VISIBLE CONFIDENCE
            # =================================================

            confidence_html = dedent(f"""
            <div class="confidence-box">
                Confidence: {confidence * 100:.2f}%
            </div>
            """)

            st.markdown(
                confidence_html,
                unsafe_allow_html=True
            )

            # =================================================
            # CONFIDENCE MESSAGE
            # =================================================

            if confidence >= 0.80:

                st.success(
                    "The model has relatively high confidence "
                    "in this prediction."
                )

            elif confidence >= 0.50:

                st.warning(
                    "The model has moderate confidence. "
                    "Consider uploading a clearer image."
                )

            else:

                st.error(
                    "Low confidence prediction. "
                    "Please consult an agricultural expert "
                    "before taking treatment decisions."
                )

            st.divider()

            st.subheader("Recommended Next Steps")

            st.write(
                """
                1. Inspect nearby leaves for similar symptoms.
                2. Check whether the disease is spreading.
                3. Check soil moisture and drainage.
                4. Remove severely affected plant material if appropriate.
                5. Consult a local agricultural officer before applying pesticides.
                """
            )

            st.info(
                "AI predictions are advisory only and should be verified "
                "with a qualified agricultural expert."
            )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    if not st.session_state.logged_in:

        login_page()

        return

    sidebar()

    if st.session_state.page == "Dashboard":

        dashboard_page()

    elif st.session_state.page == "Crop Advisory":

        crop_advisory_page()

    elif st.session_state.page == "AI Crop Recommendation":

        crop_recommendation_page()

    elif st.session_state.page == "Plant Disease Detection":

        disease_detection_page()

    elif st.session_state.page == "AI Agrigpt":

        agrigpt_page()

    elif st.session_state.page == "Weather":

        weather_page()

    elif st.session_state.page == "Chat History":

        chat_history_page()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()
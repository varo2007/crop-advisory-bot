import os
import requests


def get_api_settings():

    api_url = os.getenv("AGRIGPT_API_URL", "")
    api_key = os.getenv("AGRIGPT_API_KEY", "")
    model = os.getenv("AGRIGPT_MODEL", "gpt-4o-mini")

    try:
        import streamlit as st

        if not api_url:
            api_url = st.secrets.get("AGRIGPT_API_URL", "")

        if not api_key:
            api_key = st.secrets.get("AGRIGPT_API_KEY", "")

        model = st.secrets.get("AGRIGPT_MODEL", model)

    except Exception:
        pass

    return api_url, api_key, model


def agrigpt_response(prompt):

    api_url, api_key, model = get_api_settings()

    if not api_url:
        raise RuntimeError(
            "AGRIGPT_API_URL is missing."
        )

    if not api_key:
        raise RuntimeError(
            "AGRIGPT_API_KEY is missing."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": """
You are AgriGPT, an AI agricultural assistant for Indian farmers.

Your responsibilities include:

1. Crop selection
2. Irrigation guidance
3. Fertilizer management
4. Pest management
5. Plant disease awareness
6. Soil health
7. Weather-based farming advice
8. Crop growth improvement
9. Yield improvement

Rules:
- Give simple and practical answers.
- Consider crop type, soil, location and weather.
- Do not invent pesticide doses.
- Do not provide dangerous chemical instructions.
- Recommend consulting agricultural experts when field inspection is needed.
- Answer the farmer's question directly.
"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3
    }

    try:

        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"API Error {response.status_code}: "
                f"{response.text[:500]}"
            )

        data = response.json()

        if "choices" not in data or not data["choices"]:
            raise RuntimeError("AI returned no response.")

        answer = data["choices"][0]["message"]["content"]

        if not answer:
            raise RuntimeError("AI returned an empty answer.")

        return answer.strip()

    except requests.exceptions.Timeout:
        raise RuntimeError("AI request timed out.")

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to the AI service. Check your internet connection."
        )

    except requests.exceptions.RequestException as error:
        raise RuntimeError(f"Request failed: {error}")

    except ValueError:
        raise RuntimeError("Invalid JSON response received from AI.")
# @MausamMitra — IMD Forecast Assistant

Conversational weather & disaster-alert assistant for India, built for **Smart India Hackathon 2026** (Ministry of Earth Sciences / IMD problem statement).

Ask about weather, forecasts, severe-weather alerts, or crop advisories for any location in India — every number comes from a live lookup, never a guess. Voice input/output, 10 Indian languages, and installable as an offline-ready PWA.

## Features

- 💬 **Conversational chat** grounded in live tool-calling (Google Gemini) — the model never states a weather number from memory
- 🗺️ **Forecast tab** — searchable map, 5-day outlook, live current conditions, wind/pressure/humidity gauges
- 🚨 **Alerts tab** — severe-weather ticker, always-visible Emergency Helpline card (112/1070/1077/1078), Offline SMS Weather Beacon
- 🌾 **Kisan Agro-Clock** — 24-hour farming advisory timeline (irrigation, heat, spraying, rain-risk windows) from real hourly forecast data
- 🎙️ **Voice in/out** — speech-to-text input, text-to-speech playback with speed control
- 🌐 **10 languages** — English, Hindi, Gujarati, Marathi, Tamil, Telugu, Kannada, Bengali, Punjabi, Malayalam — including a full on-screen native-script keyboard
- 📴 **Installable PWA** — home-screen icon, offline app shell, works like a native app on Android/iOS

## Tech stack

- **Backend:** FastAPI (Python), Google Gemini API for conversational grounding, Open-Meteo for live weather/geocoding (free, no key needed)
- **Frontend:** Vanilla HTML/CSS/JS (no framework), Leaflet.js for maps, Chart.js for forecasts

## Running locally

**Backend:**
```bash
cd "@MausamMitra/WeatherGPT_v0.3/WeatherGPT/backend"
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"   # PowerShell: $env:GEMINI_API_KEY="your-key-here"
uvicorn main:app --reload
```

**Frontend:**
```bash
cd "@MausamMitra/WeatherGPT_v0.3/WeatherGPT/frontend"
python -m http.server 5500
```

Open `http://localhost:5500` in your browser.

### Optional environment variables
| Variable | Purpose |
|---|---|
| `GEMINI_MODEL` | Override the default Gemini model (e.g. `gemini-flash-lite-latest` for a higher free-tier quota) |
| `DEMO_MODE` | Set to `true` to run without any Gemini API calls — canned answers for the 4 example questions, using real live weather data |

## Roadmap — what's live vs. what's next

- Weather data: live via Open-Meteo today → direct IMD/NWP feed in production
- Alerts: rule-based engine today → NDMA SACHET feed integration in production
- Multilingual voice: browser Web Speech API today → Bhashini for deeper Indic coverage

## Team

Built for SIH 2026 — Ministry of Earth Sciences / India Meteorological Department problem statement.

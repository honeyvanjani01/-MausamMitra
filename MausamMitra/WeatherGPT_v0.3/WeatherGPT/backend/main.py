"""
main.py
FastAPI entrypoint for @MausamMitra, powered by Gemini (google-genai) with
automatic function calling over real Open-Meteo weather data.

Run locally:
    pip install -r requirements.txt
    setx GEMINI_API_KEY "AQ.xxxxxxxxxxxx"      (Windows, persists across sessions)
    -- or, for the current PowerShell session only --
    $env:GEMINI_API_KEY="AQ.xxxxxxxxxxxx"
    uvicorn main:app --reload

Endpoints:
    POST /chat            -> conversational query, full reply at once
    POST /chat/stream      -> same, but Server-Sent Events streaming (for the
                              typing-indicator UI in the new frontend)
    GET  /geocode/search   -> place-name suggestions, for a location search box
    GET  /alerts           -> active rule-based alerts for a location (name or lat/lon)
    GET  /forecast         -> raw forecast for a location (name or lat/lon)
    POST /reset            -> clear a session's conversation history
    GET  /health           -> liveness check

v0.3 changes: fully async (no request thread ever blocks on Gemini or
Open-Meteo), a streaming chat endpoint for the new UI, and coordinate-based
lookups so the map picker doesn't have to round-trip through geocoding.
"""

import json
import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

import weather_tools as wt
import llm_agent
import demo_mode

# DEMO_MODE lets you build/polish the whole app — every tab, every screen —
# without spending any Gemini API quota: the chat still gives real answers
# for the 4 example questions (using live Open-Meteo data, same as the
# Forecast tab), it just skips the Gemini call entirely. Flip to "false"
# (or unset it) once GEMINI_API_KEY is ready to go live — no other code
# changes needed. Accepts "1"/"true"/"yes" (case-insensitive) as "on".
DEMO_MODE = os.environ.get("DEMO_MODE", "").strip().lower() in ("1", "true", "yes")

app = FastAPI(title="@MausamMitra API", version="0.3.0")

# In production, restrict this to your app's actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    language: Optional[str] = "en"  # e.g. "en", "hi", "gu" — see llm_agent.LANGUAGE_NAMES


class ChatResponse(BaseModel):
    reply: str
    trace: list[dict]


class ResetRequest(BaseModel):
    session_id: str


@app.on_event("shutdown")
async def _shutdown():
    await wt.aclose()


@app.get("/")
def root():
    """Friendly landing response — this is an API, not a webpage, so there's
    nothing to render at '/'. {"detail":"Not Found"} here is expected and
    harmless; the frontend never calls this path, only /chat, /forecast,
    /alerts, /geocode/search, and /health.
    """
    return {
        "service": "@MausamMitra API",
        "status": "ok",
        "demo_mode": DEMO_MODE,
        "hint": "This is the backend API, not a webpage. Open the frontend "
                "(served separately, e.g. http://localhost:5500) to use the app.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok", "demo_mode": DEMO_MODE}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if DEMO_MODE:
        result = await demo_mode.demo_reply(req.message, req.language or "en")
        return {"reply": result["reply"], "trace": result["trace"]}
    try:
        result = await llm_agent.chat(req.session_id, req.message, req.language or "en")
    except Exception as exc:
        if llm_agent._is_rate_limit_error(exc):
            # 429 (not 502): this is Gemini's free-tier daily quota, not a
            # backend fault — give the real reason instead of a generic
            # "couldn't reach the assistant" that looks like our code is broken.
            raise HTTPException(429, llm_agent.RATE_LIMIT_MESSAGE)
        # Surface a clean, user-facing message instead of a raw Gemini/SDK
        # stack trace leaking to the frontend as "Server error: 400 ...".
        raise HTTPException(502, f"Couldn't reach the assistant right now: {exc}")
    return {"reply": result["reply"], "trace": result["trace"]}


@app.post("/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    async def event_source():
        stream = (
            demo_mode.demo_reply_stream(req.message, req.language or "en")
            if DEMO_MODE
            else llm_agent.chat_stream(req.session_id, req.message, req.language or "en")
        )
        async for event in stream:
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/reset")
def reset_endpoint(req: ResetRequest):
    llm_agent.reset_session(req.session_id)
    return {"status": "reset", "session_id": req.session_id}


@app.get("/geocode/search")
async def geocode_search_endpoint(q: str, count: int = 5):
    """Place-name suggestions for a search-as-you-type location box."""
    return await wt.geocode_suggestions(q, count)


async def _resolve_location(place: Optional[str], lat: Optional[float], lon: Optional[float]) -> dict:
    if lat is not None and lon is not None:
        return {"name": f"{lat:.2f}, {lon:.2f}", "admin1": "", "lat": lat, "lon": lon}
    if place:
        loc = await wt.geocode_location(place)
        if "error" in loc:
            raise HTTPException(404, "location not found")
        return loc
    raise HTTPException(400, "provide either 'place' or both 'lat' and 'lon'")


@app.get("/alerts")
async def alerts_endpoint(
    place: Optional[str] = None,
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
    days: int = 5,
):
    loc = await _resolve_location(place, lat, lon)
    forecast = await wt.get_forecast(loc["lat"], loc["lon"], days)
    return {"location": loc, **wt.evaluate_alerts(forecast)}


@app.get("/forecast")
async def forecast_endpoint(
    place: Optional[str] = None,
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
    days: int = 5,
):
    loc = await _resolve_location(place, lat, lon)
    forecast = await wt.get_forecast(loc["lat"], loc["lon"], days)
    # Alerts are derived from the same forecast payload we already fetched
    # (evaluate_alerts is pure/local, no extra network call) — included here
    # so the frontend can get forecast + alerts in ONE request instead of
    # two separate round trips that raced each other for the same data.
    return {"location": loc, "forecast": forecast, "alerts": wt.evaluate_alerts(forecast)["alerts"]}

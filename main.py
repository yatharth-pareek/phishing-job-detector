"""FastAPI backend: python -m uvicorn main:app --reload  ->  http://localhost:8000"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy.sparse import hstack, csr_matrix

from features import clean_text, job_flag_vector, job_indicators, url_features, url_indicators

BASE = Path(__file__).parent
app = FastAPI(title="Phishing Website & Fake Job Detection System")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


def _load(name):
    p = BASE / "models" / name
    return joblib.load(p) if p.exists() else None


url_bundle = _load("url_model.pkl")
job_bundle = _load("job_model.pkl")


class URLInput(BaseModel):
    url: str


class JobInput(BaseModel):
    text: str


def risk_level(score: int) -> str:
    return "High" if score >= 70 else "Medium" if score >= 40 else "Low"


@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")


@app.post("/predict-url")
def predict_url(data: URLInput):
    if url_bundle is None:
        raise HTTPException(503, "URL model not trained. Run: python train_url.py")
    if len(data.url.strip()) < 4:
        raise HTTPException(400, "Please enter a valid URL")
    feats = url_features(data.url)
    row = pd.DataFrame([feats])[url_bundle["features"]]
    prob = float(url_bundle["model"].predict_proba(row)[0][1])
    score = round(prob * 100)
    return {"result": "phishing" if prob >= 0.5 else "safe", "risk_score": score,
            "risk_level": risk_level(score), "indicators": url_indicators(data.url, feats)}


@app.post("/predict-job")
def predict_job(data: JobInput):
    if job_bundle is None:
        raise HTTPException(503, "Job model not trained. Run: python train_job.py")
    if len(data.text.strip()) < 20:
        raise HTTPException(400, "Please paste the full job description")
    X = hstack([job_bundle["tfidf"].transform([clean_text(data.text)]),
                csr_matrix(job_flag_vector(data.text).reshape(1, -1))]).tocsr()
    prob = float(job_bundle["model"].predict_proba(X)[0][1])
    score = round(prob * 100)
    return {"result": "fake" if prob >= 0.5 else "genuine", "risk_score": score,
            "risk_level": risk_level(score), "indicators": job_indicators(data.text)}


@app.get("/metrics")
def metrics():
    out = {}
    for k in ("url", "job"):
        p = BASE / "models" / f"metrics_{k}.json"
        out[k] = json.loads(p.read_text()) if p.exists() else None
    return out

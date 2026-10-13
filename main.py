"""FastAPI service and static frontend for insurance purchase predictions."""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
MODEL_PATH = ROOT / "model.pkl"

app = FastAPI(title="Insurance Purchase Predictor", version="1.0.0")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=18, le=100)
    annual_income: float = Field(ge=1_000, le=2_000_000)
    annual_premium: float = Field(ge=100, le=50_000)
    vehicle_age_years: float = Field(ge=0, le=50)
    previously_insured: bool
    vehicle_damage: bool
    region: Literal["Central", "Coastal", "Mountain", "Metro"]


class PredictionOutput(BaseModel):
    prediction: Literal[0, 1]
    probability: float
    result: str


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/api/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput) -> PredictionOutput:
    if model is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Model is not trained. Run python train_model.py first.")

    probability = float(model.predict_proba(pd.DataFrame([payload.model_dump()]))[0][1])
    prediction = int(probability >= 0.5)
    return PredictionOutput(
        prediction=prediction,
        probability=probability,
        result="Likely to purchase" if prediction else "Unlikely to purchase",
    )
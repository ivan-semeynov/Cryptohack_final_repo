from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_utils import load_model_package, predict_records


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., description="List of observations with model features.")


@lru_cache(maxsize=1)
def get_package():
    model_dir = os.getenv("MODEL_DIR", "./artifacts")
    return load_model_package(model_dir)


app = FastAPI(title="Catalyst Risk API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    package = get_package()
    return {
        "threshold": package.threshold,
        "config": package.config,
        "feature_columns": package.feature_columns,
        "metrics": package.metrics,
        "top_features": package.feature_importance[:10],
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    if not request.records:
        raise HTTPException(status_code=400, detail="records must not be empty")
    package = get_package()
    predictions = predict_records(request.records, package)
    return {
        "count": len(predictions),
        "threshold": package.threshold,
        "predictions": predictions,
    }

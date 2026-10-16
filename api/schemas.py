from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    subscriber_id: str
    features: dict[str, Any] = Field(..., description="Feature name -> value")


class PredictionResponse(BaseModel):
    subscriber_id: str
    risk_probability: float
    risk_prediction: int
    threshold: float
    model_name: str
    model_version: str
    request_id: str
from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    subscriber_id: str
    features: dict[str, Any] = Field(..., description="Feature name -> value")


class Reason(BaseModel):
    feature: str
    value: Any = None
    contribution: float = Field(..., description="Push towards higher risk, in log-odds")


class PredictionResponse(BaseModel):
    subscriber_id: str
    risk_probability: float
    risk_prediction: int
    threshold: float
    model_name: str
    model_version: str
    request_id: str
    top_reasons: list[Reason] = Field(default_factory=list)

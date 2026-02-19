"""
schemas.py — Pydantic Request / Response Models
================================================
Defines the data contracts for the FastAPI serving layer.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PassengerInput(BaseModel):
    """Single passenger feature vector for prediction."""

    Pclass: int = Field(..., ge=1, le=3, description="Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)")
    Sex: str = Field(..., description="Sex (male / female)")
    Age: Optional[float] = Field(None, ge=0, le=120, description="Age in years")
    SibSp: int = Field(0, ge=0, description="Number of siblings / spouses aboard")
    Parch: int = Field(0, ge=0, description="Number of parents / children aboard")
    Fare: float = Field(..., ge=0, description="Passenger fare")
    Embarked: Optional[str] = Field("S", description="Port of embarkation (C / Q / S)")
    Name: Optional[str] = Field(
        "Unknown, Mr.",
        description="Passenger name (used for title extraction)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Pclass": 3,
                    "Sex": "male",
                    "Age": 22.0,
                    "SibSp": 1,
                    "Parch": 0,
                    "Fare": 7.25,
                    "Embarked": "S",
                    "Name": "Braund, Mr. Owen Harris",
                }
            ]
        }
    }


class PredictionOutput(BaseModel):
    """Prediction result for a single passenger."""

    survived: int = Field(..., description="Predicted survival (0 = No, 1 = Yes)")
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of survival"
    )


class BatchInput(BaseModel):
    """Batch of passengers for bulk prediction."""

    passengers: List[PassengerInput]


class BatchOutput(BaseModel):
    """Batch prediction results."""

    predictions: List[PredictionOutput]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    model_loaded: bool = False
    version: str = "1.0.0"

"""Pydantic models used both as Gemini response schemas and for validation.

Fields intentionally have no defaults: every field is required in the JSON
schema sent to the API, which makes the model's output complete and
predictable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

Sentiment = Literal["positive", "negative", "neutral", "mixed"]
Polarity = Literal["positive", "negative", "neutral"]
Level = Literal["high", "medium", "low"]


class AspectSentiment(BaseModel):
    aspect: str
    sentiment: Polarity
    evidence: str

    @field_validator("aspect")
    @classmethod
    def _normalise_aspect(cls, value: str) -> str:
        return " ".join(value.lower().split())


class ReviewAnalysis(BaseModel):
    id: int
    sentiment: Sentiment
    score: float
    aspects: list[AspectSentiment]
    emotions: list[str]
    key_phrases: list[str]
    summary: str
    actionable: bool
    language: str

    @field_validator("score")
    @classmethod
    def _clamp_score(cls, value: float) -> float:
        return max(-1.0, min(1.0, value))

    @field_validator("emotions", "key_phrases")
    @classmethod
    def _lowercase(cls, values: list[str]) -> list[str]:
        return [v.strip().lower() for v in values if v.strip()]


class BatchAnalysis(BaseModel):
    results: list[ReviewAnalysis]


class Theme(BaseModel):
    title: str
    description: str
    frequency: Level
    example_review_ids: list[int]


class Recommendation(BaseModel):
    action: str
    rationale: str
    priority: Level
    related_aspect: str


class InsightReport(BaseModel):
    executive_summary: str
    strengths: list[Theme]
    pain_points: list[Theme]
    recommendations: list[Recommendation]
    emerging_issues: list[str]

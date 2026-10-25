"""Public API contract: scores are model signals, never calibrated probabilities."""
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Segment(BaseModel):
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(gt=0, allow_inf_nan=False)
    score: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode='after')
    def ordered(self):
        if self.end <= self.start:
            raise ValueError('Segment end must follow start')
        return self

class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra='allow')
    classification: Literal['human', 'ai', 'uncertain']
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    segments: list[Segment]
    model: dict[str, Any]
    limitations: list[str]
    duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    sample_rate: Literal[16000]
    processing: dict[str, Any]
    latency_ms: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode='after')
    def segment_bounds(self):
        previous = 0.0
        for segment in self.segments:
            if segment.start < previous or segment.end > self.duration_seconds + 1e-6:
                raise ValueError('Segments must be ordered within the recording')
            previous = segment.end
        return self

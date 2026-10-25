"""Read explicitly configured, offline evaluation summaries; never generate metrics."""
import json
import os
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, ValidationError

class EvaluationReport(BaseModel):
    dataset: str = Field(min_length=1, max_length=300)
    sample_count: int = Field(gt=0, strict=True)
    model: str = Field(min_length=1, max_length=300)
    conditions: str = Field(min_length=1, max_length=2000)
    metrics: dict[str, float] = Field(min_length=1, max_length=40)

    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


def research_reports():
    configured = os.getenv('VOXGUARD_RESEARCH_REPORTS_PATH')
    if not configured:
        return {'status': 'not_evaluated', 'reports': []}
    try:
        path = Path(configured)
        if path.stat().st_size > 262144:
            raise ValueError('Report file too large')
        payload = json.loads(path.read_text(encoding='utf8'))
        reports = payload['reports']
        if not isinstance(reports, list) or len(reports) > 20:
            raise ValueError('Invalid report collection')
        validated = [EvaluationReport.model_validate(report).model_dump() for report in reports]
        return {'status': 'available' if validated else 'not_evaluated', 'reports': validated}
    except (OSError, ValueError, TypeError, KeyError, ValidationError):
        return {'status': 'unavailable', 'reports': [], 'detail': 'Configured evaluation summaries are invalid or unavailable.'}

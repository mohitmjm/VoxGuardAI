import pytest
from pydantic import ValidationError
from backend.schemas import Segment, AnalysisResult

@pytest.mark.parametrize('values', [dict(start=1,end=1,score=.5),dict(start=0,end=1,score=float('nan')),dict(start=0,end=1,score=2)])
def test_invalid_segment_rejected(values):
    with pytest.raises(ValidationError): Segment(**values)

def test_segment_cannot_extend_beyond_recording():
    with pytest.raises(ValidationError):
        AnalysisResult(classification='uncertain',score=.2,segments=[dict(start=0,end=2,score=.2)],model={},limitations=[],duration_seconds=1,sample_rate=16000,processing={},latency_ms=10)

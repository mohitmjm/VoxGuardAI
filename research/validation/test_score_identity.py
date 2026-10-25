"""Identity join contract with an explicit inference double, not model evidence."""
import hashlib
import numpy as np
import pytest
from research import evaluate


@pytest.mark.parametrize('explicit',[None,'official-recording-007'])
def test_scored_rows_preserve_or_derive_byte_identity(tmp_path,monkeypatch,explicit):
    from ml import service
    class InferenceDouble:
        def analyze(self,audio,sample_rate): return {'score':.25}
    monkeypatch.setattr(service,'ModelService',InferenceDouble)
    monkeypatch.setattr(evaluate,'decode',lambda path:np.zeros(16000,dtype=np.float32))
    path=tmp_path/'recording.wav';path.write_bytes(b'identity fixture only')
    row={'audio_path':str(path),'label':'human','speaker_id':'official-speaker','source_id':'official-source','split':'test'}
    if explicit is not None: row['recording_id']=explicit
    result=list(evaluate.score([row]))[0]
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['audio_sha256']==digest
    assert result['recording_id']==(explicit or 'sha256:'+digest)
    assert result['speaker_id']=='official-speaker' and result['source_id']=='official-source'

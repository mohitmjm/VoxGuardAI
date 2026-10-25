"""Byte identity/alignment tests; no training data or detection metrics."""
import hashlib
import pytest
from experiments.extract_features import recording_identities

def test_original_and_derived_ids_preserved(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.write_bytes(b'a');b.write_bytes(b'b')
    ids, digests=recording_identities([{'audio_path':a,'recording_id':'official-a'},{'audio_path':b}])
    assert ids == ['official-a','sha256:'+hashlib.sha256(b'b').hexdigest()]
    assert digests[0] == hashlib.sha256(b'a').hexdigest()

@pytest.mark.parametrize('same_bytes',[True,False])
def test_duplicates_rejected(tmp_path,same_bytes):
    a=tmp_path/'a';b=tmp_path/'b';a.write_bytes(b'a');b.write_bytes(b'a' if same_bytes else b'b')
    rows=[{'audio_path':a,'recording_id':'x'},{'audio_path':b,'recording_id':'y' if same_bytes else 'x'}]
    with pytest.raises(ValueError,match='inflate'): recording_identities(rows)


def test_mutating_source_cannot_produce_matched_features(tmp_path, monkeypatch):
    import numpy as np
    import experiments.extract_features as features
    path=tmp_path/'recording.wav';path.write_bytes(b'original bytes')
    expected=hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(features, 'decode', lambda _: np.zeros(16000, dtype=np.float32))
    class FixtureExtractor:
        def extract(self, samples):
            path.write_bytes(b'changed recording')
            return np.zeros(1536), np.zeros(40)
    with pytest.raises(ValueError, match='changed during'):
        features.extract_verified({'audio_path':path}, expected, FixtureExtractor())

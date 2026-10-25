"""Network-free acquisition contract tests; fixture bytes are not speech data."""
import hashlib
import io
import json
from pathlib import Path
import pytest
from scripts.datasets.download import download
from scripts.datasets.import_local import import_rows, asvspoof5_rows

class Reply(io.BytesIO):
    def __init__(self,data,status=200,headers=None):
        super().__init__(data);self.status=status;self.headers=headers or {}

@pytest.fixture
def args(tmp_path):
    rights=tmp_path/'rights.txt';rights.write_text('Explicit recording evaluation license fixture')
    return dict(output=tmp_path/'archive',sha256=hashlib.sha256(b'abcdef').hexdigest(),max_bytes=6,license_file=rights,license_url='https://example.org/LICENSE')

def test_download_complete_integrity(args):
    download('https://example.org/data',**args,opener=lambda *a,**k:Reply(b'abcdef'))
    assert args['output'].read_bytes()==b'abcdef'
    download('https://example.org/data',**args,opener=lambda *a,**k:pytest.fail('Already verified'))

def test_resume_validates_offset_and_etag(args):
    class Interrupted(Reply):
        def read(self,n):
            if self.tell():raise OSError('Connection interrupted')
            return super().read(3)
    with pytest.raises(OSError):download('https://example.org/data',**args,opener=lambda *a,**k:Interrupted(b'abcdef',headers={'ETag':'"v1"'}))
    def resume(request,**kwargs):
        assert request.get_header('Range')=='bytes=3-'
        assert request.get_header('If-range')=='"v1"'
        return Reply(b'def',206,{'Content-Range':'bytes 3-5/6','ETag':'"v1"'})
    download('https://example.org/data',**args,opener=resume)
    assert args['output'].read_bytes()==b'abcdef'

def test_wrong_hash_size_and_private_url_rejected(args):
    with pytest.raises(ValueError,match='SHA-256'):download('https://example.org/data',**args,opener=lambda *a,**k:Reply(b'bad'))
    args['output'].with_name('archive.part').unlink()
    with pytest.raises(ValueError,match='byte limit'):download('https://example.org/data',**args,opener=lambda *a,**k:Reply(b'1234567'))
    with pytest.raises(ValueError,match='credential-free'):download('https://example.org/data?token=secret',**args)

def row(name='clip.wav',**kw):
    return dict(audio_path=name,label='human',speaker_id='sp1',source_id='utterance1',generator='human',split='test',language='en',**kw)

def test_local_import_rejects_missing_provenance_and_escaping_path(tmp_path):
    (tmp_path/'clip.wav').write_bytes(b'fixture')
    valid=row();assert import_rows([valid],tmp_path)[0][0]['speaker_id']=='sp1'
    invalid={**valid,'source_id':'unknown'}
    with pytest.raises(ValueError,match='Unknown'):import_rows([invalid],tmp_path)
    with pytest.raises(ValueError,match='escapes'):import_rows([{**valid,'audio_path':'../outside.wav'}],tmp_path)

def test_partial_import_preserves_protocol_split_and_ids(tmp_path):
    (tmp_path/'clip.wav').write_bytes(b'fixture')
    rows=[row(),row('missing.wav')]
    selected,missing=import_rows(rows,tmp_path,allow_missing=True,limit=1)
    assert missing==1 and selected[0]['source_id']=='utterance1' and selected[0]['split']=='test'

def test_official_baseline_adapter_requires_source_mapping(tmp_path):
    protocol=tmp_path/'protocol.txt';protocol.write_text('speaker1 record1 - - - bonafide\n')
    with pytest.raises(ValueError,match='source grouping'):asvspoof5_rows(protocol,{},'eval','en')
    rows=asvspoof5_rows(protocol,{'record1':{'source_id':'reviewed-source'}},'eval','en')
    assert rows[0]['speaker_id']=='speaker1' and rows[0]['recording_id']=='record1'
    assert rows[0]['original_partition']=='eval' and rows[0]['generator']=='unknown'

def test_resume_rejects_changed_etag_or_wrong_range(args):
    class Interrupted(Reply):
        def read(self,n):
            if self.tell():raise OSError('interrupted')
            return super().read(3)
    with pytest.raises(OSError):download('https://example.org/data',**args,opener=lambda *a,**k:Interrupted(b'abcdef',headers={'ETag':'"v1"'}))
    with pytest.raises(ValueError,match='ETag'):
        download('https://example.org/data',**args,opener=lambda *a,**k:Reply(b'def',206,{'Content-Range':'bytes 3-5/6','ETag':'"v2"'}))
    with pytest.raises(ValueError,match='range'):
        download('https://example.org/data',**args,opener=lambda *a,**k:Reply(b'def',206,{'Content-Range':'bytes 0-2/6','ETag':'"v1"'}))
    assert args['output'].with_name('archive.part').read_bytes()==b'abc'

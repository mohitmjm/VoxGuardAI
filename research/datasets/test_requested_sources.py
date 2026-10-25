"""Network-free fixtures test acquisition mechanics, never real research findings."""
import hashlib
import io
import json
from pathlib import Path
import pytest
from scripts.datasets.source_catalog import inspect
from scripts.datasets.hf_subset import selection, fetch
from scripts.datasets.transfer import seal, verify, checksum


def catalog():
    return {'status':'metadata_available','source':'mlaad-tiny','metadata':{'id':'mueller91/MLAAD-tiny','sha':'a'*40,'siblings':[{'rfilename':'audio.wav','lfs':{'size':4,'sha256':hashlib.sha256(b'clip').hexdigest()}}]}}


def test_catalog_failure_preserves_no_rights_claim():
    def denied(*a,**k):raise OSError('Tunnel403')
    result=inspect('mlaad-tiny',opener=denied)
    assert result['status']=='unavailable' and result['license_status']=='not_retrieved'
    assert result['recordings_downloaded']==0


def test_catalog_success_is_metadata_not_permission():
    result=inspect('in-the-wild',opener=lambda *a,**k:io.BytesIO(b'{"sha":"abc","cardData":{"license":"unknown"}}'))
    assert result['status']=='metadata_available'
    assert 'require review' in result['provenance_status']


def test_selection_requires_revision_explicit_files_sizes_budget():
    assert selection(catalog(),['audio.wav'],4)[2]==4
    for names,limit in [(['audio.wav'],3),(['missing'],4),(['../audio.wav'],4),(['audio.wav','audio.wav'],8)]:
        with pytest.raises(ValueError):selection(catalog(),names,limit)
    data=catalog();data['metadata']['sha']='main'
    with pytest.raises(ValueError,match='revision'):selection(data,['audio.wav'],4)


def test_fetch_pins_revision_and_checks_lfs(tmp_path):
    rights=tmp_path/'rights.txt';rights.write_text('Explicit recording rights unit fixture')
    def download(**kw):
        assert kw['repo_id']=='mueller91/MLAAD-tiny' and kw['revision']=='a'*40
        path=Path(kw['local_dir'])/kw['filename'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'clip');return path
    result=fetch(catalog(),['audio.wav'],tmp_path/'out',rights,4,download=download)
    assert result['files'][0]['checksum_provenance']=='HF LFS SHA-256'
    with pytest.raises(ValueError,match='complete'):fetch(catalog(),['audio.wav'],tmp_path/'out',rights,4,download=download)


def test_transfer_verifies_original_digest_and_detects_damage(tmp_path):
    root=tmp_path/'bundle';root.mkdir();(root/'clip.wav').write_bytes(b'fixture')
    manifest=tmp_path/'inventory.json';manifest.write_text(json.dumps(seal(root,['clip.wav'])))
    assert verify(root,manifest,checksum(manifest))==1
    (root/'clip.wav').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):verify(root,manifest,checksum(manifest))
    with pytest.raises(ValueError,match='Inventory digest'):verify(root,manifest,'0'*64)


def test_transfer_rejects_escape_and_duplicate(tmp_path):
    (tmp_path/'clip').write_bytes(b'fixture')
    with pytest.raises(ValueError):seal(tmp_path,['../outside'])
    with pytest.raises(ValueError):seal(tmp_path,['clip','clip'])


def test_download_rejects_symlink_escape(tmp_path):
    rights=tmp_path/'rights';rights.write_text('Unit rights')
    root=tmp_path/'out';root.mkdir();outside=tmp_path/'outside';outside.mkdir()
    try:(root/'files').symlink_to(outside,target_is_directory=True)
    except OSError:pytest.skip('Symlinks unavailable on this platform')
    with pytest.raises(ValueError,match='escapes'):fetch(catalog(),['audio.wav'],root,rights,4,download=lambda **kw:pytest.fail('Do not download'))


def test_selection_rejects_wrong_repo_bad_digest_unknown_size_and_boolean_budget():
    data=catalog();data['metadata']['id']='other/repository'
    with pytest.raises(ValueError,match='identity'):selection(data,['audio.wav'],4)
    data=catalog();data['metadata']['siblings'][0]['lfs']['sha256']='not-a-digest'
    with pytest.raises(ValueError,match='SHA-256'):selection(data,['audio.wav'],4)
    data=catalog();data['metadata']['siblings'][0]['lfs'].pop('size')
    with pytest.raises(ValueError,match='size'):selection(data,['audio.wav'],4)
    with pytest.raises(ValueError,match='budget'):selection(catalog(),['audio.wav'],True)


def test_download_cannot_substitute_another_file_in_the_bundle(tmp_path):
    rights=tmp_path/'rights';rights.write_text('Explicit recording rights unit fixture')
    def wrong_file(**kw):
        path=Path(kw['local_dir'])/'other.wav';path.parent.mkdir(parents=True);path.write_bytes(b'clip');return path
    with pytest.raises(ValueError,match='different file'):
        fetch(catalog(),['audio.wav'],tmp_path/'out',rights,4,download=wrong_file)


def test_transfer_rejects_boolean_file_size(tmp_path):
    (tmp_path/'clip').write_bytes(b'x')
    data=seal(tmp_path,['clip']);data['files'][0]['bytes']=True
    inventory=tmp_path/'inventory.json';inventory.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='checksum'):verify(tmp_path,inventory,checksum(inventory))


def test_download_catalog_must_name_its_repository():
    data=catalog();data['metadata'].pop('id')
    with pytest.raises(ValueError,match='identity'):
        selection(data,['audio.wav'],4)

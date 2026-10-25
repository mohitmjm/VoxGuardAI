"""Numerical fixtures verify protocol mechanics; not detector measurements."""
import csv
from pathlib import Path
import pytest
from ml.constants import SHA256
from ml.policy import SCORE_PROTOCOL
from research.validation import run as pipeline


def write_manifest(root, partitions=('validation','test')):
    rows=[]
    for split in partitions:
        for index,label in enumerate(('human','human','ai','ai')):
            name=f'{split}-{index}.wav';(root/name).write_bytes(name.encode())
            rows.append(dict(audio_path=name,label=label,speaker_id=f'{split}-speaker-{index}',source_id=f'{split}-source-{index}',generator='human' if label=='human' else 'fixture-generator',split=split,language='en'))
    path=root/'manifest.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return path,rows


def scores(rows):
    for row in rows:
        yield {**row,'score':.1 if row['label']=='human' else .9,'latency_ms':1.,'duration_seconds':1.,'checkpoint_sha256':SHA256,'score_protocol':SCORE_PROTOCOL}


def test_pipeline_uses_only_validation_for_fit_and_test_for_report(tmp_path,monkeypatch):
    manifest,_=write_manifest(tmp_path)
    monkeypatch.setattr(pipeline,'score',scores)
    result=pipeline.run(manifest,tmp_path/'run',dataset='unit fixture',conditions='not empirical data',fit_policy=True,min_count=2,min_speakers=2)
    assert result['policy_status']=='fitted_research_only'
    assert result['policy']['counts']==dict(human=2,ai=2,human_speakers=2,ai_speakers=2)
    assert result['test_decisions']['sample_count']==4
    assert result['test_decisions']['coverage']==1.
    with pytest.raises(FileExistsError):
        pipeline.run(manifest,tmp_path/'run',dataset='fixture',conditions='fixture')


def test_insufficient_validation_retains_abstention(tmp_path,monkeypatch):
    manifest,_=write_manifest(tmp_path,('test',))
    monkeypatch.setattr(pipeline,'score',scores)
    result=pipeline.run(manifest,tmp_path/'run',dataset='fixture',conditions='fixture',fit_policy=True)
    assert result['policy_status']=='not_established'
    assert result['policy'] is None and result['test_decisions'] is None
    assert not (tmp_path/'run'/'policy.json').exists()


def test_duplicate_recordings_cannot_inflate_sample_count(tmp_path):
    path=tmp_path/'audio.wav';path.write_bytes(b'one recording')
    rows=[dict(audio_path=str(path),speaker_id='speaker',source_id='source',split='test')]*2
    with pytest.raises(ValueError,match='Repeated recording'):
        pipeline.audit_recordings(rows)


def test_generator_heldout_excludes_validation_generators(tmp_path):
    manifest,_=write_manifest(tmp_path)
    with pytest.raises(ValueError,match='generators overlap'):
        pipeline.run(manifest,tmp_path/'run',dataset='fixture',conditions='fixture',held_out_generators=True)
    assert not (tmp_path/'run').exists()


def test_failure_preserves_provenance_and_never_success(tmp_path,monkeypatch):
    manifest,_=write_manifest(tmp_path)
    def broken(rows): raise RuntimeError('unit simulated model failure')
    monkeypatch.setattr(pipeline,'score',broken)
    with pytest.raises(RuntimeError):pipeline.run(manifest,tmp_path/'run',dataset='fixture',conditions='fixture')
    assert (tmp_path/'run'/'failure.json').is_file()
    assert (tmp_path/'run'/'provenance.json').is_file()
    assert not (tmp_path/'run'/'summary.json').exists()


def test_fractional_labels_not_silently_coerced():
    from research.evaluate import curves, metrics
    with pytest.raises(ValueError):curves([.2,1],[.1,.9])
    with pytest.raises(ValueError):metrics([.2,1],[.1,.9],.5)


def test_policy_blocks_reencoded_metadata_claims_with_identical_content():
    import hashlib
    from research.fit_policy import fit
    from research.report_policy import evaluate
    rows=[]
    for split in ('validation','test'):
        for i,label in enumerate(('human','human','ai','ai')):
            rows.append(dict(label=label,score=.1 if label=='human' else .9,split=split,speaker_id=f'{split}-s{i}',source_id=f'{split}-r{i}',language='en',latency_ms=1.,checkpoint_sha256=SHA256,score_protocol=SCORE_PROTOCOL,audio_sha256=hashlib.sha256(f'{split}{i}'.encode()).hexdigest()))
    validation=rows[:4];test=rows[4:]
    policy=pipeline.fit(validation,checkpoint_sha256=SHA256,scores_sha256='a'*64,domain='unit fixture only',min_count=2,min_speakers=2)
    test[0]['audio_sha256']=validation[0]['audio_sha256']
    with pytest.raises(ValueError,match='content leakage'):evaluate(test,policy,SHA256)


@pytest.mark.parametrize('generator',['unknown','unspecified','dataset:unknown','synthetic','tts','unknown-generator'])
def test_unknown_generator_cannot_count_as_unseen(tmp_path,generator):
    audio=tmp_path/'clip.wav';audio.write_bytes(b'unique fixture audio')
    row=dict(audio_path=str(audio),label='ai',speaker_id='s',source_id='r',generator=generator,split='test')
    with pytest.raises(ValueError,match='known synthesis-system identifiers'):
        pipeline.audit_recordings([row],held_out_generators=True)


def test_sorted_threshold_search_matches_inclusive_reference():
    from research.fit_policy import fit
    rows=[]
    for i,(label,score) in enumerate([('human',.1),('human',.2),('human',.8),('ai',.2),('ai',.8),('ai',.9)]):
        rows.append(dict(label=label,score=score,split='validation',speaker_id=f's{i}',source_id=f'r{i}',language='en'))
    policy=fit(rows,checkpoint_sha256=SHA256,scores_sha256='a'*64,domain='numerical fixture only',max_error=.34,min_count=2,min_speakers=2)
    human=[r['score'] for r in rows if r['label']=='human'];ai=[r['score'] for r in rows if r['label']=='ai']
    lows=[s for s in sorted(set(human)) if sum(a<=s for a in ai)/len(ai)<=.34]
    highs=[s for s in sorted(set(ai)) if sum(h>=s for h in human)/len(human)<=.34]
    assert policy['human_max']==max(lows) and policy['ai_min']==min(highs)

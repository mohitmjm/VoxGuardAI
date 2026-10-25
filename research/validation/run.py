"""Audit -> score -> validation-only policy -> untouched test/robustness report.

Every run uses a new output directory. Manifest partitions are never inferred or
changed. Scores and fit failures are retained; no policy is installed in the app.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from research.evaluate import manifest, audit, score, curves
from research.fit_policy import fit
from research.report_policy import evaluate
from ml.constants import SHA256, COMMIT
from ml.policy import SCORE_PROTOCOL, validate_artifact
from research.validation.uncertainty import bootstrap, generator_breakdown


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def write_csv(path, rows):
    with Path(path).open('x', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def audit_recordings(rows, held_out_generators=False):
    result = audit(rows, held_out_generators)
    seen_paths, seen_hashes, seen_ids = set(), set(), set()
    enriched = []
    for row in rows:
        path = str(Path(row['audio_path']).resolve())
        checksum = digest(path)
        identity = row.get('recording_id') or 'sha256:'+checksum
        if not isinstance(identity,str) or not identity.strip() or identity != identity.strip():
            raise ValueError('Recording identifiers must be nonempty without surrounding whitespace')
        if path in seen_paths or checksum in seen_hashes or identity in seen_ids:
            raise ValueError('Repeated recording paths/content would inflate sample counts; review the original protocol before proceeding')
        seen_paths.add(path); seen_hashes.add(checksum); seen_ids.add(identity)
        enriched.append({**row, 'recording_id':identity, 'audio_sha256': checksum})
    return enriched, result


def ranking(rows):
    if not rows:
        raise ValueError('Empty evaluation split')
    y = [int(row['label'] == 'ai') for row in rows]
    scores = [float(row['score']) for row in rows]
    _, tpr, fpr = curves(y, scores)
    difference = fpr - (1-tpr)
    i = int(np.flatnonzero(difference >= 0)[0])
    alpha = 0 if i == 0 else -difference[i-1]/(difference[i]-difference[i-1])
    eer = float(fpr[0] if i == 0 else fpr[i-1]+alpha*(fpr[i]-fpr[i-1]))
    latency = np.asarray([float(row['latency_ms']) for row in rows])
    if not np.isfinite(latency).all() or (latency < 0).any():
        raise ValueError('Invalid measured latency')
    return {'sample_count': len(rows), 'class_counts': {label:sum(r['label']==label for r in rows) for label in ('human','ai')},
            'speaker_count': len({r['speaker_id'] for r in rows}),
            'auroc': float(np.sum(np.diff(fpr)*(tpr[:-1]+tpr[1:])/2)), 'eer': eer,
            'latency_ms': {'median':float(np.median(latency)), 'p95':float(np.quantile(latency,.95))},
            'binary_decision_metrics': None, 'warning': 'Ranking is descriptive; no decision threshold is selected using these test scores.'}


def environment():
    try:
        revision = subprocess.check_output(['git','rev-parse','HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git','status','--porcelain'], text=True))
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    versions = {}
    for name in ('torch','numpy','soundfile','fastapi'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return {'git_revision':revision, 'git_dirty':dirty, 'platform':platform.platform(), 'python':platform.python_version(),
            'dependencies':versions, 'cpu_count':os.cpu_count(), 'torch_threads':os.getenv('VOXGUARD_TORCH_THREADS','2'),
            'device':os.getenv('VOXGUARD_DEVICE','cpu'), 'model_source_commit':COMMIT, 'checkpoint_sha256':SHA256,'score_protocol':SCORE_PROTOCOL,
            'source_sha256':{str(Path(__file__).name):digest(__file__), 'evaluate.py':digest(Path(__file__).parents[1]/'evaluate.py'), 'model_service.py':digest(Path(__file__).parents[2]/'ml/service.py'), 'model_config.json':digest(Path(__file__).parents[2]/'ml/aasist_config.json')}}


def run(manifest_path, output, *, dataset, conditions, fit_policy=False, held_out_generators=False,
        robustness=False, seed=42, max_error=.01, min_count=100, min_speakers=10, bootstrap_replicates=1000):
    if not isinstance(dataset, str) or not dataset.strip() or not isinstance(conditions, str) or not conditions.strip():
        raise ValueError('Explicit dataset and evaluation conditions are required')
    if not isinstance(bootstrap_replicates,int) or isinstance(bootstrap_replicates,bool) or not 100 <= bootstrap_replicates <= 10000:
        raise ValueError('Use 100..10000 bootstrap replicates')
    source, output = Path(manifest_path), Path(output)
    rows, audit_result = audit_recordings(manifest(source), held_out_generators)
    test = [row for row in rows if row['split']=='test']
    if {r['label'] for r in test} != {'human','ai'}:
        raise ValueError('Untouched test partition must contain both classes')
    # mkdir is exclusive, so existing runs and their evidence are never overwritten.
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    metadata = {'dataset':dataset,'conditions':conditions,'manifest_sha256':digest(source),'audit':audit_result,
                'environment':environment(),'seed':seed,'held_out_generators_requested':held_out_generators,
                'limitations':['Original partition/speaker/source/generator metadata remains the dataset custodian responsibility.',
                               'Byte hashes cannot detect re-encoded source leakage or undisclosed pretrained overlap.',
                               'This runner does not establish that the selected dataset is representative.',
                               'Research decoding caps recordings at120seconds; use documented shorter benchmark segments.',
                               'No production policy is enabled automatically.']}
    write_json(output/'provenance.json', metadata)
    write_json(output/'protocol.json', {
        'frozen_before_inference': True, 'dataset':dataset, 'conditions':conditions,
        'manifest_sha256':metadata['manifest_sha256'],
        'recordings':[{'audio_sha256':r['audio_sha256'],'recording_id':r.get('recording_id'),
                       'speaker_id':r['speaker_id'],'source_id':r['source_id'],'generator':r['generator'],
                       'language':r['language'],'label':r['label'],'split':r['split']} for r in rows],
        'checkpoint_sha256':SHA256,'score_protocol':SCORE_PROTOCOL,'seed':seed,
        'selection':'All supplied manifest validation/test rows; no inference of representativeness or resplitting',
        'fit_requested':fit_policy, 'held_out_generators':held_out_generators,
        'threshold_objective':{'max_empirical_error_per_class':max_error,'minimum_recordings_per_class':min_count,'minimum_speakers_per_class':min_speakers},
        'robustness_requested':robustness,'uncertainty_replicates':bootstrap_replicates,
        'minimum_uncertainty_clusters':10})
    frozen_protocol_digest = digest(output/'protocol.json')
    try:
        selected = [r for r in rows if r['split'] in {'validation','test'}]
        scored = list(score(selected))
        if len(scored) != len(selected):
            raise ValueError('Scored recording count differs from the frozen protocol')
        identity_fields = ('recording_id','audio_sha256','label','speaker_id','source_id','generator','language','split')
        if any(any(result.get(field) != original.get(field) for field in identity_fields) for original,result in zip(selected,scored)):
            raise ValueError('Scored recording identity or label differs from the frozen protocol')
        if digest(output/'protocol.json') != frozen_protocol_digest:
            raise ValueError('Protocol changed after freeze')
        if digest(source) != metadata['manifest_sha256']:
            raise ValueError('Manifest changed after protocol freeze')
        write_csv(output/'scores.csv',scored)
        for row in selected:
            if digest(row["audio_path"]) != row["audio_sha256"]:
                raise ValueError("A recording changed after the audit; evidence is invalid")
        validation = [r for r in scored if r['split']=='validation']
        test_scores = [r for r in scored if r['split']=='test']
        if validation:
            write_csv(output/'validation_scores.csv',validation)
        write_csv(output/'test_scores.csv',test_scores)
        result = {'dataset':dataset,'conditions':conditions,'test_ranking':ranking(test_scores),
                  'protocol_sha256':digest(output/'protocol.json'),
                  'validation_count':len(validation),'policy_status':'not_requested','policy':None,'test_decisions':None}
        if fit_policy:
            try:
                policy = fit(validation,checkpoint_sha256=SHA256,scores_sha256=digest(output/'validation_scores.csv') if validation else '0'*64,
                             domain=conditions,max_error=max_error,min_count=min_count,min_speakers=min_speakers)
                write_json(output/'policy.json',policy)
                result.update(policy_status='fitted_research_only', policy=policy,
                              test_decisions=evaluate(test_scores,policy,SHA256))
            except ValueError as exc:
                result.update(policy_status='not_established', policy_failure=str(exc))
        active_policy = validate_artifact(result['policy'],SHA256) if result['policy'] else None
        result['uncertainty'] = bootstrap(test_scores, policy=active_policy, replicates=bootstrap_replicates, seed=seed)
        result['generator_comparisons'] = generator_breakdown(test_scores)
        if robustness:
            from experiments.robustness import observations
            # Test-only transforms reuse the same untouched recordings, never validation.
            variants = list(observations([r for r in rows if r['split']=='test'], seed=seed))
            write_csv(output/'robustness_scores.csv',variants)
            result['robustness'] = {}
            for condition in sorted({r['condition'] for r in variants}):
                subset = [r for r in variants if r['condition']==condition]
                entry = {'ranking':ranking(subset), 'uncertainty':bootstrap(subset,policy=active_policy,replicates=bootstrap_replicates,seed=seed)}
                if result['policy'] is not None:
                    entry['fixed_clean_validation_policy'] = evaluate(subset,result['policy'],SHA256)
                result['robustness'][condition] = entry
            result['robustness_observations'] = len(variants)
            result['robustness_warning'] = 'Conditions are paired repeated observations, not independent recordings. No thresholds are refitted on transformed test data.'
        result['elapsed_seconds'] = time.perf_counter()-started
        result['status'] = 'completed'
        write_json(output/'summary.json',result)
        return result
    except Exception as exc:
        write_json(output/'failure.json',{'status':'failed','error_type':type(exc).__name__,'detail':str(exc),'elapsed_seconds':time.perf_counter()-started})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--dataset',required=True);parser.add_argument('--conditions',required=True)
    parser.add_argument('--fit-policy',action='store_true');parser.add_argument('--held-out-generators',action='store_true')
    parser.add_argument('--robustness',action='store_true');parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--bootstrap-replicates',type=int,default=1000)
    parser.add_argument('--max-error',type=float,default=.01);parser.add_argument('--min-count',type=int,default=100);parser.add_argument('--min-speakers',type=int,default=10)
    args=parser.parse_args()
    result=run(args.manifest,args.output,dataset=args.dataset,conditions=args.conditions,fit_policy=args.fit_policy,
               held_out_generators=args.held_out_generators,robustness=args.robustness,seed=args.seed,
               max_error=args.max_error,min_count=args.min_count,min_speakers=args.min_speakers,bootstrap_replicates=args.bootstrap_replicates)
    print(json.dumps({'status':result['status'],'test_count':result['test_ranking']['sample_count'],'policy_status':result['policy_status']}))

if __name__=='__main__':main()

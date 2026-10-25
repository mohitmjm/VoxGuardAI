"""Train frozen-feature ablations and evaluate untouched test rows against aligned AASIST.

Research only. Separate train/validation/test data and original identities are
mandatory. No benchmark is executed by merely installing this module.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
import torch

from experiments.hybrid import ABLATIONS, architecture, digest, inputs, load_features, save_checkpoint, train
from ml.constants import SHA256, ROOT
from ml.policy import SCORE_PROTOCOL
from research.evaluate import curves, metrics


def validation_threshold(labels,scores):
    thresholds,tpr,fpr=curves(labels,scores)
    finite=np.isfinite(thresholds)
    candidates=thresholds[finite]
    return float(candidates[np.argmin(np.abs(fpr[finite]-(1-tpr[finite])))])


def predict(checkpoint,data,mask):
    model=architecture(checkpoint['input_dim']);model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
    x=inputs(data,checkpoint['ablation'])[mask]
    x=torch.tensor(x,dtype=torch.float32)
    mean,scale=checkpoint['mean'],checkpoint['scale']
    with torch.inference_mode():
        # Warm up kernels before timing head-only inference; no labels participate.
        model((x[:1]-mean)/scale)
        start=time.perf_counter();scores=torch.sigmoid(model((x-mean)/scale).squeeze(1)).numpy();elapsed=time.perf_counter()-start
    return scores,elapsed*1000/len(x)


def aligned_baseline(rows,data):
    needed=np.flatnonzero(data['split']!='train')
    by_id={}
    for row in rows:
        key=row.get('recording_id')
        if not key or key in by_id:raise ValueError('Unique explicit baseline recording IDs required')
        by_id[key]=row
    scores=np.full(len(data['label']),np.nan)
    latencies=[]
    for i in needed:
        row=by_id.get(str(data['recording_id'][i]))
        if row is None:raise ValueError('Baseline is missing an aligned validation/test recording')
        if row.get('audio_sha256')!=str(data['audio_sha256'][i]) or row.get('split')!=str(data['split'][i]):
            raise ValueError('Baseline recording content or partition differs from features')
        if any(row.get(field) != str(data[field][i]) for field in ('speaker_id','source_id')):
            raise ValueError('Baseline speaker/source provenance differs from features')
        if row.get('label')!=('ai' if data['label'][i] else 'human'):
            raise ValueError('Baseline label differs from feature label')
        if row.get('checkpoint_sha256')!=SHA256 or row.get('score_protocol')!=SCORE_PROTOCOL:
            raise ValueError('Baseline must be the frozen AASIST checkpoint and score protocol')
        value=float(row['score'])
        if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid baseline score')
        scores[i]=value
        latency=float(row['latency_ms'])
        if not math.isfinite(latency) or latency<0:raise ValueError('Invalid baseline latency')
        if data['split'][i]=='test':latencies.append(latency)
    return scores,float(np.median(latencies))


def size_accounting(checkpoint, extractor):
    """Never describe a frozen-encoder system using only its small head count."""
    uses_encoder = checkpoint['ablation'] != 'acoustic'
    encoder = extractor.get('parameter_count') if uses_encoder else 0
    if encoder is not None and (not isinstance(encoder, int) or isinstance(encoder, bool) or encoder < 0):
        raise ValueError('Invalid frozen encoder parameter count in feature provenance')
    head = checkpoint['parameter_count']
    return {'head_parameter_count': head, 'frozen_encoder_parameter_count': encoder,
            'total_parameter_count': head + encoder if encoder is not None else None,
            'parameter_scope': 'parameter_count/parameter_bytes refer to the head only; missing encoder counts are unknown, not zero',
            'frozen_encoder_checkpoint_bytes': extractor.get('checkpoint_size_bytes') if uses_encoder else 0}


def baseline_parameter_count():
    # Count the declared official architecture, without loading weights or audio.
    from ml.vendor.aasist import Model
    config = json.loads((ROOT/'aasist_config.json').read_text())['model_config']
    with torch.random.fork_rng(devices=[]):
        return sum(parameter.numel() for parameter in Model(config).parameters())


def comparison(features,baseline,output,*,dataset,seed=42,epochs=100,patience=10,min_per_class=100):
    data,metadata=load_features(features,min_per_class)
    with Path(baseline).open(newline='',encoding='utf8') as f: baseline_rows=list(csv.DictReader(f))
    baseline_scores,baseline_latency=aligned_baseline(baseline_rows,data)
    validation,test=data['split']=='validation',data['split']=='test'
    y=data['label']
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    baseline_threshold=validation_threshold(y[validation],baseline_scores[validation])
    reports={'aasist':{'metrics':metrics(y[test],baseline_scores[test],baseline_threshold),
                        'validation_threshold':baseline_threshold,'latency_ms':baseline_latency,
                        'latency_scope':'recorded full AASIST model-call median; excludes decoder',
                        'checkpoint_sha256':SHA256,
                        'parameter_count':baseline_parameter_count(),
                        'parameter_scope':'official full AASIST architecture'}}
    for ablation in ABLATIONS:
        checkpoint=train(data,ablation=ablation,seed=seed,epochs=epochs,patience=patience,min_per_class=min_per_class)
        checkpoint['feature_sha256']=digest(features);checkpoint['extractor']=metadata['extractor']
        validation_scores,_=predict(checkpoint,data,validation)
        threshold=validation_threshold(y[validation],validation_scores)
        # Fixed only after validation; test labels cannot select this threshold.
        checkpoint['threshold']=threshold
        destination=output/(ablation+'.pt');save_checkpoint(checkpoint,destination)
        test_scores,latency=predict(checkpoint,data,test)
        reports[ablation]={'metrics':metrics(y[test],test_scores,threshold),'validation_threshold':threshold,
            **size_accounting(checkpoint,metadata['extractor']),
            'parameter_count':checkpoint['parameter_count'],'parameter_bytes':checkpoint['parameter_bytes'],'checkpoint_bytes':destination.stat().st_size,
            'checkpoint_sha256':digest(destination),'best_epoch':checkpoint['epoch'],'epochs_executed':checkpoint['epochs_executed'],
            'latency_ms':latency,'latency_scope':'warm batched CPU head-only milliseconds per recording; EXCLUDES Wav2Vec2/LFCC extraction; not comparable to end-to-end detector latency',
            'test_scores':[{'recording_id':str(data['recording_id'][i]),'score':float(s)} for i,s in zip(np.flatnonzero(test),test_scores)]}
    result={'dataset':dataset,'seed':seed,'counts':{s:int(np.sum(data['split']==s)) for s in ('train','validation','test')},
        'features_sha256':digest(features),'baseline_scores_sha256':digest(baseline),'extractor':metadata['extractor'],
        'feature_extraction_timing':metadata.get('timing'),
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','threads':torch.get_num_threads()},
        'methods':reports,'limitations':['Research comparison only; no model registered in production.',
        'Thresholds selected exclusively on validation; binary outputs are not calibrated probabilities.',
        'Minimum counts do not guarantee representative sampling or generalization; inspect dataset rights and protocol.',
        'Run multiple prespecified seeds, report all seeds, and audit unknown pretraining overlap.',
        'Head-only latency omits the substantial frozen speech encoder and LFCC cost.']}
    with (output/'comparison.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('features',type=Path);p.add_argument('--baseline-scores',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=10);p.add_argument('--min-per-class',type=int,default=100)
    a=p.parse_args();torch.set_num_threads(2)
    comparison(a.features,a.baseline_scores,a.output,dataset=a.dataset,seed=a.seed,epochs=a.epochs,patience=a.patience,min_per_class=a.min_per_class)
    print('Experimental comparison recorded. No reliability, novelty or production-validity claim is implied.')

if __name__=='__main__':main()

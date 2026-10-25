"""Research-only frozen-feature heads. No production detector registration.

Requires independently audited train/validation/test features and provenance from
experiments.extract_features. Numerical unit fixtures are not research results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ml.representations import WAV2VEC2_SHA256

ABLATIONS = ('learned', 'acoustic', 'hybrid')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def validate_features(data, min_per_class=100):
    required = {'learned','acoustic','label','split','speaker_id','source_id','recording_id','audio_sha256'}
    if not required.issubset(data):
        raise ValueError('Missing feature identities or arrays; regenerate with experiments.extract_features')
    y, splits = np.asarray(data['label']), np.asarray(data['split'])
    n = len(y)
    if y.ndim != 1 or not n or not np.isin(y, [0,1]).all():
        raise ValueError('Labels must be a nonempty binary vector')
    if splits.ndim != 1 or min_per_class < 1 or set(splits.tolist()) != {'train','validation','test'}:
        raise ValueError('Separate train, validation and test partitions are required')
    for name in required:
        if len(data[name]) != n:
            raise ValueError('All feature arrays must have matching lengths')
    for name in ('learned','acoustic'):
        x = np.asarray(data[name])
        if x.ndim != 2 or not x.shape[1] or not np.isfinite(x).all():
            raise ValueError('Features must be finite two-dimensional arrays')
    for name in ('speaker_id','source_id','recording_id','audio_sha256'):
        values = np.asarray(data[name])
        if values.ndim != 1 or any(not str(v).strip() or str(v) != str(v).strip() or str(v).strip().lower() in {'unknown','-','n/a'} for v in values):
            raise ValueError('Explicit nonempty identity metadata is required')
        seen = {}
        for value, split in zip(values, splits):
            if value in seen and (seen[value] != split or name in {'recording_id','audio_sha256'}):
                raise ValueError(f'{name} leaks or duplicates recordings')
            seen[value] = split
    import re
    if any(not re.fullmatch('[0-9a-f]{64}', str(x)) for x in data['audio_sha256']):
        raise ValueError('Valid audio content digests are required')
    for split in ('train','validation','test'):
        for label in (0,1):
            if np.sum((splits == split) & (y == label)) < min_per_class:
                raise ValueError('Insufficient recordings per split/class; do not reuse pilot test clips for training')
    return data


def load_features(path, min_per_class=100):
    path = Path(path)
    sidecar = json.loads(path.with_suffix(path.suffix+'.json').read_text())
    if not isinstance(sidecar, dict) or sidecar.get('feature_sha256') != digest(path):
        raise ValueError('Feature file/extractor provenance is missing or mismatched')
    extractor = sidecar.get('extractor')
    if not isinstance(extractor, dict) or extractor.get('checkpoint_sha256') != WAV2VEC2_SHA256:
        raise ValueError('Feature extractor provenance must match the documented official Wav2Vec2 checkpoint')
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    validate_features(data, min_per_class)
    return data, sidecar


def inputs(data, ablation):
    if ablation not in ABLATIONS:
        raise ValueError('Unknown ablation')
    return np.concatenate([data['learned'],data['acoustic']],axis=1) if ablation=='hybrid' else data[ablation]


def architecture(dimension):
    return nn.Sequential(nn.Linear(dimension,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1))


def train(data, *, ablation='hybrid', seed=42, epochs=100, patience=10, min_per_class=100):
    validate_features(data,min_per_class)
    if epochs < 1 or patience < 1:
        raise ValueError('Epochs and patience must be positive')
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    x = inputs(data,ablation)
    training, validation = data['split']=='train', data['split']=='validation'
    mean=x[training].mean(0); scale=np.maximum(x[training].std(0),1e-6)
    # Test features do not enter normalization, training, early stopping or thresholds.
    xtrain=torch.tensor((x[training]-mean)/scale,dtype=torch.float32)
    xval=torch.tensor((x[validation]-mean)/scale,dtype=torch.float32)
    ytrain=torch.tensor(data['label'][training],dtype=torch.float32)
    yval=torch.tensor(data['label'][validation],dtype=torch.float32)
    model=architecture(x.shape[1]); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    criterion=nn.BCEWithLogitsLoss(); best=float('inf'); stale=0; best_state=None
    for epoch in range(epochs):
        model.train();optimizer.zero_grad();loss=criterion(model(xtrain).squeeze(1),ytrain);loss.backward();optimizer.step()
        model.eval()
        with torch.inference_mode(): validation_loss=float(criterion(model(xval).squeeze(1),yval))
        if not np.isfinite(validation_loss): raise ValueError('Training produced nonfinite loss')
        if validation_loss < best:
            best=validation_loss;best_epoch=epoch;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};stale=0
        else: stale+=1
        if stale>=patience: break
    return {'state_dict':best_state,'mean':torch.tensor(mean),'scale':torch.tensor(scale),
            'ablation':ablation,'seed':seed,'epoch':best_epoch,'epochs_executed':epoch+1,'input_dim':x.shape[1],
            'validation_loss':best,'parameter_count':sum(p.numel() for p in model.parameters()),
            'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
            'training_recording_ids':data['recording_id'][training].tolist(),
            'validation_recording_ids':data['recording_id'][validation].tolist(),
            'min_recordings_per_split_class':min_per_class,'experimental':True}


def save_checkpoint(checkpoint, output):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f: torch.save(checkpoint,f)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('features',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ablation',choices=ABLATIONS,default='hybrid');p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=10);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--min-per-class',type=int,default=100)
    a=p.parse_args();data,metadata=load_features(a.features,a.min_per_class)
    checkpoint=train(data,ablation=a.ablation,seed=a.seed,epochs=a.epochs,patience=a.patience,min_per_class=a.min_per_class)
    checkpoint['feature_sha256']=digest(a.features);checkpoint['extractor']=metadata['extractor']
    save_checkpoint(checkpoint,a.output)
    print('Experimental head saved. No test result, probability calibration, or production validity is implied.')

if __name__=='__main__':main()

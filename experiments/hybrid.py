"""Experimental fusion classifier on precomputed speech embeddings and LFCC features.
No production registration. Features must be prepared with frozen, documented encoders.
NPZ fields: learned [N,D], acoustic [N,F], label [N], split [N], speaker_id [N], source_id [N].
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('features',type=Path); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ablation',choices=['hybrid','learned','acoustic'],default='hybrid')
    p.add_argument('--epochs',type=int,default=30); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.epochs < 1: raise ValueError('epochs must be positive')
    torch.manual_seed(a.seed)
    data=np.load(a.features,allow_pickle=False)
    split=data['split']; y=data['label']
    if not set(np.unique(split)).issubset({'train','validation','test'}): raise ValueError('Invalid split')
    for field in ['speaker_id','source_id']:
        for value in np.unique(data[field]):
            if len(np.unique(split[data[field]==value]))>1: raise ValueError(f'{field} leaks across splits')
    x=np.concatenate([data['learned'],data['acoustic']],axis=1) if a.ablation=='hybrid' else data[a.ablation]
    train=split=='train'; val=split=='validation'
    if not train.any() or not val.any() or set(y[train].tolist())!={0,1}: raise ValueError('Both training classes and validation data required')
    if not np.isfinite(x).all(): raise ValueError('Features must be finite')
    # Normalization is fit exclusively on training data.
    mean=x[train].mean(0); scale=np.maximum(x[train].std(0),1e-6)
    x=torch.tensor((x-mean)/scale,dtype=torch.float32); target=torch.tensor(y,dtype=torch.float32)
    model=nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    criterion=nn.BCEWithLogitsLoss(); best=float('inf')
    for epoch in range(a.epochs):
        model.train(); optimizer.zero_grad(); loss=criterion(model(x[train]).squeeze(1),target[train]); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): validation=float(criterion(model(x[val]).squeeze(1),target[val]))
        if validation<best:
            best=validation
            torch.save({'state_dict':model.state_dict(),'mean':torch.tensor(mean),'scale':torch.tensor(scale),'ablation':a.ablation,'seed':a.seed,'epoch':epoch,'input_dim':x.shape[1],'validation_loss':best,'experimental':True},a.output)
    print('Experimental fusion checkpoint saved. No held-out performance claim is made.')

if __name__=='__main__': main()

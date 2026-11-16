"""Применение сохранённого адаптера без обучения, label позднего хоста не используется."""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import Window,read_window,signature
from .neighbors import apply_residual
from .run_neighbors import transfer_late


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True,type=Path)
    ap.add_argument('--checkpoint',required=True,type=Path)
    ap.add_argument('--flows',required=True,type=Path)
    ap.add_argument('--start',required=True,type=float)
    ap.add_argument('--end',required=True,type=float)
    ap.add_argument('--day',required=True,type=int)
    ap.add_argument('--cache',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--neighbor-checkpoint',type=Path)
    ap.add_argument('--source-output',type=Path)
    args=ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if bool(args.neighbor_checkpoint)!=bool(args.source_output):
        ap.error('--neighbor-checkpoint требует --source-output для изменённых source-векторов')
    if args.source_output and args.source_output.exists(): raise FileExistsError(args.source_output)
    before=signature(args.source)
    source=pd.read_csv(args.source,index_col=0)
    source.index=source.index.astype(str)
    model=joblib.load(args.checkpoint)
    if not hasattr(model,'hosts') or model.hosts!=list(source.index):
        raise ValueError('checkpoint не соответствует порядку/составу source-хостов')
    if model.source_sha256!=before: raise ValueError('source-файл отличается от замороженного')
    np.testing.assert_array_equal(model.source,source.to_numpy())
    obs=read_window(Window('inference',args.day,args.start,args.end,str(args.flows)),model.hosts,args.cache)
    later=model.predict(obs,model.selected['variant'],model.selected['temperature'])
    if args.neighbor_checkpoint:
        head=joblib.load(args.neighbor_checkpoint)
        if head['source_sha256']!=before: raise ValueError('голова соседей от другого source')
        updated=apply_residual(source.to_numpy(),head['state'],head['selected_strength']) if head['state'] else source.to_numpy()
        later=transfer_late(later,source.to_numpy(),updated)
        args.source_output.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(updated,index=source.index,columns=source.columns).to_csv(args.source_output,index_label='host_id')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(later,index=source.index,columns=source.columns).to_csv(args.output,index_label='host_id')
    if signature(args.source)!=before: raise RuntimeError('source изменён')
    print(args.output)


if __name__=='__main__':
    main()

"""Raw flow features only; no pretrained encoder or codebook is used."""
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from evaluation.knn_consistency import build_reference_features
from .data import Observation


def behavior(window, hosts, cache, fitted=None,include_self=False):
    """Project's actual behavioral features; columns/scalers fitted on source only."""
    path = Path(window.flows)
    stamp = (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns,
             window.start, window.end, hosts, joblib.hash(fitted), 'behavior-v1-with-self' if include_self else 'behavior-v1')
    saved = Path(cache) / ('behavior-' + hashlib.sha256(repr(stamp).encode()).hexdigest() + '.joblib')
    if saved.exists():
        return joblib.load(saved)
    columns = ['src','dst','dst_port','proto','bytes','packets','ts','duration']
    parts = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250_000,
                             dtype={c:'str' for c in ('src','dst','dst_port','proto')}):
        mask=(chunk.ts>=window.start)&(chunk.ts<window.end)&(chunk.ts+chunk.duration<window.end)
        if not include_self:mask&=chunk.src!=chunk.dst
        part=chunk.loc[mask]
        parts.append(part)
    flows = pd.concat(parts, ignore_index=True)
    for column in ('src','dst','dst_port','proto'):
        flows[column] = flows[column].astype('category')
    features, state = build_reference_features(flows, hosts=hosts, fitted=fitted, return_fitted=True)
    result = (features.to_numpy(dtype=np.float32), state)
    saved.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, saved, compress=3)
    return result


def with_context(observation,hosts,cache):
    """Joint peer/protocol/port traffic tokens. Own ID is grouping only; self flows are excluded."""
    from collections import Counter
    window=observation.window;path=Path(window.flows);stat=path.stat()
    stamp=(str(path.resolve()),stat.st_size,stat.st_mtime_ns,window.start,window.end,list(hosts),'joint-context-v1')
    saved=Path(cache)/('context-'+hashlib.sha256(repr(stamp).encode()).hexdigest()+'.joblib')
    if saved.exists():extra=joblib.load(saved)
    else:
        index={h:i for i,h in enumerate(hosts)};extra=[Counter() for _ in hosts]
        columns=['src','dst','proto','dst_port','ts','duration']
        for chunk in pd.read_csv(path,usecols=columns,chunksize=250_000,
                                 dtype={c:str for c in ['src','dst','proto','dst_port']}):
            f=chunk.loc[(chunk.ts>=window.start)&(chunk.ts<window.end)&
                        (chunk.ts+chunk.duration<window.end)&(chunk.src!=chunk.dst)]
            for direction,endpoint,peer in [('o','src','dst'),('i','dst','src')]:
                selected=f.loc[f[endpoint].isin(index)]
                grouped=selected.groupby([endpoint,peer,'proto','dst_port'],sort=False).size()
                for (host,partner,proto,port),count in grouped.items():
                    extra[index[host]][f'c:{direction}:{partner}|{proto}/{port}']+=int(count)
        saved.parent.mkdir(parents=True,exist_ok=True);joblib.dump(extra,saved,compress=3)
    counts=[]
    for original,joint in zip(observation.counts,extra):
        row=dict(original);row.update(joint);counts.append(row)
    return Observation(window,counts,observation.numeric,observation.events)

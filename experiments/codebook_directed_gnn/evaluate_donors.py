"""Полная оценка изменённого source и фиксированного late на development13."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .base import sha
from experiments.temporal_graph_mixer.common.evaluate import geometry,with_identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','training','late','reference','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    f=json.loads((a.training/'completed.json').read_text())
    if sha(a.training/'source.csv')!=f['source_sha256']:raise ValueError('Source изменён')
    source=pd.read_csv(a.training/'source.csv',index_col=0)
    late=pd.read_csv(a.late,index_col=0);a.output.mkdir(parents=True)
    result=with_identity(source,late,a.reference,geometry(source,a.reference,
        a.data/'late_test/reference/validation',a.output/'cache'))
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result['score'])


if __name__=='__main__':main()
